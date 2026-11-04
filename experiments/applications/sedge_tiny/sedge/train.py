import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model.classifier import SpeechCommandsClassifier
from .protocol import (
    BATCH_SIZE,
    EPOCHS,
    OPTIMIZER_STEPS,
    SEEDS,
    STEPS_PER_EPOCH,
    data_directory,
    run_name,
    sha256,
    validate_case,
)
from .scheduler import CosineAnnealingWarmupRestarts
from .sugar import replace_activations


def build_model():
    return SpeechCommandsClassifier(
        input_size=1,
        classes=35,
        hidden_sizes=[32, 16, 8],
        output_sizes=[8, 32, 64],
        ZeroOrderHoldRegularization=[],
        input_bias=True,
        bias_init="uniform",
        output_bias=True,
        norm=True,
        norm_type="bn",
        complex_output=False,
        B_C_init="orthogonal",
        stability="abs",
        trainable_SkipLayer=True,
        act="LeakyRELu",
        dropout=0.0,
    )


def build_optimizer(model):
    names = ["B", "C", "Lambda", "log_step"]
    return torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(key in n for key in names)
                ],
                "lr": 0.002,
                "weight_decay": 0,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(key in n for key in names)
                ],
                "lr": 0.008,
                "weight_decay": 0.04,
            },
        ]
    )


def build_scheduler(optimizer):
    # The original scalar schedule assigns the same LR to both groups.
    return CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=OPTIMIZER_STEPS,
        cycle_mult=1.0,
        max_lr=0.008,
        min_lr=0,
        warmup_steps=100,
        gamma=1,
    )


def decoder_regularization(weights):
    orth = weights @ weights.T
    orth = orth * (1 - torch.eye(weights.shape[0], device=weights.device))
    return 0.1 * torch.mean(orth**2).sqrt()


def train_one_epoch(model, criterion, optimizer, loader, scheduler):
    running_loss = 0.0
    correct = 0
    model.train()
    device = next(model.parameters()).device
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(True)
        outputs = model(inputs)
        loss_weight = criterion(outputs, labels)
        correct += outputs.argmax(dim=-1).squeeze().eq(labels).sum().item()
        loss = loss_weight + decoder_regularization(model.decoder.weight)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss_weight.item()
    return running_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, criterion, loader):
    losses = []
    correct = 0
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            correct += outputs.argmax(dim=-1).squeeze().eq(labels).sum().item()
    return np.sum(losses) / len(loader), correct / len(loader.dataset)


def write_json(path, value):
    with Path(path).open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def initialization_path(data_dir, seed):
    name = f"initialization/seed_{seed}.pt"
    path = Path(data_dir) / name
    provenance = json.loads((Path(data_dir) / "provenance.json").read_text())
    digest = provenance["initialization"]["files"][name]
    if sha256(path) != digest:
        raise ValueError(f"Initialization checksum mismatch: {name}")
    return path, digest


def run(args):
    import torchaudio

    from .dataset import SpeechCommands

    validate_case(args.surrogate, args.alpha, args.seed)
    if args.workers < 1 or args.cpu_threads < 1:
        raise ValueError("Workers and CPU threads must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The published training protocol requires a CUDA GPU")
    initial_path, initial_digest = initialization_path(args.data_dir, args.seed)
    destination = args.output_dir / run_name(args.surrogate, args.alpha, args.seed)
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"{destination.name}: loading Speech Commands", flush=True)
    train_set = SpeechCommands(args.dataset_dir, "training")
    normalization = (train_set.mean, train_set.std)
    test_set = SpeechCommands(args.dataset_dir, "testing", normalization)
    valid_set = SpeechCommands(args.dataset_dir, "validation", normalization)
    dataset_ordering = {
        split: hashlib.sha256(
            "\n".join(
                str(path.relative_to(args.dataset_dir)) for path in dataset.paths
            ).encode()
        ).hexdigest()
        for split, dataset in (
            ("training", train_set),
            ("testing", test_set),
            ("validation", valid_set),
        )
    }
    provenance = json.loads((args.data_dir / "provenance.json").read_text())
    if dataset_ordering != provenance["dataset"]["ordering_sha256"]:
        raise ValueError(
            "Speech Commands ordering differs from the published experiment"
        )
    loader_options = dict(
        batch_size=BATCH_SIZE,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_options)
    valid_loader = DataLoader(
        valid_set, shuffle=False, drop_last=False, **loader_options
    )
    if len(train_loader) != STEPS_PER_EPOCH:
        raise ValueError("Unexpected number of optimizer steps per epoch")

    # Construction consumes the original RNG sequence before archived weights are loaded.
    model = build_model()
    model.load_state_dict(
        torch.load(initial_path, map_location="cpu", weights_only=True), strict=True
    )
    model = model.to(device)
    model = replace_activations(model, args.surrogate, args.alpha)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    configuration = {
        "surrogate": args.surrogate,
        "alpha": args.alpha,
        "seed": args.seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "nominal_group_learning_rates": [0.002, 0.008],
        "weight_decay": [0, 0.04],
        "scheduler": {
            "max_lr": 0.008,
            "min_lr": 0,
            "warmup_steps": 100,
            "first_cycle_steps": OPTIMIZER_STEPS,
        },
        "decoder_regularization": 0.1,
        "augmentation": None,
        "dropout": 0,
        "batch_normalization": True,
        "checkpoint_selection": "best_validation_loss",
        "initialization_sha256": initial_digest,
        "workers": args.workers,
        "cpu_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "dataset_ordering_sha256": dataset_ordering,
    }
    configuration["torchaudio"] = torchaudio.__version__
    write_json(destination / "configuration.json", configuration)
    best_val_loss = 1e3
    best_val_epoch = 0
    steps = 0
    start = time.monotonic()
    history_fields = [
        "epoch",
        "train_loss",
        "train_accuracy_percent",
        "validation_loss",
        "validation_accuracy_percent",
        "learning_rate",
        "optimizer_steps",
        "elapsed_seconds",
    ]
    with (destination / "history.csv").open("x", newline="") as handle:
        history = csv.DictWriter(handle, fieldnames=history_fields)
        history.writeheader()
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, criterion, optimizer, train_loader, scheduler
            )
            steps += len(train_loader)
            valid_loss, valid_acc = evaluate(model, criterion, valid_loader)
            if not np.isfinite([train_loss, valid_loss]).all():
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            if valid_loss < best_val_loss:
                best_val_loss, best_val_epoch = float(valid_loss), epoch + 1
                torch.save(model.state_dict(), destination / "best_validation_loss.pt")
            history.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy_percent": 100 * train_acc,
                    "validation_loss": valid_loss,
                    "validation_accuracy_percent": 100 * valid_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "optimizer_steps": steps,
                    "elapsed_seconds": time.monotonic() - start,
                }
            )
            handle.flush()
            print(
                f"{destination.name}: epoch {epoch + 1}/{EPOCHS}, validation {100 * valid_acc:.2f}%",
                flush=True,
            )
    if steps != OPTIMIZER_STEPS:
        raise RuntimeError(f"Expected {OPTIMIZER_STEPS} updates, got {steps}")
    model.load_state_dict(
        torch.load(
            destination / "best_validation_loss.pt",
            map_location=device,
            weights_only=True,
        )
    )
    test_loss, test_acc = evaluate(model, criterion, test_loader)
    if not np.isfinite([test_loss, test_acc]).all():
        raise FloatingPointError("Non-finite test result")
    result = {
        "surrogate": args.surrogate,
        "alpha": args.alpha,
        "seed": args.seed,
        "test_accuracy_percent": 100 * test_acc,
        "test_loss": float(test_loss),
        "epochs": EPOCHS,
        "optimizer_steps": steps,
        "best_validation_loss": best_val_loss,
        "best_validation_epoch": best_val_epoch,
        "checkpoint_selection": "best_validation_loss",
    }
    with (destination / "result.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(f"{destination.name}: test {100 * test_acc:.4f}%", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train one S-Edge-Tiny Table 2 configuration"
    )
    parser.add_argument(
        "--surrogate", choices=["relu", "bsilu", "leaky", "nelu"], required=True
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("datasets/speech_commands")
    )
    parser.add_argument("--data-dir", type=Path, default=data_directory())
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=16)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
