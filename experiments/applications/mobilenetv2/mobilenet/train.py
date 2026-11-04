import os
import time
import logging
import argparse
from argparse import Namespace
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from mobilenet import datasets
from mobilenet.model import build_model
from mobilenet.sugar import (
    SUGAR_NeLU, SUGAR_BSiLU, SUGAR_LeakyReLU,
    replace_modules_, list_activations,
)
from mobilenet.utils import (
    set_seed, human_time, train_one_epoch, evaluate,
    freeze_all_but_classifier_, unfreeze_last_n_params_,
)


# Project root: anchors data/, output/, results/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


SUGAR_BUILDERS = {
    "nelu":  lambda a: SUGAR_NeLU(alpha=a),
    "bsilu": lambda a: SUGAR_BSiLU(),
    "leaky": lambda a: SUGAR_LeakyReLU(alpha=a),
}


def run_phase(name: str, epochs: int, model: nn.Module, optimizer: optim.Optimizer,
              scaler: torch.amp.GradScaler, train_loader, val_loader, device: torch.device,
              log: logging.Logger, best_path: str, best_val: float) -> float:
    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0
        is_best = val_acc > best_val
        if is_best:
            best_val = val_acc
            torch.save(
                {"model_state": model.state_dict(),
                 "val_acc": best_val},
                best_path,
            )
        tag = "(+)" if is_best else "(-)"
        log.info(
            f"{tag} [{name}][Epoch {epoch + 1:02d}] "
            f"train_loss={train_loss:.4f} acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} acc={val_acc:.2f}% | "
            f"{human_time(dt)}"
        )
    return best_val


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="oxford102", choices=list(datasets.REGISTRY),
                        help="Dataset to train on")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs_head", type=int, default=10)
    parser.add_argument("--epochs_finetune", type=int, default=150)
    parser.add_argument("--lr_head", type=float, default=3e-3)
    parser.add_argument("--lr_finetune", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--unfreeze_last", type=int, default=200, # considers the complete architecture
                        help="Number of last parameters to unfreeze for fine-tuning")
    parser.add_argument("--data_root", type=Path, default=None,
                        help="Defaults to <repo>/data/<dataset>/")
    parser.add_argument("--seed", type=int, default=42)
    # SUGAR params
    parser.add_argument("--sugar", choices=["base", "leaky", "nelu", "bsilu"],
                        default="base", help="SUGAR variant (base = no replacement)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Activation function parameter (used by nelu, leaky)")
    # HAM10000-only (Oxford102 uses fixed setid.mat splits and ignores these)
    parser.add_argument("--train_frac", type=float, default=0.7,
                        help="HAM10000 only: fraction of data used for training")
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="HAM10000 only: fraction of data used for validation")

    return parser


def parse_args() -> Namespace:
    parser = build_arg_parser()
    pre_args, _ = parser.parse_known_args()
    overrides = getattr(datasets.REGISTRY[pre_args.dataset], "DEFAULTS", {})
    parser.set_defaults(**overrides)

    return parser.parse_args()


def train(args: Namespace) -> float:
    ds = datasets.REGISTRY[args.dataset]
    if args.data_root is None:
        args.data_root = PROJECT_ROOT / "data" / args.dataset

    set_seed(args.seed)

    alpha_dir = "" if args.alpha is None else str(args.alpha).replace(".", "") + "_"
    run_name = f"{args.sugar}_{alpha_dir}{args.seed}"
    args.output_dir = PROJECT_ROOT / "output" / args.dataset / run_name
    os.makedirs(args.output_dir, exist_ok=True)

    # Remove all existing handlers before creating a global logger (basicConfig)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "results.txt")),
        ],
    )
    log = logging.getLogger(__name__)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # get DataLoader
    train_loader, val_loader, test_loader = ds.get_data(args)

    # initialize model
    model = build_model(num_classes=ds.NUM_CLASSES, pretrained=True)
    model = model.to(device)
    log.info(f"full model params: {sum(p.numel() for p in model.parameters())}")
    log.info(f"seed: {args.seed}")

    # SUGAR
    if args.sugar != "base":
        builder = SUGAR_BUILDERS[args.sugar]
        n = replace_modules_(model, nn.ReLU6, lambda: builder(args.alpha))
        log.info(f"Replaced {n} ReLU6 with {builder(args.alpha)}")

    log.info(f"Remaining activation modules: {list_activations(model)}")

    best_val = 0.0
    best_path = os.path.join(args.output_dir, "best_model.pth")

    ####################################
    # Stage 1: head-only training
    ####################################
    freeze_all_but_classifier_(model)
    log.info(f"Trainable params [head]: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr_head, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    best_val = run_phase("Head", args.epochs_head, model, optimizer, scaler,
                         train_loader, val_loader, device, log, best_path, best_val)

    ####################################
    # Stage 2: fine-tuning
    ####################################
    unfreeze_last_n_params_(model, args.unfreeze_last)
    log.info(f"Trainable params [fine-tune]: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr_finetune, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    best_val = run_phase("fine-tune", args.epochs_finetune, model, optimizer, scaler,
                         train_loader, val_loader, device, log, best_path, best_val)

    log.info(f"Best validation accuracy: {best_val:.2f}%")

    # Final eval on test set
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    log.info(f"Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")

    return test_acc


if __name__ == "__main__":
    train(parse_args())
