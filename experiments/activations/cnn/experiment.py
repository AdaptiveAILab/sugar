import argparse
import gc
import itertools
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import torch
import yaml

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
from cnn.train import Experiment

ALPHA_FGI = {"nelu"}


def _epochs_for(dataset: str, model: str) -> int:
    dataset = dataset.lower()
    model = model.lower()
    if dataset == "tinyimg":
        return 150 if model == "resnet18" else 100
    if dataset == "cifar100":
        return 100 if model == "resnet18" else 50
    return 50


def run_case(args, cfg):
    fgi, dataset, model, use_fgi, seed, alpha, gpu_id = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cfg_epochs = cfg.get("num_epochs", None)
    if isinstance(cfg_epochs, int):
        num_epochs = cfg_epochs
    elif isinstance(cfg_epochs, dict):
        de = cfg_epochs.get(dataset, None)
        if isinstance(de, int):
            num_epochs = de
        elif isinstance(de, dict):
            num_epochs = de.get(model, de.get("default"))
        else:
            num_epochs = _epochs_for(dataset, model)
    else:
        num_epochs = _epochs_for(dataset, model)
    exp = Experiment(
        use_fgi=use_fgi,
        dataset=dataset,
        model_name=model,
        data_dir=cfg["data_dir"],
        batch_size=cfg["batch_size"],
        valid_size=cfg["valid_size"],
        seed=seed,
        num_workers=cfg["num_workers"],
        fgi_type=fgi,
        num_epochs=num_epochs,
        learning_rate=cfg["learning_rate"],
        optimizer_name=cfg["optimizer_name"],
        milestones=cfg["milestones"],
        alpha=alpha or 0.0,
        results_dir=cfg["results_dir"],
        gpu_id=0,
        reg=cfg.get("reg", False),
    )
    out_dir = exp._make_results_path()
    if any((p.endswith(".pt") for p in os.listdir(out_dir))):
        print(f"skip (already done): {out_dir}")
        return None
    exp.log_args()
    result = exp.train(analyze_freq=cfg.get("analyze_freq", 10000))
    gc.collect()
    torch.cuda.empty_cache()
    return result


def build_cases(cfg):
    cases = []
    modes = cfg.get("modes", ["sugar", "baseline"])
    if not modes or set(modes) - {"sugar", "baseline"}:
        raise ValueError("modes must contain sugar and/or baseline")
    for fgi, dataset, model, seed in itertools.product(
        cfg["fgi_types"], cfg["datasets"], cfg["models"], cfg["seeds"]
    ):
        alphas = cfg["alphas"] if fgi in ALPHA_FGI else [None]
        for alpha in alphas:
            for mode in modes:
                use_fgi = mode == "sugar"
                if not use_fgi and fgi == "relu":
                    continue
                cases.append([fgi, dataset, model, use_fgi, seed, alpha, None])
    random.Random(42).shuffle(cases)
    if len({tuple(case) for case in cases}) != len(cases):
        raise ValueError("The configuration contains duplicate runs")
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs") / "training.yaml",
    )
    parser.add_argument("--model", choices=["vgg16", "resnet18"])
    parser.add_argument("--activation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=["sugar", "baseline"])
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("--data-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", "--n_shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.n_shards < 1 or not 0 <= args.shard < args.n_shards:
        parser.error("shard must be in [0, n-shards)")
    with args.config.open() as handle:
        cfg = yaml.safe_load(handle)
    for key in ("data_dir", "results_dir"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.gpus is not None:
        cfg["gpu_cycle"] = args.gpus
    gpu_cycle = cfg["gpu_cycle"]
    if not gpu_cycle or len(set(gpu_cycle)) != len(gpu_cycle):
        parser.error("gpu_cycle must contain distinct GPU indices")
    if cfg.get("max_parallel_per_gpu", 2) < 1 or cfg.get("analyze_freq", 10000) < 1:
        parser.error("parallelism and analyze_freq must be positive")
    if cfg["optimizer_name"] != "SGD":
        parser.error("this experiment uses SGD with momentum and Nesterov")
    cases = [
        case
        for case in build_cases(cfg)
        if (args.model is None or case[2] == args.model)
        and (args.activation is None or case[0] == args.activation)
        and (args.seed is None or case[4] == args.seed)
        and (args.mode is None or case[3] == (args.mode == "sugar"))
    ]
    if not cases:
        parser.error("no runs match the requested filters")
    cases = [
        case for index, case in enumerate(cases) if index % args.n_shards == args.shard
    ]
    for index, case in enumerate(cases):
        case[-1] = gpu_cycle[index % len(gpu_cycle)]
    if args.dry_run:
        print(json.dumps({"num_runs": len(cases), "runs": cases}, indent=2))
        return
    if not cases:
        print(json.dumps({"num_runs": 0}))
        return
    if torch.version.cuda is None:
        parser.error(
            "the reproduction launcher requires CUDA; --dry-run works without a GPU"
        )
    from torchvision import datasets

    for dataset in sorted({case[1] for case in cases}):
        dataset_type = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[
            dataset
        ]
        dataset_type(cfg["data_dir"], train=True, download=True)
    max_parallel = len(gpu_cycle) * cfg.get("max_parallel_per_gpu", 2)
    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        worker = partial(run_case, cfg=cfg)
        completed = sum(1 for _ in pool.map(worker, cases))
    print(json.dumps({"num_runs": completed}, indent=2))


if __name__ == "__main__":
    main()
