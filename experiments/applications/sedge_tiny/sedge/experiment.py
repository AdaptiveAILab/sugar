import argparse
import subprocess
import sys
from pathlib import Path

from .protocol import SEEDS, VARIANTS, data_directory, run_name


def cases(shard=0, n_shards=1):
    if n_shards < 1 or not 0 <= shard < n_shards:
        raise ValueError("Require n-shards >= 1 and 0 <= shard < n-shards")
    grid = [(surrogate, alpha, seed) for surrogate, alpha in VARIANTS for seed in SEEDS]
    return grid[shard::n_shards]


def main():
    parser = argparse.ArgumentParser(
        description="Run the 45 S-Edge-Tiny Table 2 experiments"
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("datasets/speech_commands")
    )
    parser.add_argument("--data-dir", type=Path, default=data_directory())
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = cases(args.shard, args.n_shards)
    destinations = [args.output_dir / run_name(*case) for case in selected]
    if not args.dry_run:
        for path in destinations:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite an existing run: {path}")
    for surrogate, alpha, seed in selected:
        command = [
            sys.executable,
            "-m",
            "sedge.train",
            "--surrogate",
            surrogate,
            "--alpha",
            str(alpha),
            "--seed",
            str(seed),
            "--dataset-dir",
            str(args.dataset_dir),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(args.output_dir),
            "--device",
            args.device,
            "--workers",
            str(args.workers),
            "--cpu-threads",
            str(args.cpu_threads),
        ]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
