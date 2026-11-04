import copy
import statistics

from mobilenet.train import PROJECT_ROOT, parse_args, train


SEEDS = [1, 10, 20, 25, 42]

SUGAR_SETTINGS: list[tuple[str, float | None]] = [
    ("base",  None),
    ("nelu",  0.01),
    ("nelu",  0.05),
    ("leaky", -0.01),
    ("bsilu", None),
]


def format_sugar_label(sugar: str, alpha: float | None) -> str:
    if alpha is not None:
        return f"{sugar}(alpha={alpha})"
    return sugar


def run_single(base_args, seed: int, sugar: str, alpha: float | None) -> float:
    args = copy.copy(base_args)
    args.seed = seed
    args.sugar = sugar
    args.alpha = alpha
    return train(args)


def main() -> None:
    args = parse_args()

    results: dict[str, list[float]] = {}

    for sugar, alpha in SUGAR_SETTINGS:
        label = format_sugar_label(sugar, alpha)
        accs: list[float] = []

        print(f"\n=== Config: SUGAR={label} ===")
        for seed in SEEDS:
            print(f"Running seed={seed}")
            test_acc = run_single(args, seed, sugar, alpha)
            print(f"test accuracy: {test_acc:.6f}")
            accs.append(test_acc)

        results[label] = accs

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{args.dataset}.txt"
    with open(results_path, "a") as f:
        f.write("Experiment results\n")
        f.write("==================\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Seeds: {SEEDS}\n\n")

        for label, accs in results.items():
            mean_acc = statistics.mean(accs)
            std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0

            f.write(f"SUGAR: {label}\n")
            f.write("  test accuracies: " + ", ".join(f"{a:.6f}" for a in accs) + "\n")
            f.write(f"  mean: {mean_acc:.6f}\n")
            f.write(f"  std:  {std_acc:.6f}\n")
            f.write("\n")

    print(f"\nAll runs finished. Aggregated results written to {results_path}")


if __name__ == "__main__":
    main()
