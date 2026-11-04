import argparse
import csv
import io
import math
import statistics
from pathlib import Path

from .protocol import (
    EPOCHS,
    OPTIMIZER_STEPS,
    SEEDS,
    VARIANTS,
    validate_case,
)


def read_runs(path):
    path = Path(path)
    files = sorted(path.glob("*/result.csv")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No result CSV files in {path}")
    rows = []
    for file in files:
        with file.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "surrogate",
                "alpha",
                "seed",
                "test_accuracy_percent",
                "epochs",
                "optimizer_steps",
            }
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing required result columns in {file}")
            rows.extend(reader)
    return rows


def summarize(rows):
    groups = {variant: {} for variant in VARIANTS}
    for row in rows:
        surrogate, alpha, seed = row["surrogate"], float(row["alpha"]), int(row["seed"])
        validate_case(surrogate, alpha, seed)
        if (
            int(row["epochs"]) != EPOCHS
            or int(row["optimizer_steps"]) != OPTIMIZER_STEPS
        ):
            raise ValueError(f"Incomplete run: {surrogate}, alpha={alpha}, seed={seed}")
        accuracy = float(row["test_accuracy_percent"])
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
            raise ValueError(f"Invalid test accuracy: {accuracy}")
        group = groups[(surrogate, alpha)]
        if seed in group:
            raise ValueError(f"Duplicate run: {surrogate}, alpha={alpha}, seed={seed}")
        group[seed] = accuracy
    summary = []
    for (surrogate, alpha), values in groups.items():
        if set(values) != set(SEEDS):
            raise ValueError(
                f"Expected exactly seeds {SEEDS} for {surrogate}, alpha={alpha}; got {sorted(values)}"
            )
        accuracies = [values[seed] for seed in SEEDS]
        summary.append(
            {
                "surrogate": surrogate,
                "alpha": alpha,
                "n": len(accuracies),
                "mean_accuracy_percent": statistics.mean(accuracies),
                "std_accuracy_percent": statistics.stdev(accuracies),
                "best_accuracy_percent": max(accuracies),
            }
        )
    return summary


def format_summary(summary, format="csv"):
    if format == "csv":
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
        return output.getvalue()
    names = {
        "relu": "ReLU",
        "bsilu": "SUGAR (B-SiLU)",
        "leaky": "SUGAR (Leaky)",
        "nelu": "SUGAR (NeLU)",
    }
    best_mean = max(row["mean_accuracy_percent"] for row in summary)
    best_run = max(row["best_accuracy_percent"] for row in summary)
    lines = []
    for row in summary:
        mean = f"{row['mean_accuracy_percent']:.2f} $\\pm$ {row['std_accuracy_percent']:.2f}"
        best = f"{row['best_accuracy_percent']:.2f}"
        if row["mean_accuracy_percent"] == best_mean:
            mean = "\\textbf{" + mean + "}"
        if row["best_accuracy_percent"] == best_run:
            best = "\\textbf{" + best + "}"
        alpha = "--" if row["surrogate"] == "relu" else f"{row['alpha']:g}"
        lines.append(f"{names[row['surrogate']]} & {alpha} & {mean} & {best} \\\\")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate Table 2 with five seeds and sample SD"
    )
    parser.add_argument(
        "input", nargs="?", type=Path, default=Path("output")
    )
    parser.add_argument("--format", choices=["csv", "latex"], default="csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = format_summary(summarize(read_runs(args.input)), args.format)
    if args.output:
        with args.output.open("x") as handle:
            handle.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
