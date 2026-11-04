import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from cnn.checkpoints import seeded_files
from cnn.figures import accuracy, activations, loss, panels


def check_results(results):
    for mode in ("fgi", "non_fgi"):
        for activation in loss.ORDER:
            if activation == "relu" and mode == "non_fgi":
                continue
            folder = results / "cifar100" / "resnet18" / mode / activation
            if activation == "nelu":
                folder /= "alpha_0.01"
            seeded_files(folder)
        for activation, alpha, _ in accuracy.CONFIGS:
            if activation == "relu" and mode == "non_fgi":
                continue
            folder = results / "cifar100" / "vgg16" / mode / activation
            if alpha:
                folder /= alpha
            seeded_files(folder)


def analysis_checkpoint(root, activation):
    folder = root / "cifar100" / "vgg16" / "fgi" / activation / "seed_1"
    files = sorted(folder.glob("run_acc_*.pt"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one analysis checkpoint in {folder}")
    return files[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--analysis-dir", type=Path, default=Path("analyze_freq_results")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument(
        "--epoch", type=int, default=40, help="zero-based stored epoch key"
    )
    args = parser.parse_args()
    check_results(args.results_dir)
    relu_path = analysis_checkpoint(args.analysis_dir, "relu")
    sugar_path = analysis_checkpoint(args.analysis_dir, "param_swish")
    relu, _ = activations.load_activations(relu_path, args.epoch)
    sugar, _ = activations.load_activations(sugar_path, args.epoch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss.plot_loss_curves(
        args.results_dir, args.output_dir / "cifar100_resnet18_loss_curves.pdf"
    )
    accuracy.plot_accuracy(
        args.results_dir, args.output_dir / "VGG16_CIFAR100_accuracy_c.pdf"
    )
    activations.plot_combined(
        relu,
        sugar,
        args.output_dir / "vgg16_cifar100_activations_combined.pdf",
        args.epoch,
        z_axis_side="back-left",
        z_annotation_side="right",
        z_label_side="left",
        z_label_offset=0.105,
        show_both_z_labels=True,
        center_layer_ticks=True,
        fc_label_offset=1.0,
        fc_label_shift_right=0.5,
        fc_label_shift_down=1.4,
    )
    panels.export_individuals(
        args.results_dir, relu_path, sugar_path, args.output_dir, args.epoch
    )
    print(f"Saved Figure 3 and individual panels to {args.output_dir}")


if __name__ == "__main__":
    main()
