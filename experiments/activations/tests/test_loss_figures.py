import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from cnn.figures import loss, panels


def records(accuracies):
    epochs = np.arange(100)
    return [(epochs, np.full(100, 2.5), accuracy) for accuracy in accuracies]


class LossFigureTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {"param_swish": records([48, 49, 50, 51, 52])}
        self.sugar = {
            "param_swish": records([60, 61, 62, 63, 64]),
            "relu": records([41, 42, 43, 44, 45]),
        }
        self.saved = {}

    def tearDown(self):
        plt.close("all")

    def collect(self, _, mode):
        return dict(self.baseline if mode == "non_fgi" else self.sugar)

    def capture(self, figure, destination, **kwargs):
        self.saved[Path(destination).name] = [
            [label.get_text() for label in legend.get_texts()]
            for legend in figure.legends
        ]

    def test_combined_shared_legend_uses_sugar_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(loss, "collect_runs", self.collect):
                with patch.object(
                    Figure, "savefig", autospec=True, side_effect=self.capture
                ):
                    loss.plot_loss_curves(directory, Path(directory) / "loss.pdf")
        legends = self.saved["loss.pdf"]
        self.assertEqual(len(legends), 1)
        self.assertIn("B-SiLU (62.0%)", legends[0])
        self.assertNotIn("B-SiLU (50.0%)", legends[0])
        self.assertIn("ReLU (43.0%)", legends[0])

    def test_individual_legends_use_their_own_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(loss, "collect_runs", self.collect),
                patch.object(
                    Figure, "savefig", autospec=True, side_effect=self.capture
                ),
                patch.object(panels.accuracy_plot, "plot_accuracy"),
                patch.object(
                    panels.activation_plot, "load_activations", return_value=({}, 40)
                ),
                patch.object(panels, "activation_z_max", return_value=1),
                patch.object(panels, "plot_3d_panel"),
                patch.object(panels, "plot_comparison_panel"),
            ):
                panels.export_individuals(
                    directory, "relu.pt", "sugar.pt", directory, 40
                )
        first = self.saved["fig3_a.pdf"][0]
        second = self.saved["fig3_b.pdf"][0]
        self.assertIn("B-SiLU (50.0%)", first)
        self.assertNotIn("B-SiLU (62.0%)", first)
        self.assertIn("B-SiLU (62.0%)", second)
        self.assertIn("ReLU (43.0%)", first)
        self.assertIn("ReLU (43.0%)", second)


if __name__ == "__main__":
    unittest.main()
