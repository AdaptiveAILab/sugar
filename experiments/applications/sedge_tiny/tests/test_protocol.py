import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sedge.experiment import cases, main as experiment_main
from sedge.protocol import OPTIMIZER_STEPS, SEEDS, VARIANTS, run_name
from sedge.sugar import ReLUSurrogate
from sedge.summarize import summarize
from sedge.train import (
    build_model,
    build_optimizer,
    build_scheduler,
    decoder_regularization,
)


class ProtocolTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "surrogate": surrogate,
                "alpha": alpha,
                "seed": seed,
                "test_accuracy_percent": accuracy,
                "epochs": 40,
                "optimizer_steps": OPTIMIZER_STEPS,
            }
            for surrogate, alpha in VARIANTS
            for accuracy, seed in enumerate(SEEDS)
        ]

    def test_full_grid_and_disjoint_shards(self):
        expected = {
            (surrogate, alpha, seed) for surrogate, alpha in VARIANTS for seed in SEEDS
        }
        self.assertEqual(len(expected), 45)
        self.assertEqual(set(cases()), expected)
        shards = [set(cases(i, 4)) for i in range(4)]
        self.assertEqual(set.union(*shards), expected)
        self.assertEqual(sum(map(len, shards)), 45)
        for invalid in [(-1, 1), (1, 1), (0, 0)]:
            with self.assertRaises(ValueError):
                cases(*invalid)

    def test_sweep_refuses_existing_output_before_launching(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / run_name("relu", 0, 1)).mkdir()
            argv = ["sedge.experiment", "--output-dir", str(output)]
            with patch("sys.argv", argv), patch("subprocess.run") as launch:
                with self.assertRaises(FileExistsError):
                    experiment_main()
                launch.assert_not_called()

    def test_sample_statistics(self):
        summary = summarize(self.rows())
        self.assertEqual(len(summary), 9)
        for row in summary:
            self.assertEqual(row["n"], 5)
            self.assertEqual(row["mean_accuracy_percent"], 2)
            self.assertAlmostEqual(row["std_accuracy_percent"], math.sqrt(2.5))
            self.assertEqual(row["best_accuracy_percent"], 4)

    def test_reject_incomplete_duplicate_and_invalid_runs(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            summarize(rows[:-1])
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]])
        for key, value in [
            ("seed", 4),
            ("epochs", 39),
            ("optimizer_steps", 212119),
            ("test_accuracy_percent", float("nan")),
            ("test_accuracy_percent", 101),
        ]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                summarize([{**rows[0], key: value}] + rows[1:])

    def test_relu_forward_and_surrogate_derivatives(self):
        for surrogate, alpha in VARIANTS:
            with self.subTest(surrogate=surrogate, alpha=alpha):
                x = torch.tensor(
                    [-10.0, -1.0, 0.0, 1.0, 10.0],
                    dtype=torch.float64,
                    requires_grad=True,
                )
                output = ReLUSurrogate(surrogate, alpha)(x)
                torch.testing.assert_close(output, torch.relu(x), rtol=0, atol=0)
                output.sum().backward()
                if surrogate == "relu":
                    expected = (x > 0).to(x.dtype)
                elif surrogate == "leaky":
                    expected = torch.where(
                        x > 0, torch.ones_like(x), torch.full_like(x, alpha)
                    )
                elif surrogate == "nelu":
                    expected = torch.where(
                        x > 0, 1.0, 2 * alpha * x / (1 + x.square()).square()
                    )
                else:
                    sigmoid = torch.sigmoid(x)
                    expected = sigmoid + (x + alpha) * sigmoid * (1 - sigmoid)
                torch.testing.assert_close(x.grad, expected, rtol=1e-12, atol=1e-12)

    def test_optimizer_groups_and_scalar_schedule(self):
        model = build_model()
        optimizer = build_optimizer(model)
        selected = {
            id(p)
            for name, p in model.named_parameters()
            if any(key in name for key in ["B", "C", "Lambda", "log_step"])
        }
        self.assertEqual({id(p) for p in optimizer.param_groups[0]["params"]}, selected)
        self.assertEqual(
            [group["weight_decay"] for group in optimizer.param_groups], [0, 0.04]
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups], [0.002, 0.008]
        )
        scheduler = build_scheduler(optimizer)
        for step, expected in [
            (0, 0.0),
            (1, 0.00008),
            (100, 0.008),
            (OPTIMIZER_STEPS, 0.0),
        ]:
            scheduler.step(step)
            self.assertEqual(
                [group["lr"] for group in optimizer.param_groups], [expected, expected]
            )

    def test_decoder_regularization(self):
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expected = 0.1 * math.sqrt((11**2 + 11**2) / 4)
        self.assertAlmostEqual(
            decoder_regularization(weights).item(), expected, places=7
        )


if __name__ == "__main__":
    unittest.main()
