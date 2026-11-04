import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from cnn import hooks
from cnn.hook_injector import HookInjector
from cnn.sugar import apply_sugar
from cnn.train import run_analysis_epoch


def make_model(activation, compiled):
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3), nn.ReLU(), nn.Linear(3, 2)
    )
    compile_module = torch.compile
    with patch.object(
        torch,
        "compile",
        side_effect=lambda module: compile_module(module, backend="aot_eager"),
    ):
        return apply_sugar(model, activation, alpha=0.01, compile_layer=compiled)


def batches():
    values = torch.linspace(-2, 2, 24).reshape(6, 4)
    return [
        (values[:3], torch.tensor([0, 1, 0])),
        (values[3:], torch.tensor([1, 0, 1])),
    ]


def containers():
    return {
        key: {}
        for key in (
            "acts_hists",
            "grads_hists",
            "nonzero_counts",
            "acts_minmax",
            "grads_minmax",
            "nonzero_neuron_acts",
        )
    }


def measure(model, layer_type, loader=None, criterion=None, epoch=0):
    result = containers()
    injector = HookInjector(model)
    run_analysis_epoch(
        epoch,
        2,
        model,
        batches() if loader is None else loader,
        criterion or nn.CrossEntropyLoss(),
        injector,
        hooks,
        "cpu",
        layer_type,
        10,
        result,
    )
    return result, injector


def expected_statistics(model, layer_type):
    captured = []
    gradients, activations = {}, {}
    handles = []

    def capture(name):
        def hook(module, inputs, output):
            captured.append((name, inputs[0], output))

        return hook

    for name, module in model.named_modules():
        if isinstance(module, layer_type):
            handles.append(module.register_forward_hook(capture(name)))
    try:
        for inputs, targets in batches():
            captured.clear()
            output = model(inputs)
            values = torch.autograd.grad(
                nn.CrossEntropyLoss()(output, targets), [item[1] for item in captured]
            )
            for (name, _, activation), gradient in zip(captured, values):
                gradients.setdefault(name, []).append(gradient.detach())
                activations.setdefault(name, []).append(activation.detach())
    finally:
        for handle in handles:
            handle.remove()
    return gradients, activations


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_analysis_matches_autograd_for_eager_and_warmed_compiled_layers(self):
        for activation in ("param_swish", "nelu", "relu"):
            reference, reference_type = make_model(activation, False)
            gradients, activations = expected_statistics(reference, reference_type)
            lower = min(
                value.min().item() for values in gradients.values() for value in values
            )
            upper = max(
                value.max().item() for values in gradients.values() for value in values
            )
            for inputs, targets in batches():
                nn.CrossEntropyLoss()(reference(inputs), targets).backward()
            expected_parameter_gradients = [
                parameter.grad.clone() for parameter in reference.parameters()
            ]
            for compiled in (False, True):
                with self.subTest(activation=activation, compiled=compiled):
                    model, layer_type = make_model(activation, compiled)
                    inputs, targets = batches()[0]
                    nn.CrossEntropyLoss()(model(inputs), targets).backward()
                    model.zero_grad()
                    state = {
                        name: value.clone()
                        for name, value in model.state_dict().items()
                    }
                    rng = torch.get_rng_state().clone()
                    result, injector = measure(model, layer_type)
                    torch.testing.assert_close(
                        torch.get_rng_state(), rng, rtol=0, atol=0
                    )
                    np.testing.assert_allclose(
                        result["grads_minmax"][0], [lower, upper], rtol=1e-6
                    )
                    self.assertEqual(set(result["grads_hists"][0]), set(gradients))
                    for name, values in gradients.items():
                        expected = sum(
                            torch.histc(value, min=lower, max=upper) for value in values
                        )
                        np.testing.assert_array_equal(
                            result["grads_hists"][0][name].data, expected.numpy()
                        )
                        counts = sum(
                            value.gt(0).sum(dim=0) for value in activations[name]
                        )
                        np.testing.assert_array_equal(
                            result["nonzero_neuron_acts"][0][name], counts.numpy()
                        )
                        self.assertEqual(
                            result["grads_hists"][0][name].layer_size,
                            sum(value.numel() for value in values),
                        )
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
                    for parameter, gradient in zip(
                        model.parameters(), expected_parameter_gradients
                    ):
                        torch.testing.assert_close(
                            parameter.grad, gradient, rtol=1e-5, atol=1e-7
                        )
                    self.assertFalse(injector.forward_handles)
                    self.assertFalse(injector.backward_handles)

    def test_skipped_epoch_collects_nothing(self):
        model, layer_type = make_model("param_swish", False)
        result, injector = measure(model, layer_type, epoch=1)
        self.assertTrue(all(not value for value in result.values()))
        self.assertFalse(injector.forward_handles)
        self.assertFalse(injector.backward_handles)

    def test_empty_loader_has_a_clear_error(self):
        model, layer_type = make_model("param_swish", False)
        with self.assertRaisesRegex(ValueError, "empty data loader"):
            measure(model, layer_type, loader=[])

    def test_missing_layers_have_a_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "No matching activation layers"):
            measure(nn.Linear(4, 2), nn.ReLU)

    def test_missing_input_gradients_are_not_replaced_by_zeros(self):
        with self.assertRaisesRegex(
            RuntimeError, "no activation values or input gradients"
        ):
            measure(nn.Sequential(nn.ReLU(), nn.Linear(4, 2)), nn.ReLU)

    def test_hooks_are_removed_when_analysis_fails(self):
        model, layer_type = make_model("param_swish", True)
        injector = HookInjector(model)
        for failure_call in (1, 3):
            with self.subTest(failure_call=failure_call):
                calls = 0

                def failing_criterion(output, targets):
                    nonlocal calls
                    calls += 1
                    if calls == failure_call:
                        raise RuntimeError("analysis interrupted")
                    return nn.CrossEntropyLoss()(output, targets)

                with self.assertRaisesRegex(RuntimeError, "analysis interrupted"):
                    run_analysis_epoch(
                        0,
                        2,
                        model,
                        batches(),
                        failing_criterion,
                        injector,
                        hooks,
                        "cpu",
                        layer_type,
                        10,
                        containers(),
                    )
                self.assertFalse(injector.forward_handles)
                self.assertFalse(injector.backward_handles)
                self.assertTrue(
                    all(
                        not module._forward_hooks and not module._forward_pre_hooks
                        for module in model.modules()
                    )
                )

    def test_input_hooks_isolate_shared_and_repeated_inputs(self):
        for compiled in (False, True):
            with self.subTest(compiled=compiled):
                model, layer_type = make_model("param_swish", compiled)
                activation = model[1]
                injector = HookInjector(activation)
                captured = []

                def capture(name, module, target):
                    return lambda gradient: target.append(gradient.clone())

                injector.inject_input_gradient(
                    capture, hooks.is_layer_type(layer_type), captured
                )
                inputs = torch.tensor([-2.0, -0.5, 1.0], requires_grad=True)
                try:
                    (
                        activation(inputs).sum()
                        + 2 * activation(inputs).sum()
                        + 3 * inputs.sum()
                    ).backward()
                finally:
                    injector.remove_all_backward()
                self.assertEqual(len(captured), 2)
                reference, _ = make_model("param_swish", False)
                expected = torch.autograd.grad(reference[1](inputs).sum(), inputs)[0]
                captured.sort(key=lambda gradient: gradient.abs().sum().item())
                torch.testing.assert_close(captured[0], expected)
                torch.testing.assert_close(captured[1], 2 * expected)
                torch.testing.assert_close(inputs.grad, 3 * expected + 3)


if __name__ == "__main__":
    unittest.main()
