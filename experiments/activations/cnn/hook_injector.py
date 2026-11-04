from typing import Callable, List

import torch
import torch.nn as nn


class HookInjector:
    def __init__(self, model: nn.Module):
        self.model = model
        self.forward_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.backward_handles: List[torch.utils.hooks.RemovableHandle] = []

    def change_model(self, model: nn.Module, reset_old=True):
        if reset_old:
            self.remove_all_forward()
            self.remove_all_backward()
        self.model = model
        self.forward_handles.clear()
        self.backward_handles.clear()

    def inject_forward(
        self,
        fwd_hook_fnct: Callable[
            [str, nn.Module, any],
            Callable[[nn.Module, torch.Tensor, torch.Tensor], None],
        ],
        layer_condition_fnct: Callable[[nn.Module], bool],
        *data,
    ) -> List[torch.utils.hooks.RemovableHandle]:
        new_handles = []
        for name, module in self.model.named_modules():
            if layer_condition_fnct(module):
                handle = module.register_forward_hook(
                    fwd_hook_fnct(name, module, *data)
                )
                new_handles.append(handle)
        self.forward_handles += new_handles
        return new_handles

    def inject_backward(
        self,
        bwd_hook_fnct: Callable[
            [str, nn.Module, any],
            Callable[[nn.Module, torch.Tensor, torch.Tensor], None],
        ],
        layer_condition_fnct: Callable[[nn.Module], bool],
        *data,
    ) -> List[torch.utils.hooks.RemovableHandle]:
        new_handles = []
        for name, module in self.model.named_modules():
            if layer_condition_fnct(module):
                handle = module.register_full_backward_hook(
                    bwd_hook_fnct(name, module, *data)
                )
                new_handles.append(handle)
        self.backward_handles += new_handles
        return new_handles

    def inject_input_gradient(self, hook_factory, layer_condition_fnct, *data):
        new_handles = []
        for name, module in self.model.named_modules():
            if not layer_condition_fnct(module):
                continue
            callback = hook_factory(name, module, *data)

            def capture_input(module, inputs, callback=callback):
                if not inputs or not inputs[0].requires_grad:
                    return
                # A per-call view isolates this layer's gradient from other input users.
                tracked_input = inputs[0].view_as(inputs[0])
                handle = tracked_input.register_hook(callback)
                self.backward_handles.append(handle)
                return (tracked_input, *inputs[1:])

            new_handles.append(module.register_forward_pre_hook(capture_input))
        self.backward_handles += new_handles
        return new_handles

    def remove_all_forward(self):
        for handle in self.forward_handles:
            handle.remove()
        self.forward_handles.clear()

    def remove_all_backward(self):
        for handle in self.backward_handles:
            handle.remove()
        self.backward_handles.clear()
