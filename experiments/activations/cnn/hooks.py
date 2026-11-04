import torch
import torch.nn as nn


class LayerHookResult:
    def __init__(self, data, tensor):
        self.data = data
        self.layer_size = torch.numel(tensor)


def get_fwd_acts(name, module, act_dict):

    def hook_fn(module, input, output):
        act_dict[str(name)] = LayerHookResult(output, output)

    return hook_fn


def get_fwd_acts_minmax(name, module, minmax_list):

    def hook_fn(module, input, output):
        min_val = torch.min(output).cpu().detach().numpy().item()
        max_val = torch.max(output).cpu().detach().numpy().item()
        if len(minmax_list) == 0:
            minmax_list.extend([min_val, max_val])
        else:
            minmax_list[0] = min(minmax_list[0], min_val)
            minmax_list[1] = max(minmax_list[1], max_val)

    return hook_fn


def get_input_grads_minmax(name, module, minmax_list):

    def hook_fn(gradient):
        min_val = torch.min(gradient).cpu().detach().numpy().item()
        max_val = torch.max(gradient).cpu().detach().numpy().item()
        if len(minmax_list) == 0:
            minmax_list.extend([min_val, max_val])
        else:
            minmax_list[0] = min(minmax_list[0], min_val)
            minmax_list[1] = max(minmax_list[1], max_val)

    return hook_fn


def get_bw_grads(name, module, grad_dict):

    def hook_fn(module, grad_input, grad_output):
        grad_dict[str(name)] = LayerHookResult(grad_input[0], grad_input[0])

    return hook_fn


def get_fwd_acts_attach(name, module, act_dict):

    def hook_fn(module, input, output):
        layer_name = str(name)
        if layer_name in act_dict:
            act_dict[layer_name].data = torch.vstack(
                (act_dict[layer_name].data, output[0])
            )
            act_dict[layer_name].layer_size += torch.numel(output)
        else:
            act_dict[layer_name] = LayerHookResult(output, output)

    return hook_fn


def get_bw_grads_attach(name, module, grad_dict):

    def hook_fn(module, grad_input, grad_output):
        layer_name = str(name)
        if layer_name in grad_dict:
            grad_dict[layer_name].data = torch.vstack(
                (grad_dict[layer_name].data, grad_input[0])
            )
            grad_dict[layer_name].layer_size += torch.numel(grad_input[0])
        else:
            grad_dict[layer_name] = LayerHookResult(grad_input[0], grad_input[0])

    return hook_fn


def get_fwd_acts_hist(name, module, act_dict, min_val=0.0, max_val=10.0):

    def hook_fn(module, input, output):
        layer_name = str(name)
        hist = torch.histc(output, min=min_val, max=max_val).cpu().numpy()
        if layer_name in act_dict:
            act_dict[layer_name].data += hist
            act_dict[layer_name].layer_size += torch.numel(output)
        else:
            act_dict[layer_name] = LayerHookResult(hist, output)

    return hook_fn


def get_input_grads_hist(name, module, grad_dict, min_val=0.0, max_val=10.0):

    def hook_fn(gradient):
        layer_name = str(name)
        hist = torch.histc(gradient, min=min_val, max=max_val).cpu().numpy()
        if layer_name in grad_dict:
            grad_dict[layer_name].data += hist
            grad_dict[layer_name].layer_size += torch.numel(gradient)
        else:
            grad_dict[layer_name] = LayerHookResult(hist, gradient)

    return hook_fn


def get_fwd_nonzero_count(name, module, act_dict):

    def hook_fn(module, input, output):
        layer_name = str(name)
        nonzero_count = torch.count_nonzero(output).cpu().numpy()
        if layer_name in act_dict:
            act_dict[layer_name].data += nonzero_count
            act_dict[layer_name].layer_size += torch.numel(output)
        else:
            act_dict[layer_name] = LayerHookResult(nonzero_count, output)

    return hook_fn


def is_relu(module):
    return isinstance(module, nn.ReLU)


def is_layer_type(layer_type):
    return lambda module: isinstance(module, layer_type)


def get_fwd_acts_nonzero_act_cnt(name, module, act_dict):

    def hook_fn(module, input, output):
        layer_name = str(name)
        n_samples = output.shape[0]
        for i in range(n_samples):
            sample_acts = output[i]
            nonzero_act_neurons = torch.where(sample_acts > 0, 1, 0)
            if layer_name in act_dict:
                act_dict[layer_name].data += nonzero_act_neurons
                act_dict[layer_name].layer_size += torch.numel(nonzero_act_neurons)
            else:
                act_dict[layer_name] = LayerHookResult(
                    nonzero_act_neurons, nonzero_act_neurons
                )

    return hook_fn
