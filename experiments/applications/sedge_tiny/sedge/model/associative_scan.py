# Adapted from S-Edge for the SUGAR experiments; packaging and comments simplified.
# See THIRD_PARTY.md and licenses/ for Apache-2.0 and MPL-2.0 notices.

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from functools import partial
from typing import Callable, overload, Any, List, Iterable, TypeVar, Tuple

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


@overload
def safe_map(f: Callable[[T1], T], __arg1: Iterable[T1]) -> List[T]: ...


@overload
def safe_map(
    f: Callable[[T1, T2], T], __arg1: Iterable[T1], __arg2: Iterable[T2]
) -> List[T]: ...


@overload
def safe_map(
    f: Callable[[T1, T2, T3], T],
    __arg1: Iterable[T1],
    __arg2: Iterable[T2],
    __arg3: Iterable[T3],
) -> List[T]: ...


@overload
def safe_map(
    f: Callable[..., T],
    __arg1: Iterable[Any],
    __arg2: Iterable[Any],
    __arg3: Iterable[Any],
    __arg4: Iterable[Any],
    *args,
) -> List[T]: ...


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(map(f, *args))


def combine(tree, operator, a_flat, b_flat):
    a = tree_unflatten(a_flat, tree)
    b = tree_unflatten(b_flat, tree)
    c = operator(a, b)
    c_flat, _ = tree_flatten(c)
    return c_flat


def _interleave(a, b, axis: int):
    b_trunc = a.shape[axis] == b.shape[axis] + 1
    if b_trunc:
        pad = [0, 0] * b.ndim
        pad[(b.ndim - axis - 1) * 2 + 1] = 1
        b = torch.nn.functional.pad(b, pad)
    stacked = torch.stack([a, b], dim=axis + 1)
    interleaved = torch.flatten(stacked, start_dim=axis, end_dim=axis + 1)
    if b_trunc:
        interleaved = torch.ops.aten.slice(
            interleaved, axis, 0, b.shape[axis] + a.shape[axis] - 1
        )
    return interleaved


def _scan(tree, operator, elems, axis: int):
    num_elems = elems[0].shape[axis]
    if num_elems < 2:
        return elems
    reduced_elems = combine(
        tree,
        operator,
        [torch.ops.aten.slice(elem, axis, 0, -1, 2) for elem in elems],
        [torch.ops.aten.slice(elem, axis, 1, None, 2) for elem in elems],
    )
    odd_elems = _scan(tree, operator, reduced_elems, axis)
    if num_elems % 2 == 0:
        even_elems = combine(
            tree,
            operator,
            [torch.ops.aten.slice(e, axis, 0, -1) for e in odd_elems],
            [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems],
        )
    else:
        even_elems = combine(
            tree,
            operator,
            odd_elems,
            [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems],
        )
    even_elems = [
        torch.cat([torch.ops.aten.slice(elem, axis, 0, 1), result], dim=axis)
        if result.shape.numel() > 0 and elem.shape[axis] > 0
        else result
        if result.shape.numel() > 0
        else torch.ops.aten.slice(elem, axis, 0, 1)
        for elem, result in zip(elems, even_elems)
    ]
    return list(safe_map(partial(_interleave, axis=axis), even_elems, odd_elems))


@torch.jit.script
def binary_operator(
    q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]
):
    A_i, b_i = q_i
    A_j, b_j = q_j
    return (A_j * A_i, torch.addcmul(b_j, A_j, b_i))


def associative_scan(operator: Callable, elems, axis: int = 0, reverse: bool = False):
    elems_flat, tree = tree_flatten(elems)
    if reverse:
        elems_flat = [torch.flip(elem, [axis]) for elem in elems_flat]
    assert axis >= 0 or axis < elems_flat[0].ndim, (
        "Axis should be within bounds of input"
    )
    num_elems = int(elems_flat[0].shape[axis])
    if not all((int(elem.shape[axis]) == num_elems for elem in elems_flat[1:])):
        raise ValueError(
            "Array inputs to associative_scan must have the same first dimension. (saw: {})".format(
                [elem.shape for elem in elems_flat]
            )
        )
    scans = _scan(tree, operator, elems_flat, axis)
    if reverse:
        scans = [torch.flip(scanned, [axis]) for scanned in scans]
    return tree_unflatten(scans, tree)


def apply_ssm(
    Lambda_bars: torch.Tensor, B, B_bias, C, C_bias, input_sequence, complex_output
):
    cinput_sequence = input_sequence.type(Lambda_bars.dtype)
    Bu_elements = cinput_sequence @ B.T + B_bias.view(1, 1, -1)
    if Lambda_bars.ndim == 1:
        Lambda_bars = Lambda_bars.tile(input_sequence.shape[1], 1)
    _, xs = torch.vmap(lambda Bu: associative_scan(binary_operator, (Lambda_bars, Bu)))(
        Bu_elements
    )
    if complex_output:
        out = xs @ C.T + C_bias.view(1, 1, -1)
    else:
        out = (xs @ C.T + C_bias.view(1, 1, -1)).real
    return out
