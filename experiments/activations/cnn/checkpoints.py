import pickle
from pathlib import Path
from types import ModuleType

import torch

from cnn.hooks import LayerHookResult

SEEDS = (1, 10, 20, 25, 42)


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("tools.hooks", "LayerHookResult"):
            return LayerHookResult
        return super().find_class(module, name)


_legacy_pickle = ModuleType("cnn_legacy_pickle")
_legacy_pickle.Unpickler = _LegacyUnpickler
_legacy_pickle.load = pickle.load


def load_checkpoint(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_legacy_pickle,
    )


def seeded_files(root):
    root = Path(root)
    files = sorted(root.glob("seed_*/run_acc_*.pt"))
    expected = [f"seed_{seed}" for seed in SEEDS]
    if sorted(path.parent.name for path in files) != expected:
        raise ValueError(
            f"Expected exactly one checkpoint for each seed {SEEDS} in {root}"
        )
    return files
