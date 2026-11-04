import hashlib
import sys
from pathlib import Path


SEEDS = (1, 10, 20, 25, 42)
VARIANTS = (
    ("relu", 0.0),
    ("bsilu", 1.67),
    ("leaky", 0.005),
    ("leaky", 0.01),
    ("leaky", 0.015),
    ("nelu", 0.005),
    ("nelu", 0.01),
    ("nelu", 0.02),
    ("nelu", 0.05),
)
EPOCHS = 40
BATCH_SIZE = 16
STEPS_PER_EPOCH = 5303
OPTIMIZER_STEPS = EPOCHS * STEPS_PER_EPOCH
SPLIT_COUNTS = {"training": 84843, "validation": 9981, "testing": 11005}
ARCHIVE_SHA256 = "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58"
ARCHIVE_URL = "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def data_directory():
    local = Path(__file__).resolve().parents[1] / "data"
    if local.is_dir():
        return local
    return Path(sys.prefix) / "share" / "sugar-sedge" / "data"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_case(surrogate, alpha, seed):
    if (surrogate, alpha) not in VARIANTS:
        raise ValueError(f"Not a Table 2 configuration: {surrogate}, alpha={alpha}")
    if seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")


def run_name(surrogate, alpha, seed):
    validate_case(surrogate, alpha, seed)
    return f"{surrogate}_alpha_{alpha:g}_seed_{seed}"
