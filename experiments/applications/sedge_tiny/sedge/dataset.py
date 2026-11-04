# Adapted from the Apache-2.0 S-Edge loader; local paths and cache handling simplified.
# See THIRD_PARTY.md and licenses/Apache-2.0.txt.

import argparse
import tarfile
import urllib.request
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from .protocol import ARCHIVE_SHA256, ARCHIVE_URL, SPLIT_COUNTS, sha256


LABELS = (
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
)


def prepare_dataset(destination, archive=None):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Dataset destination already exists: {destination}")
    archive = (
        Path(archive)
        if archive
        else destination.parent / "speech_commands_v0.02.tar.gz"
    )
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(".part")
        with (
            partial.open("xb") as output,
            urllib.request.urlopen(ARCHIVE_URL) as response,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if sha256(partial) != ARCHIVE_SHA256:
            raise ValueError("Speech Commands archive checksum mismatch")
        partial.rename(archive)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Speech Commands archive checksum mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            path = destination / member.name
            if not path.resolve().is_relative_to(destination.resolve()) or not (
                member.isfile() or member.isdir()
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
        handle.extractall(destination)


def split_paths(root, subset):
    root = Path(root)

    def load_list(*filenames):
        output = []
        for filename in filenames:
            with (root / filename).open() as handle:
                output += [root / line.strip() for line in handle]
        return output

    if subset == "validation":
        paths = load_list("validation_list.txt")
    elif subset == "testing":
        paths = load_list("testing_list.txt")
    elif subset == "training":
        paths = sorted(root.glob("*/*.wav"))
        paths = [
            path
            for path in paths
            if "_nohash_" in str(path) and "_background_noise_" not in str(path)
        ]
        excludes = set(load_list("validation_list.txt", "testing_list.txt"))
        paths = [path for path in paths if path not in excludes]
    else:
        raise ValueError(f"Unknown split: {subset}")
    if len(paths) != SPLIT_COUNTS[subset]:
        raise ValueError(f"Unexpected {subset} split size: {len(paths)}")
    return paths


class SpeechCommands(Dataset):
    def __init__(self, root, subset, normalization=None):
        self.paths = split_paths(root, subset)
        audios = []
        labels = []
        for path in tqdm(self.paths, desc=f"Loading {subset}"):
            waveform, sample_rate = torchaudio.load(path)
            if sample_rate != 16000 or waveform.shape[0] != 1:
                raise ValueError(f"Expected mono 16 kHz audio: {path}")
            audios.append(waveform)
            labels.append(LABELS.index(path.parent.name))
        batch = [item.t() for item in audios]
        self.audios = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=0.0
        )
        self.labels = torch.tensor(labels)
        if self.audios.shape != (len(self.paths), 16000, 1):
            raise ValueError(f"Unexpected padded audio shape: {self.audios.shape}")
        if subset == "training":
            self.mean = self.audios.mean()
            self.std = self.audios.std()
        elif normalization is not None:
            self.mean, self.std = normalization
        else:
            raise ValueError(
                "Validation and test data require training-set normalization"
            )
        self.audios = (self.audios - self.mean) / (self.std + 1e-5)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.audios[index], self.labels[index]


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify Speech Commands v0.02"
    )
    parser.add_argument("--output", type=Path, default=Path("datasets/speech_commands"))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    prepare_dataset(args.output, args.archive)


if __name__ == "__main__":
    main()
