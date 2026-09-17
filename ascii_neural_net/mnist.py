"""MNIST IDX loading, ASCII digits, and from-scratch training helpers."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

from .network import NeuralNetwork


MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")


def download_mnist(directory: str | Path) -> Path:
    target = Path(directory); target.mkdir(parents=True, exist_ok=True)
    for name in MNIST_FILES:
        path = target / name
        if not path.exists(): urllib.request.urlretrieve(MNIST_URL + name, path)
    return target


def _read_images(path: Path) -> list[list[float]]:
    with gzip.open(path, "rb") as file:
        _, count, rows, columns = struct.unpack(">IIII", file.read(16)); raw = file.read()
    return [[value / 255.0 for value in raw[index * rows * columns:(index + 1) * rows * columns]] for index in range(count)]


def _read_labels(path: Path) -> list[int]:
    with gzip.open(path, "rb") as file:
        _, count = struct.unpack(">II", file.read(8)); return list(file.read(count))


def load_mnist(directory: str | Path, *, train: bool = True, limit: int | None = None) -> tuple[list[list[float]], list[int]]:
    root = Path(directory); prefix = "train" if train else "t10k"
    images, labels = _read_images(root / f"{prefix}-images-idx3-ubyte.gz"), _read_labels(root / f"{prefix}-labels-idx1-ubyte.gz")
    return images[:limit], labels[:limit]


def render_digit(pixels: list[float] | list[int], *, rows: int = 28, columns: int = 28) -> str:
    chars = " ░▒▓█"
    return "\n".join("".join(chars[min(4, round(float(value) * 4 if float(value) <= 1 else float(value) / 255 * 4))] for value in pixels[row * columns:(row + 1) * columns]) for row in range(rows))


def prediction_bars(probabilities: list[float]) -> str:
    prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
    lines = [f"Prediction: {prediction}", ""]
    for digit, probability in enumerate(probabilities):
        lines.append(f"{digit} {'█' * round(probability * 10):<10} {probability:.0%}")
    return "\n".join(lines)


def train_mnist(features: list[list[float]], labels: list[int], *, hidden_neurons: int = 32, epochs: int = 3) -> NeuralNetwork:
    network = NeuralNetwork([784, hidden_neurons, 10], ["relu", "softmax"], seed=4, initialization="he")
    targets = [[float(index == label) for index in range(10)] for label in labels]
    network.fit(features, targets, epochs=epochs, learning_rate=0.1, batch_size=16, loss="categorical_cross_entropy")
    return network
