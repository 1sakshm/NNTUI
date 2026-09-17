"""Built-in datasets, CSV loading, and small-data preparation helpers."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


@dataclass
class Dataset:
    name: str
    features: list[list[float]]
    labels: list[int]
    label_names: list[str]


def _logic(name: str) -> Dataset:
    features = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    operations = {"xor": [0, 1, 1, 0], "and": [0, 0, 0, 1], "or": [0, 1, 1, 1]}
    return Dataset(name.upper(), features, operations[name], ["0", "1"])


def builtin_dataset(name: str, *, samples: int = 100, seed: int = 1) -> Dataset:
    name = name.lower()
    if name in {"xor", "and", "or"}:
        return _logic(name)
    rng = random.Random(seed)
    features: list[list[float]] = []
    labels: list[int] = []
    if name == "linear":
        for _ in range(samples):
            x, y = rng.uniform(-1, 1), rng.uniform(-1, 1)
            features.append([x, y])
            labels.append(int(x + y > 0))
    elif name == "circles":
        for _ in range(samples):
            angle = rng.uniform(0, math.tau)
            label = rng.randrange(2)
            radius = rng.uniform(0.1, 0.45) if label == 0 else rng.uniform(0.65, 1.0)
            features.append([radius * math.cos(angle), radius * math.sin(angle)])
            labels.append(label)
    elif name in {"spiral", "spirals"}:
        for index in range(samples):
            label = index % 2
            radius = (index // 2) / max(1, samples // 2 - 1)
            angle = radius * math.pi * 3 + label * math.pi + rng.uniform(-0.15, 0.15)
            features.append([radius * math.cos(angle), radius * math.sin(angle)])
            labels.append(label)
        name = "spirals"
    else:
        raise ValueError(f"Unknown built-in dataset: {name}")
    return Dataset(name.title(), features, labels, ["Class 0", "Class 1"])


def load_csv(path: str | Path, *, target_column: int = -1, has_header: bool = True) -> Dataset:
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    if has_header and rows:
        rows = rows[1:]
    if not rows:
        raise ValueError("CSV contains no data rows")
    labels_by_name: dict[str, int] = {}
    features: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        if not row:
            continue
        column = target_column if target_column >= 0 else len(row) + target_column
        if not 0 <= column < len(row):
            raise ValueError("Target column is outside the CSV row")
        label = row[column]
        if label not in labels_by_name:
            labels_by_name[label] = len(labels_by_name)
        features.append([float(value) for index, value in enumerate(row) if index != column])
        labels.append(labels_by_name[label])
    return Dataset(Path(path).stem, features, labels, list(labels_by_name))


def train_test_split(dataset: Dataset, *, test_ratio: float = 0.2, seed: int = 1) -> tuple[Dataset, Dataset]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("Test ratio must be between 0 and 1")
    indices = list(range(len(dataset.features)))
    random.Random(seed).shuffle(indices)
    test_count = max(1, round(len(indices) * test_ratio))
    def pick(selected: Sequence[int], suffix: str) -> Dataset:
        return Dataset(f"{dataset.name} {suffix}", [dataset.features[i] for i in selected], [dataset.labels[i] for i in selected], dataset.label_names)
    return pick(indices[test_count:], "train"), pick(indices[:test_count], "test")


def normalize_features(dataset: Dataset) -> Dataset:
    columns = list(zip(*dataset.features))
    means = [sum(column) / len(column) for column in columns]
    deviations = [math.sqrt(sum((value - mean) ** 2 for value in column) / len(column)) or 1.0 for column, mean in zip(columns, means)]
    features = [[(value - means[index]) / deviations[index] for index, value in enumerate(row)] for row in dataset.features]
    return Dataset(dataset.name, features, list(dataset.labels), list(dataset.label_names))


def batches(dataset: Dataset, *, batch_size: int) -> Iterator[Dataset]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(dataset.features), batch_size):
        yield Dataset(dataset.name, dataset.features[start : start + batch_size], dataset.labels[start : start + batch_size], dataset.label_names)


def render_dataset(dataset: Dataset, *, width: int = 40, height: int = 16) -> str:
    if not dataset.features or len(dataset.features[0]) < 2:
        raise ValueError("ASCII scatter plots require at least two features")
    xs, ys = [row[0] for row in dataset.features], [row[1] for row in dataset.features]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for (x, y), label in zip(dataset.features, dataset.labels):
        column = round((x - x_min) / (x_max - x_min or 1.0) * (width - 1))
        row = height - 1 - round((y - y_min) / (y_max - y_min or 1.0) * (height - 1))
        grid[row][column] = "●" if label % 2 else "○"
    lines = [f"{dataset.name.upper()}  •  {len(dataset.features)} samples", *["".join(row) for row in grid], "○ Class 0   ● Class 1"]
    return "\n".join(lines)
