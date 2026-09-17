"""Live training metrics and an ASCII dashboard for XOR."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

from .network import NeuralNetwork, xor_dataset
from .visualizer import render_network


@dataclass
class TrainingResult:
    network: NeuralNetwork
    history: list[float]
    predictions: list[int]


def _accuracy(network: NeuralNetwork) -> float:
    features, labels = xor_dataset()
    correct = sum(int(network.predict(row)[0] >= 0.5) == int(label[0]) for row, label in zip(features, labels))
    return correct / len(features) * 100


def render_training_dashboard(network: NeuralNetwork, history: list[float], *, epoch: int, total_epochs: int, learning_rate: float, accuracy: float) -> str:
    recent = history[-20:] or [0.0]
    maximum = max(recent) or 1.0
    graph = "".join("█" * max(1, round(value / maximum * 12)) + "\n" for value in recent[-8:])
    return "\n".join([
        "TRAINING DASHBOARD",
        f"Epoch: {epoch} / {total_epochs}",
        f"Loss: {history[-1]:.4f}" if history else "Loss: --",
        f"Accuracy: {accuracy:.1f}%",
        f"Learning rate: {learning_rate}",
        "", "LOSS (recent epochs)", graph.rstrip(), "", render_network(network, [0.0, 1.0], color=False),
    ])


def train_xor_with_progress(*, epochs: int = 4000, on_progress: Callable[[TrainingResult], None] | None = None) -> TrainingResult:
    features, labels = xor_dataset()
    network = NeuralNetwork([2, 4, 1], ["tanh", "sigmoid"], seed=4)
    history: list[float] = []
    step = max(1, epochs // 20)
    for completed in range(0, epochs, step):
        history.extend(network.fit(features, labels, epochs=min(step, epochs - completed), learning_rate=0.8, batch_size=4))
        predictions = [int(network.predict(row)[0] >= 0.5) for row in features]
        result = TrainingResult(network, history, predictions)
        if on_progress:
            on_progress(result)
    return TrainingResult(network, history, [int(network.predict(row)[0] >= 0.5) for row in features])


def run_training_dashboard() -> None:
    def draw(result: TrainingResult) -> None:
        screen = render_training_dashboard(result.network, result.history, epoch=len(result.history), total_epochs=4000, learning_rate=0.8, accuracy=_accuracy(result.network))
        sys.stdout.write("\033[2J\033[H" + screen)
        sys.stdout.flush()
        time.sleep(0.04)
    train_xor_with_progress(on_progress=draw)
    sys.stdout.write("\n\nTraining complete. Press any key to return.")
    sys.stdout.flush()
    from .app import _read_key
    _read_key()
