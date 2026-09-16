"""Interactive network configuration for the terminal application."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from .network import NeuralNetwork, xor_dataset


ACTIVATION_OPTIONS = ("relu", "sigmoid", "tanh", "softmax")
LEARNING_RATES = (0.1, 0.3, 0.8, 1.0)
INITIALIZATIONS = ("xavier", "he")


@dataclass
class LayerConfig:
    kind: str
    neurons: int
    activation: str | None = None


@dataclass
class TrainingResult:
    loss: float
    predictions: list[int]


class NetworkBuilder:
    def __init__(self) -> None:
        self.layers = [LayerConfig("Input", 2), LayerConfig("Hidden", 4, "tanh"), LayerConfig("Output", 1, "sigmoid")]
        self.selected_index = 1
        self.learning_rate = 0.8
        self.initialization = "xavier"
        self.seed = 4

    @property
    def layer_sizes(self) -> list[int]:
        return [layer.neurons for layer in self.layers]

    @property
    def activations(self) -> list[str]:
        return [layer.activation for layer in self.layers[1:] if layer.activation is not None]

    def add_hidden_layer(self) -> None:
        insert_at = len(self.layers) - 1
        self.layers.insert(insert_at, LayerConfig("Hidden", 4, "relu"))
        self.selected_index = insert_at

    def delete_selected_layer(self) -> bool:
        if self.selected_index in {0, len(self.layers) - 1}:
            return False
        del self.layers[self.selected_index]
        self.selected_index = min(self.selected_index, len(self.layers) - 2)
        return True

    def adjust_selected_neurons(self, amount: int) -> None:
        layer = self.layers[self.selected_index]
        layer.neurons = max(1, layer.neurons + amount)

    def cycle_selected_activation(self) -> bool:
        layer = self.layers[self.selected_index]
        if layer.activation is None:
            return False
        layer.activation = ACTIVATION_OPTIONS[(ACTIVATION_OPTIONS.index(layer.activation) + 1) % len(ACTIVATION_OPTIONS)]
        return True

    def cycle_learning_rate(self) -> None:
        self.learning_rate = LEARNING_RATES[(LEARNING_RATES.index(self.learning_rate) + 1) % len(LEARNING_RATES)]

    def cycle_initialization(self) -> None:
        self.initialization = INITIALIZATIONS[(INITIALIZATIONS.index(self.initialization) + 1) % len(INITIALIZATIONS)]

    def randomize_weights(self) -> None:
        self.seed += 1

    def build(self) -> NeuralNetwork:
        return NeuralNetwork(self.layer_sizes, self.activations, seed=self.seed, initialization=self.initialization)

    def train_xor(self, *, epochs: int = 4000) -> TrainingResult:
        if self.layer_sizes[0] != 2 or self.layer_sizes[-1] != 1 or self.activations[-1] != "sigmoid":
            raise ValueError("XOR training requires 2 inputs, 1 sigmoid output, and any hidden layers.")
        features, labels = xor_dataset()
        network = self.build()
        history = network.fit(features, labels, epochs=epochs, learning_rate=self.learning_rate, batch_size=4)
        predictions = [int(network.predict(row)[0] >= 0.5) for row in features]
        return TrainingResult(history[-1], predictions)


def _paint(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def render_builder(builder: NetworkBuilder, *, color: bool) -> str:
    lines = ["NETWORK BUILDER", "", "Layers:"]
    for index, layer in enumerate(builder.layers, start=1):
        marker = "›" if index - 1 == builder.selected_index else " "
        details = f"{layer.kind:<7} {layer.neurons:>3}"
        if layer.activation:
            details += f"   {layer.activation.title()}"
        lines.append(_paint(f" {marker} [{index}] {details}", "1;36", color and index - 1 == builder.selected_index))
    lines.extend(
        [
            "",
            f"Learning rate: {builder.learning_rate}",
            f"Initialization: {builder.initialization.title()}",
            "",
            "[↑/↓] Select    [←/→] Neurons    [A] Add layer    [D] Delete layer",
            "[E] Cycle activation    [L] Learning rate    [I] Initialization",
            "[R] Randomize weights    [T] Train XOR    [B/Q] Back",
        ]
    )
    return "\n".join(lines)


def run_builder() -> None:
    from .app import _read_key

    builder = NetworkBuilder()
    status = "Edit a layer, then train it on XOR."
    while True:
        color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        sys.stdout.write("\033[2J\033[H" + render_builder(builder, color=color) + "\n\n" + status)
        sys.stdout.flush()
        key = _read_key().lower()
        if key in {"q", "b", "esc"}:
            return
        if key == "up":
            builder.selected_index = (builder.selected_index - 1) % len(builder.layers)
        elif key == "down":
            builder.selected_index = (builder.selected_index + 1) % len(builder.layers)
        elif key == "left":
            builder.adjust_selected_neurons(-1)
        elif key == "right":
            builder.adjust_selected_neurons(1)
        elif key == "a":
            builder.add_hidden_layer()
        elif key == "d":
            status = "Layer deleted." if builder.delete_selected_layer() else "Input and output layers cannot be deleted."
        elif key == "e":
            status = "Activation changed." if builder.cycle_selected_activation() else "Input layers have no activation."
        elif key == "l":
            builder.cycle_learning_rate()
        elif key == "i":
            builder.cycle_initialization()
        elif key == "r":
            builder.randomize_weights()
            status = f"Weights randomized with seed {builder.seed}."
        elif key == "t":
            try:
                result = builder.train_xor()
                status = f"Training complete — loss {result.loss:.4f}, predictions {result.predictions}."
            except ValueError as error:
                status = str(error)
