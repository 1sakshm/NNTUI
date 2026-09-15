"""Render a trained XOR network and its live inference values."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .app import configure_terminal_output
from .network import NeuralNetwork, xor_dataset
from .visualizer import render_network


def visualize_xor_input(inputs: Sequence[float], *, color: bool = False) -> str:
    if len(inputs) != 2:
        raise ValueError("XOR visualizer requires exactly two input values")
    features, labels = xor_dataset()
    network = NeuralNetwork([2, 4, 1], ["tanh", "sigmoid"], seed=4)
    network.fit(features, labels, epochs=4000, learning_rate=0.8, batch_size=4)
    return render_network(network, inputs, color=color)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a trained XOR neural network.")
    parser.add_argument("left", type=float, nargs="?", default=0.0, help="First XOR input (default: 0)")
    parser.add_argument("right", type=float, nargs="?", default=1.0, help="Second XOR input (default: 1)")
    arguments = parser.parse_args()
    configure_terminal_output(sys.stdout)
    print(visualize_xor_input([arguments.left, arguments.right], color=sys.stdout.isatty()))


if __name__ == "__main__":
    main()
