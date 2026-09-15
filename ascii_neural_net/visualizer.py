"""ASCII rendering for fully connected neural networks."""

from __future__ import annotations

from typing import Sequence

from .network import NeuralNetwork


def _paint(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def _node(value: float, label: str) -> str:
    strength = min(abs(value), 1.0)
    glyph = "●" if strength >= 0.66 else "◉" if strength >= 0.20 else "○"
    return f"{glyph} {label} {value:+.2f}"


def _edge(weight: float, color: bool) -> str:
    magnitude = abs(weight)
    stroke = "━━━" if magnitude >= 0.70 else "━━" if magnitude >= 0.30 else "─"
    return _paint(f"{stroke} {weight:+.2f} {stroke}", "32" if weight >= 0.0 else "31", color)


def _layer_title(layer_index: int, total_layers: int) -> str:
    if layer_index == 0:
        return "INPUT"
    if layer_index == total_layers - 1:
        return "OUTPUT"
    return f"HIDDEN {layer_index}"


def _labels(layer_index: int, count: int, total_layers: int) -> list[str]:
    prefix = "I" if layer_index == 0 else "O" if layer_index == total_layers - 1 else "H"
    return [f"{prefix}{index + 1}" for index in range(count)]


def render_network(network: NeuralNetwork, inputs: Sequence[float], *, color: bool = False) -> str:
    """Render activations and every weighted connection after one inference pass."""
    outputs = network.predict(inputs)
    activations = [list(inputs), *[layer.outputs for layer in network.layers]]
    labels = [_labels(index, len(values), len(activations)) for index, values in enumerate(activations)]
    column_width = 24
    lines = ["ASCII NEURAL NET  •  LIVE ACTIVATIONS", ""]
    lines.append("".join(_layer_title(index, len(activations)).ljust(column_width) for index in range(len(activations))).rstrip())
    for row in range(max(map(len, activations))):
        line = ""
        for layer_index, values in enumerate(activations):
            content = _node(values[row], labels[layer_index][row]) if row < len(values) else ""
            line += content.ljust(column_width)
        lines.append(line.rstrip())

    lines.extend(["", "CONNECTIONS  (+ positive / - negative)"])
    for layer_index, layer in enumerate(network.layers):
        source_title = _layer_title(layer_index, len(activations))
        target_title = _layer_title(layer_index + 1, len(activations))
        lines.append(f"{source_title} → {target_title}")
        for target_index, weights in enumerate(layer.weights):
            target = _node(layer.outputs[target_index], labels[layer_index + 1][target_index])
            for source_index, weight in enumerate(weights):
                source = _node(activations[layer_index][source_index], labels[layer_index][source_index])
                lines.append(f"  {source} {_edge(weight, color)}▶ {target}")
    lines.extend(["", f"OUTPUT: {', '.join(f'{value:.3f}' for value in outputs)}"])
    return "\n".join(lines)
