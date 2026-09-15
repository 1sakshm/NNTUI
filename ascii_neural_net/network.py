"""A small, dependency-free feed-forward neural-network implementation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence


Vector = list[float]
Matrix = list[Vector]
EPSILON = 1e-12


def relu(value: float) -> float:
    return max(0.0, value)


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def tanh(value: float) -> float:
    return math.tanh(value)


def softmax(values: Sequence[float]) -> Vector:
    largest = max(values)
    exponentials = [math.exp(value - largest) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def mean_squared_error(targets: Sequence[float], predictions: Sequence[float]) -> float:
    return sum((target - prediction) ** 2 for target, prediction in zip(targets, predictions)) / len(targets)


def binary_cross_entropy(targets: Sequence[float], predictions: Sequence[float]) -> float:
    return -sum(
        target * math.log(max(prediction, EPSILON))
        + (1.0 - target) * math.log(max(1.0 - prediction, EPSILON))
        for target, prediction in zip(targets, predictions)
    ) / len(targets)


def categorical_cross_entropy(targets: Sequence[float], predictions: Sequence[float]) -> float:
    return -sum(target * math.log(max(prediction, EPSILON)) for target, prediction in zip(targets, predictions))


ACTIVATIONS: dict[str, Callable[[float], float]] = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
}


@dataclass
class DenseLayer:
    input_size: int
    output_size: int
    activation: str
    rng: random.Random
    weights: Matrix = field(init=False)
    biases: Vector = field(init=False)
    inputs: Vector = field(default_factory=list, init=False)
    pre_activations: Vector = field(default_factory=list, init=False)
    outputs: Vector = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.activation not in {*ACTIVATIONS, "softmax"}:
            raise ValueError(f"Unsupported activation: {self.activation}")
        limit = math.sqrt(6.0 / (self.input_size + self.output_size))
        self.weights = [
            [self.rng.uniform(-limit, limit) for _ in range(self.input_size)]
            for _ in range(self.output_size)
        ]
        self.biases = [0.0] * self.output_size

    def forward(self, inputs: Sequence[float]) -> Vector:
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, received {len(inputs)}")
        self.inputs = list(inputs)
        self.pre_activations = [
            sum(weight * value for weight, value in zip(row, self.inputs)) + bias
            for row, bias in zip(self.weights, self.biases)
        ]
        if self.activation == "softmax":
            self.outputs = softmax(self.pre_activations)
        else:
            function = ACTIVATIONS[self.activation]
            self.outputs = [function(value) for value in self.pre_activations]
        return self.outputs

    def activation_gradient(self, output_gradient: Sequence[float]) -> Vector:
        if self.activation == "relu":
            return [gradient if pre_activation > 0.0 else 0.0 for gradient, pre_activation in zip(output_gradient, self.pre_activations)]
        if self.activation == "sigmoid":
            return [gradient * output * (1.0 - output) for gradient, output in zip(output_gradient, self.outputs)]
        if self.activation == "tanh":
            return [gradient * (1.0 - output**2) for gradient, output in zip(output_gradient, self.outputs)]
        weighted_average = sum(gradient * output for gradient, output in zip(output_gradient, self.outputs))
        return [output * (gradient - weighted_average) for gradient, output in zip(output_gradient, self.outputs)]


class NeuralNetwork:
    """A fully connected feed-forward network trained with backpropagation."""

    def __init__(self, layer_sizes: Sequence[int], activations: Sequence[str], *, seed: int | None = None) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("A network needs input and output layers")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Provide one activation for every non-input layer")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError("Layer sizes must be positive")
        self.rng = random.Random(seed)
        self.layers = [
            DenseLayer(input_size, output_size, activation, self.rng)
            for input_size, output_size, activation in zip(layer_sizes, layer_sizes[1:], activations)
        ]

    def predict(self, inputs: Sequence[float]) -> Vector:
        values = list(inputs)
        for layer in self.layers:
            values = layer.forward(values)
        return values

    def predict_batch(self, features: Sequence[Sequence[float]]) -> Matrix:
        return [self.predict(row) for row in features]

    def _loss(self, targets: Sequence[float], predictions: Sequence[float], name: str) -> float:
        if name == "mse":
            return mean_squared_error(targets, predictions)
        if name == "binary_cross_entropy":
            return binary_cross_entropy(targets, predictions)
        if name == "categorical_cross_entropy":
            return categorical_cross_entropy(targets, predictions)
        raise ValueError(f"Unsupported loss: {name}")

    def _output_delta(self, targets: Sequence[float], predictions: Sequence[float], loss: str) -> Vector:
        final_layer = self.layers[-1]
        if (loss == "binary_cross_entropy" and final_layer.activation == "sigmoid") or (
            loss == "categorical_cross_entropy" and final_layer.activation == "softmax"
        ):
            return [prediction - target for target, prediction in zip(targets, predictions)]
        if loss == "mse":
            gradient = [2.0 * (prediction - target) / len(targets) for target, prediction in zip(targets, predictions)]
            return final_layer.activation_gradient(gradient)
        raise ValueError(f"Loss {loss} is incompatible with {final_layer.activation} output")

    def _gradients(self, targets: Sequence[float], predictions: Sequence[float], loss: str) -> tuple[list[Matrix], list[Vector]]:
        weight_gradients: list[Matrix] = []
        bias_gradients: list[Vector] = []
        delta = self._output_delta(targets, predictions, loss)

        for position in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[position]
            weight_gradients.insert(0, [[error * value for value in layer.inputs] for error in delta])
            bias_gradients.insert(0, list(delta))
            if position:
                previous_layer = self.layers[position - 1]
                propagated = [
                    sum(layer.weights[neuron][input_index] * delta[neuron] for neuron in range(layer.output_size))
                    for input_index in range(layer.input_size)
                ]
                delta = previous_layer.activation_gradient(propagated)
        return weight_gradients, bias_gradients

    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[Sequence[float]],
        *,
        epochs: int,
        learning_rate: float,
        batch_size: int = 1,
        loss: str = "binary_cross_entropy",
    ) -> list[float]:
        if len(features) != len(labels) or not features:
            raise ValueError("Features and labels must be non-empty and have equal length")
        if epochs <= 0 or learning_rate <= 0 or batch_size <= 0:
            raise ValueError("Epochs, learning rate, and batch size must be positive")

        history: list[float] = []
        indices = list(range(len(features)))
        for _ in range(epochs):
            self.rng.shuffle(indices)
            epoch_loss = 0.0
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                accumulated_weights = [[[0.0] * layer.input_size for _ in range(layer.output_size)] for layer in self.layers]
                accumulated_biases = [[0.0] * layer.output_size for layer in self.layers]
                for index in batch:
                    prediction = self.predict(features[index])
                    epoch_loss += self._loss(labels[index], prediction, loss)
                    weight_gradients, bias_gradients = self._gradients(labels[index], prediction, loss)
                    for layer_index, gradients in enumerate(weight_gradients):
                        for neuron_index, row in enumerate(gradients):
                            for input_index, gradient in enumerate(row):
                                accumulated_weights[layer_index][neuron_index][input_index] += gradient
                        for neuron_index, gradient in enumerate(bias_gradients[layer_index]):
                            accumulated_biases[layer_index][neuron_index] += gradient
                scale = learning_rate / len(batch)
                for layer_index, layer in enumerate(self.layers):
                    for neuron_index in range(layer.output_size):
                        for input_index in range(layer.input_size):
                            layer.weights[neuron_index][input_index] -= scale * accumulated_weights[layer_index][neuron_index][input_index]
                        layer.biases[neuron_index] -= scale * accumulated_biases[layer_index][neuron_index]
            history.append(epoch_loss / len(features))
        return history


def xor_dataset() -> tuple[Matrix, Matrix]:
    return [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [[0.0], [1.0], [1.0], [0.0]]
