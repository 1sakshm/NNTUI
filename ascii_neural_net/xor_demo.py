"""Train the from-scratch network on the XOR truth table."""

from __future__ import annotations

from .network import NeuralNetwork, xor_dataset


def train_xor(epochs: int = 4000) -> list[tuple[float, float, int, float]]:
    features, labels = xor_dataset()
    network = NeuralNetwork([2, 4, 1], ["tanh", "sigmoid"], seed=4)
    network.fit(features, labels, epochs=epochs, learning_rate=0.8, batch_size=4)
    return [
        (row[0], row[1], int(network.predict(row)[0] >= 0.5), network.predict(row)[0])
        for row in features
    ]


def main() -> None:
    print("XOR training complete\n")
    print("Input  Expected  Predicted  Confidence")
    print("--------------------------------------")
    for left, right, prediction, confidence in train_xor():
        expected = int(left != right)
        print(f"{int(left)} {int(right)}       {expected}          {prediction}        {confidence:.3f}")


if __name__ == "__main__":
    main()
