from __future__ import annotations
import argparse
from .mnist import download_mnist, load_mnist, prediction_bars, render_digit, train_mnist

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data", default="data/mnist"); parser.add_argument("--download", action="store_true"); parser.add_argument("--limit", type=int, default=500); parser.add_argument("--epochs", type=int, default=3); args = parser.parse_args()
    if args.download: download_mnist(args.data)
    features, labels = load_mnist(args.data, limit=args.limit)
    network = train_mnist(features, labels, epochs=args.epochs)
    print(render_digit(features[0])); print(); print(f"Actual: {labels[0]}"); print(prediction_bars(network.predict(features[0])))

if __name__ == "__main__": main()
