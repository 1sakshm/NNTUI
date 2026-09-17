"""Load, prepare, and inspect a CSV classification dataset."""

from __future__ import annotations

import argparse

from .datasets import load_csv, normalize_features, render_dataset, train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a CSV dataset for ASCII Neural Net.")
    parser.add_argument("path", help="CSV file path")
    parser.add_argument("--target-column", type=int, default=-1)
    parser.add_argument("--no-header", action="store_true")
    arguments = parser.parse_args()
    dataset = load_csv(arguments.path, target_column=arguments.target_column, has_header=not arguments.no_header)
    train, test = train_test_split(dataset)
    train = normalize_features(train)
    print(f"Loaded {dataset.name}: {len(dataset.features)} rows, {len(dataset.features[0])} features, labels {dataset.label_names}")
    print(f"Train: {len(train.features)} rows; test: {len(test.features)} rows")
    if len(train.features[0]) >= 2:
        print()
        print(render_dataset(train))


if __name__ == "__main__":
    main()
