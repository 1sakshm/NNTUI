"""Interactive viewer for the built-in two-dimensional datasets."""

from __future__ import annotations

import os
import sys

from .datasets import builtin_dataset, render_dataset


DATASET_KEYS = {"1": "xor", "2": "and", "3": "or", "4": "linear", "5": "circles", "6": "spirals"}


def render_dataset_browser(name: str = "circles") -> str:
    dataset = builtin_dataset(name, samples=80, seed=3)
    plot = render_dataset(dataset, width=46, height=14)
    return "\n".join(
        [
            "DATASET BROWSER",
            "",
            "[1] XOR  [2] AND  [3] OR  [4] Linear  [5] Circles  [6] Spirals",
            "[B/Q] Back    CSV: py -3.12 -m ascii_neural_net.dataset_demo path\\to\\data.csv",
            "",
            plot,
        ]
    )


def run_dataset_browser() -> None:
    from .app import _read_key

    selected = "circles"
    while True:
        sys.stdout.write("\033[2J\033[H" + render_dataset_browser(selected))
        sys.stdout.flush()
        key = _read_key().lower()
        if key in {"q", "b", "esc"}:
            return
        if key in DATASET_KEYS:
            selected = DATASET_KEYS[key]
