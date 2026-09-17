import tempfile
import unittest
from pathlib import Path

from ascii_neural_net.datasets import (
    batches,
    builtin_dataset,
    load_csv,
    normalize_features,
    render_dataset,
    train_test_split,
)


class DatasetTests(unittest.TestCase):
    def test_all_builtin_datasets_are_available(self):
        for name in ("xor", "and", "or", "linear", "circles", "spirals"):
            dataset = builtin_dataset(name, samples=20, seed=3)
            self.assertTrue(dataset.features)
            self.assertEqual(len(dataset.features), len(dataset.labels))

    def test_csv_loader_maps_text_labels_and_reads_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            path.write_text("x,y,label\n0,1,no\n1,0,yes\n", encoding="utf-8")
            dataset = load_csv(path)

        self.assertEqual(dataset.features, [[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(dataset.labels, [0, 1])
        self.assertEqual(dataset.label_names, ["no", "yes"])

    def test_split_normalization_and_batches_preserve_samples(self):
        dataset = builtin_dataset("linear", samples=20, seed=3)
        train, test = train_test_split(dataset, test_ratio=0.25, seed=4)
        normalized = normalize_features(train)
        grouped = list(batches(normalized, batch_size=4))

        self.assertEqual(len(train.features), 15)
        self.assertEqual(len(test.features), 5)
        self.assertEqual(sum(len(group.features) for group in grouped), 15)
        self.assertAlmostEqual(sum(row[0] for row in normalized.features) / len(normalized.features), 0.0)

    def test_2d_renderer_draws_both_classes(self):
        screen = render_dataset(builtin_dataset("circles", samples=30, seed=2), width=24, height=10)

        self.assertIn("●", screen)
        self.assertIn("○", screen)


if __name__ == "__main__":
    unittest.main()
