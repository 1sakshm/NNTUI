import unittest

from ascii_neural_net.network import NeuralNetwork, xor_dataset
from ascii_neural_net.training import render_training_dashboard, train_xor_with_progress


class TrainingVisualizationTests(unittest.TestCase):
    def test_dashboard_shows_training_metrics_and_live_network(self):
        network = NeuralNetwork([2, 4, 1], ["tanh", "sigmoid"], seed=4)
        network.predict([0.0, 1.0])

        screen = render_training_dashboard(network, [0.9, 0.6, 0.2], epoch=3, total_epochs=10, learning_rate=0.8, accuracy=75.0)

        self.assertIn("TRAINING DASHBOARD", screen)
        self.assertIn("Epoch: 3 / 10", screen)
        self.assertIn("Accuracy: 75.0%", screen)
        self.assertIn("LIVE ACTIVATIONS", screen)

    def test_xor_training_reports_progress_and_learns(self):
        snapshots = []
        result = train_xor_with_progress(epochs=400, on_progress=snapshots.append)

        self.assertTrue(snapshots)
        self.assertEqual(result.predictions, [0, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
