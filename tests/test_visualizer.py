import subprocess
import sys
import unittest

from ascii_neural_net.network import NeuralNetwork
from ascii_neural_net.visualizer import render_network


class NetworkVisualizerTests(unittest.TestCase):
    def setUp(self):
        self.network = NeuralNetwork([2, 2, 1], ["relu", "sigmoid"], seed=1)
        self.network.layers[0].weights = [[0.9, -0.2], [-0.7, 0.1]]
        self.network.layers[1].weights = [[0.8, -0.6]]
        self.network.predict([1.0, 0.0])

    def test_rendering_includes_layers_nodes_and_connections(self):
        screen = render_network(self.network, [1.0, 0.0], color=False)

        self.assertIn("LIVE ACTIVATIONS", screen)
        self.assertIn("INPUT", screen)
        self.assertIn("HIDDEN 1", screen)
        self.assertIn("OUTPUT", screen)
        self.assertIn("I1", screen)
        self.assertIn("H1", screen)
        self.assertIn("O1", screen)
        self.assertIn("+0.90", screen)
        self.assertIn("-0.70", screen)

    def test_high_activations_render_as_bright_nodes(self):
        screen = render_network(self.network, [1.0, 0.0], color=False)

        self.assertIn("● I1", screen)
        self.assertIn("○ I2", screen)

    def test_colored_rendering_marks_positive_and_negative_edges(self):
        screen = render_network(self.network, [1.0, 0.0], color=True)

        self.assertIn("\033[32m", screen)
        self.assertIn("\033[31m", screen)

    def test_xor_visual_demo_renders_a_prediction_for_an_input(self):
        from ascii_neural_net.visual_demo import visualize_xor_input

        screen = visualize_xor_input([0.0, 1.0])

        self.assertIn("LIVE ACTIVATIONS", screen)
        self.assertIn("OUTPUT: 1.000", screen)

    def test_xor_visual_demo_module_runs_from_the_command_line(self):
        result = subprocess.run(
            [sys.executable, "-m", "ascii_neural_net.visual_demo", "0", "1"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LIVE ACTIVATIONS", result.stdout)


if __name__ == "__main__":
    unittest.main()
