import math
import unittest

from ascii_neural_net.network import (
    NeuralNetwork,
    binary_cross_entropy,
    categorical_cross_entropy,
    mean_squared_error,
    relu,
    sigmoid,
    softmax,
    tanh,
    xor_dataset,
)


class ActivationTests(unittest.TestCase):
    def test_scalar_activations_return_expected_values(self):
        self.assertEqual(relu(-2.0), 0.0)
        self.assertEqual(relu(3.0), 3.0)
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertAlmostEqual(tanh(0.0), 0.0)

    def test_softmax_returns_probabilities_that_sum_to_one(self):
        probabilities = softmax([1.0, 2.0, 3.0])

        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[2], probabilities[1])


class LossTests(unittest.TestCase):
    def test_loss_functions_score_perfect_predictions_as_zero(self):
        self.assertEqual(mean_squared_error([1.0, 0.0], [1.0, 0.0]), 0.0)
        self.assertAlmostEqual(binary_cross_entropy([1.0], [1.0]), 0.0)
        self.assertAlmostEqual(categorical_cross_entropy([0.0, 1.0], [0.0, 1.0]), 0.0)


class NeuralNetworkTests(unittest.TestCase):
    def test_prediction_has_the_configured_output_size(self):
        network = NeuralNetwork([2, 3, 1], ["relu", "sigmoid"], seed=7)

        prediction = network.predict([0.0, 1.0])

        self.assertEqual(len(prediction), 1)
        self.assertGreaterEqual(prediction[0], 0.0)
        self.assertLessEqual(prediction[0], 1.0)

    def test_network_learns_xor(self):
        features, labels = xor_dataset()
        network = NeuralNetwork([2, 4, 1], ["tanh", "sigmoid"], seed=4)

        history = network.fit(features, labels, epochs=4000, learning_rate=0.8, batch_size=4)
        predictions = [network.predict(row)[0] for row in features]

        self.assertLess(history[-1], 0.08)
        self.assertEqual([int(value >= 0.5) for value in predictions], [0, 1, 1, 0])

    def test_xor_demo_reports_all_four_learned_outputs(self):
        from ascii_neural_net.xor_demo import train_xor

        results = train_xor(epochs=4000)

        self.assertEqual([row[2] for row in results], [0, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
