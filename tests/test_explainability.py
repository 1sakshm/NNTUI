import unittest
from ascii_neural_net.explainability import explain_prediction
from ascii_neural_net.network import NeuralNetwork

class ExplainabilityTests(unittest.TestCase):
 def test_explanation_contains_saliency_confidence_and_hidden_neurons(self):
  network=NeuralNetwork([2,3,1],["tanh","sigmoid"],seed=4)
  report=explain_prediction(network,[0.0,1.0])
  self.assertIn("Confidence",report)
  self.assertIn("Strongest neurons",report)
  self.assertIn("INPUT SALIENCY",report)
