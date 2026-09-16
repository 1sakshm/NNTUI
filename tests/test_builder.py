import unittest

from ascii_neural_net.builder import NetworkBuilder, render_builder


class NetworkBuilderTests(unittest.TestCase):
    def test_builder_starts_with_an_xor_compatible_network(self):
        builder = NetworkBuilder()

        self.assertEqual(builder.layer_sizes, [2, 4, 1])
        self.assertEqual(builder.activations, ["tanh", "sigmoid"])

    def test_builder_can_add_edit_and_delete_a_hidden_layer(self):
        builder = NetworkBuilder()

        builder.add_hidden_layer()
        builder.selected_index = 2
        builder.adjust_selected_neurons(3)
        builder.cycle_selected_activation()
        builder.delete_selected_layer()

        self.assertEqual(builder.layer_sizes, [2, 4, 1])

    def test_builder_cycles_learning_rate_and_initialization(self):
        builder = NetworkBuilder()
        original_rate = builder.learning_rate
        original_initialization = builder.initialization

        builder.cycle_learning_rate()
        builder.cycle_initialization()

        self.assertNotEqual(builder.learning_rate, original_rate)
        self.assertNotEqual(builder.initialization, original_initialization)

    def test_builder_trains_its_configured_xor_network(self):
        builder = NetworkBuilder()

        result = builder.train_xor(epochs=4000)

        self.assertLess(result.loss, 0.08)
        self.assertEqual(result.predictions, [0, 1, 1, 0])

    def test_rendered_builder_shows_controls_and_current_settings(self):
        screen = render_builder(NetworkBuilder(), color=False)

        self.assertIn("NETWORK BUILDER", screen)
        self.assertIn("[A] Add layer", screen)
        self.assertIn("[D] Delete layer", screen)
        self.assertIn("[E] Cycle activation", screen)
        self.assertIn("[R] Randomize weights", screen)
        self.assertIn("[T] Train XOR", screen)


if __name__ == "__main__":
    unittest.main()
