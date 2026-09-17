import unittest

from ascii_neural_net.mnist import prediction_bars, render_digit


class MnistRenderingTests(unittest.TestCase):
    def test_digit_renderer_maps_bright_pixels_to_blocks(self):
        image = [0, 255, 0, 255]
        screen = render_digit(image, rows=2, columns=2)
        self.assertIn("█", screen)

    def test_prediction_bars_highlight_the_largest_probability(self):
        screen = prediction_bars([0.01] * 7 + [0.92, 0.03, 0.01])
        self.assertIn("Prediction: 7", screen)
        self.assertIn("92%", screen)


if __name__ == "__main__":
    unittest.main()
