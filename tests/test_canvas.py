import unittest
from ascii_neural_net.canvas import DrawingCanvas, render_canvas

class CanvasTests(unittest.TestCase):
    def test_canvas_draws_moves_clears_and_saves(self):
        canvas = DrawingCanvas()
        canvas.draw(); canvas.move("right"); canvas.draw()
        self.assertEqual(sum(canvas.pixels), 2.0)
        self.assertIn("█", render_canvas(canvas))
        canvas.clear()
        self.assertEqual(sum(canvas.pixels), 0.0)
