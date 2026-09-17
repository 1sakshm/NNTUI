import importlib.util
import sys
import unittest
from pathlib import Path

from ascii_neural_net.app import Menu, configure_terminal_output, render_action_screen, render_main_menu


class MenuTests(unittest.TestCase):
    def test_numeric_choice_selects_a_menu_item(self):
        menu = Menu()

        self.assertEqual(menu.handle_key("3"), "visualize")

    def test_quit_choice_returns_quit_action(self):
        menu = Menu()

        self.assertEqual(menu.handle_key("q"), "quit")

    def test_arrow_keys_move_the_highlight(self):
        menu = Menu()

        menu.handle_key("down")

        self.assertEqual(menu.selected_index, 1)


class MainMenuRenderingTests(unittest.TestCase):
    def test_main_menu_includes_title_and_all_actions(self):
        screen = render_main_menu(width=60, height=20, selected_index=0, color=False)

        self.assertIn("ASCII NEURAL NET", screen)
        self.assertIn("[1] Create Network", screen)
        self.assertIn("[2] Train Network", screen)
        self.assertIn("[3] Visualize Network", screen)
        self.assertIn("[4] Load Dataset", screen)
        self.assertIn("[Q] Quit", screen)

    def test_rendering_never_exceeds_available_width(self):
        screen = render_main_menu(width=10, height=12, selected_index=0, color=False)

        self.assertTrue(all(len(line) <= 10 for line in screen.splitlines()))


class TerminalOutputTests(unittest.TestCase):
    def test_terminal_output_uses_utf8_for_box_drawing_characters(self):
        class Stream:
            def __init__(self):
                self.encoding = "cp1252"
                self.reconfigured_to = None

            def reconfigure(self, *, encoding):
                self.reconfigured_to = encoding

        stream = Stream()

        configure_terminal_output(stream)

        self.assertEqual(stream.reconfigured_to, "utf-8")


class PhaseIntegrationTests(unittest.TestCase):
    def test_create_menu_action_renders_the_network_builder(self):
        screen = render_action_screen("create", color=False)

        self.assertIn("NETWORK BUILDER", screen)
        self.assertIn("[T] Train XOR", screen)

    def test_load_dataset_menu_action_renders_the_dataset_browser(self):
        screen = render_action_screen("load", color=False)

        self.assertIn("DATASET BROWSER", screen)
        self.assertIn("CIRCLES", screen)

    def test_train_menu_action_runs_the_xor_training_workflow(self):
        screen = render_action_screen("train", color=False)

        self.assertIn("XOR TRAINING COMPLETE", screen)
        self.assertIn("0 1", screen)
        self.assertIn("1", screen)

    def test_visualize_menu_action_renders_live_network_activations(self):
        screen = render_action_screen("visualize", color=False)

        self.assertIn("LIVE ACTIVATIONS", screen)
        self.assertIn("CONNECTIONS", screen)
        self.assertIn("OUTPUT: 1.000", screen)

    def test_direct_app_launch_can_open_the_visualization_menu(self):
        path = Path("ascii_neural_net/app.py").resolve()
        spec = importlib.util.spec_from_file_location("standalone_app", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        self.assertIn("LIVE ACTIVATIONS", module.render_action_screen("visualize", color=False))


if __name__ == "__main__":
    unittest.main()
