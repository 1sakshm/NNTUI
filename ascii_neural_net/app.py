from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass


if __package__ in {None, ""}:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


MENU_ITEMS = (
    ("Create Network", "create"),
    ("Train Network", "train"),
    ("Visualize Network", "visualize"),
    ("Load Dataset", "load"),
)


def configure_terminal_output(stream: object) -> None:
    """Enable Unicode box-drawing characters in Windows console sessions."""
    encoding = getattr(stream, "encoding", None)
    reconfigure = getattr(stream, "reconfigure", None)
    if encoding and encoding.lower().replace("_", "-") != "utf-8" and callable(reconfigure):
        reconfigure(encoding="utf-8")


def _paint(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def _center(text: str, width: int) -> str:
    return text.center(width)[:width]


def render_main_menu(width: int, height: int, selected_index: int, color: bool) -> str:
    """Render the main screen within the available terminal dimensions."""
    usable_width = max(2, width)
    inner_width = usable_width - 2
    title = "ASCII NEURAL NET"
    lines = [
        "╔" + "═" * inner_width + "╗",
        "║" + _center(title, inner_width) + "║",
        "╠" + "═" * inner_width + "╣",
    ]

    menu_lines = []
    for index, (label, _) in enumerate(MENU_ITEMS, start=1):
        text = f"[{index}] {label}"
        text = f"› {text}" if index - 1 == selected_index else f"  {text}"
        menu_lines.append(_paint(text, "1;36", color and index - 1 == selected_index))
    menu_lines.append(_paint("  [Q] Quit", "1;31", color))
    menu_lines.append("")
    menu_lines.append("  ↑/↓ navigate   Enter select   Q quit")

    available_rows = max(0, height - len(lines) - 1)
    top_padding = max(0, (available_rows - len(menu_lines)) // 2)
    body = [""] * top_padding + menu_lines
    body = body[:available_rows]
    body.extend([""] * (available_rows - len(body)))
    lines.extend("║" + line[:inner_width].ljust(inner_width) + "║" for line in body)
    lines.append("╚" + "═" * inner_width + "╝")
    return "\n".join(lines)


@dataclass
class Menu:
    selected_index: int = 0

    def handle_key(self, key: str) -> str | None:
        key = key.lower()
        if key == "up":
            self.selected_index = (self.selected_index - 1) % len(MENU_ITEMS)
        elif key == "down":
            self.selected_index = (self.selected_index + 1) % len(MENU_ITEMS)
        elif key in {"q", "esc"}:
            return "quit"
        elif key == "enter":
            return MENU_ITEMS[self.selected_index][1]
        elif key in {str(number) for number in range(1, len(MENU_ITEMS) + 1)}:
            self.selected_index = int(key) - 1
            return MENU_ITEMS[self.selected_index][1]
        return None


def _read_key() -> str:
    if os.name != "nt":
        return sys.stdin.read(1)

    import msvcrt

    key = msvcrt.getwch()
    if key in {"\x00", "\xe0"}:
        return {"H": "up", "P": "down"}.get(msvcrt.getwch(), "")
    return {"\r": "enter", "\x1b": "esc"}.get(key, key)


def _clear_and_draw(menu: Menu, status: str = "") -> None:
    size = shutil.get_terminal_size(fallback=(80, 24))
    color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    screen = render_main_menu(size.columns, size.lines - 1, menu.selected_index, color)
    sys.stdout.write("\033[2J\033[H" + screen)
    if status:
        sys.stdout.write("\n" + _paint(status, "33", color))
    sys.stdout.flush()


def render_action_screen(action: str, color: bool) -> str:
    """Return the completed Phase 2–4 screen selected from the menu."""
    if action == "create":
        from ascii_neural_net.builder import NetworkBuilder, render_builder

        return render_builder(NetworkBuilder(), color=color)
    if action == "train":
        from ascii_neural_net.xor_demo import train_xor

        lines = ["XOR TRAINING COMPLETE", "", "Input  Expected  Predicted  Confidence", "--------------------------------------"]
        for left, right, prediction, confidence in train_xor():
            expected = int(left != right)
            lines.append(f"{int(left)} {int(right)}       {expected}          {prediction}        {confidence:.3f}")
        return "\n".join(lines)
    if action == "visualize":
        from ascii_neural_net.visual_demo import visualize_xor_input

        return visualize_xor_input([0.0, 1.0], color=color)
    raise ValueError(f"No completed screen exists for action: {action}")


def _show_action_screen(action: str) -> None:
    color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    sys.stdout.write("\033[2J\033[H" + render_action_screen(action, color))
    sys.stdout.write("\n\nPress any key to return to the menu.")
    sys.stdout.flush()
    _read_key()


def run() -> int:
    configure_terminal_output(sys.stdout)
    menu = Menu()
    status = "Choose an option to begin."
    try:
        while True:
            _clear_and_draw(menu, status)
            action = menu.handle_key(_read_key())
            if action == "quit":
                return 0
            if action == "create":
                from ascii_neural_net.builder import run_builder

                run_builder()
                status = "Choose an option to begin."
                continue
            if action in {"train", "visualize"}:
                _show_action_screen(action)
                status = "Choose an option to begin."
                continue
            if action:
                label = next(label for label, name in MENU_ITEMS if name == action)
                status = f"{label} arrives in a later phase."
    except KeyboardInterrupt:
        return 0
    finally:
        sys.stdout.write("\033[0m\033[2J\033[H")
        sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(run())
