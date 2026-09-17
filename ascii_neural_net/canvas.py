"""Interactive 28×28 digit canvas for MNIST inference."""
from __future__ import annotations
import sys
from pathlib import Path
from .mnist import prediction_bars, render_digit

class DrawingCanvas:
    def __init__(self, rows: int = 28, columns: int = 28):
        self.rows, self.columns = rows, columns; self.x, self.y = columns // 2, rows // 2; self.pixels = [0.0] * (rows * columns)
    def draw(self): self.pixels[self.y * self.columns + self.x] = 1.0
    def clear(self): self.pixels = [0.0] * len(self.pixels)
    def move(self, direction: str):
        dx, dy = {"left":(-1,0),"right":(1,0),"up":(0,-1),"down":(0,1)}.get(direction,(0,0)); self.x=max(0,min(self.columns-1,self.x+dx)); self.y=max(0,min(self.rows-1,self.y+dy)); self.draw()
    def save(self, path: str | Path): Path(path).write_text("\n".join("".join("1" if v else "0" for v in self.pixels[r*self.columns:(r+1)*self.columns]) for r in range(self.rows)), encoding="utf-8")

def render_canvas(canvas: DrawingCanvas) -> str:
    return "DRAW A DIGIT\n" + render_digit(canvas.pixels, rows=canvas.rows, columns=canvas.columns) + "\n[↑↓←→] Draw  [C] Clear  [P] Predict  [S] Save  [Q] Quit"

def run_canvas(network) -> None:
    from .app import _read_key
    canvas=DrawingCanvas(); status="Draw, then press P."
    while True:
        sys.stdout.write("\033[2J\033[H"+render_canvas(canvas)+"\n"+status); sys.stdout.flush(); key=_read_key().lower()
        if key in {"q","esc"}: return
        if key in {"left","right","up","down"}: canvas.move(key)
        elif key=="c": canvas.clear()
        elif key=="p": status=prediction_bars(network.predict(canvas.pixels))
        elif key=="s": canvas.save("digit.txt"); status="Saved digit.txt"
