# ASCII Neural Net

An interactive terminal application for building and visualizing neural networks.

## Run the terminal app

```powershell
py -3.12 -m ascii_neural_net
```

Use the arrow keys or `1`–`4` to select a menu item. Press `Q` or `Esc` to quit.

## Run the Phase 2 XOR demo

```powershell
py -3.12 -m ascii_neural_net.xor_demo
```

This trains a `2 → 4 → 1` network from scratch on XOR and prints each learned prediction.

## Test

```powershell
py -3.12 -m unittest discover -s tests -v
```
