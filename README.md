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

## Run the Phase 3 visualization

```powershell
py -3.12 -m ascii_neural_net.visual_demo 0 1
```

The renderer displays live activations for each input, hidden, and output node. Green edges have positive weights; red edges have negative weights; thicker edges have larger magnitudes.

## Phase 4: network builder

Choose `Create Network` from the main menu to open the builder. Use the arrow keys to select a layer and change its neuron count, add or delete hidden layers, cycle activations, choose the learning rate and initializer, randomize weights, then press `T` to train an XOR-compatible network.

## Phase 5: datasets

Choose `Load Dataset` to explore XOR, AND, OR, linear, circle, and spiral datasets as ASCII scatter plots. Load a CSV, map its label column, create a train/test split, normalize its features, and inspect it with:

```powershell
py -3.12 -m ascii_neural_net.dataset_demo path\to\data.csv
```

## Test

```powershell
py -3.12 -m unittest discover -s tests -v
```
