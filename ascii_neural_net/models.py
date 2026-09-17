"""Versioned save/load support for ASCII Neural Net models."""
from __future__ import annotations
import json
from pathlib import Path
from .network import NeuralNetwork

def save_model(network: NeuralNetwork, path: str | Path, *, metadata: dict[str,str] | None = None) -> None:
 data={"format":"ASCII-NN MODEL","version":1,"layers":[network.layers[0].input_size,*[layer.output_size for layer in network.layers]],"activations":[layer.activation for layer in network.layers],"weights":[layer.weights for layer in network.layers],"biases":[layer.biases for layer in network.layers],"metadata":metadata or {}}
 Path(path).write_text(json.dumps(data,indent=2),encoding="utf-8")

def load_model(path: str | Path) -> tuple[NeuralNetwork,dict[str,str]]:
 data=json.loads(Path(path).read_text(encoding="utf-8"))
 if data.get("format")!="ASCII-NN MODEL" or data.get("version")!=1: raise ValueError("Unsupported ASCII-NN model file")
 network=NeuralNetwork(data["layers"],data["activations"],seed=0)
 for layer,weights,biases in zip(network.layers,data["weights"],data["biases"]): layer.weights,layer.biases=weights,biases
 return network,data.get("metadata",{})
