"""Simple weight-based explanations for network predictions."""
from __future__ import annotations
from .network import NeuralNetwork

def explain_prediction(network: NeuralNetwork, inputs: list[float]) -> str:
 prediction=network.predict(inputs); confidence=max(prediction); winner=prediction.index(confidence)
 first=network.layers[0]
 saliency=[sum(abs(first.weights[n][i]) for n in range(first.output_size))*abs(value) for i,value in enumerate(inputs)]
 hidden=sorted(enumerate(first.outputs,1),key=lambda item:abs(item[1]),reverse=True)[:3]
 lines=[f"PREDICTION: {winner}",f"Confidence: {confidence:.1%}","","INPUT SALIENCY"]
 lines += [f"Input {i+1:>2}  {'█'*max(1,round(value*10))}" for i,value in enumerate(saliency)]
 lines += ["","Strongest neurons:"]
 lines += [f"  Hidden {index:02}  {'█'*max(1,round(abs(value)*10))}  {value:+.3f}" for index,value in hidden]
 return "\n".join(lines)
