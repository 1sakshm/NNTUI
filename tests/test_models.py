import tempfile
import unittest
from pathlib import Path
from ascii_neural_net.models import load_model, save_model
from ascii_neural_net.network import NeuralNetwork, xor_dataset

class ModelPersistenceTests(unittest.TestCase):
 def test_saved_model_reloads_with_same_prediction(self):
  x,y=xor_dataset(); network=NeuralNetwork([2,4,1],["tanh","sigmoid"],seed=4); network.fit(x,y,epochs=4000,learning_rate=.8,batch_size=4)
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/"xor.ann"; save_model(network,path,metadata={"dataset":"xor"}); restored,metadata=load_model(path)
  self.assertEqual(metadata["dataset"],"xor")
  self.assertAlmostEqual(network.predict([0,1])[0],restored.predict([0,1])[0],places=9)
