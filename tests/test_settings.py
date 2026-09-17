import tempfile
import unittest
from pathlib import Path
from ascii_neural_net.settings import Settings, load_settings, save_settings
class SettingsTests(unittest.TestCase):
 def test_settings_round_trip(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/"settings.json"; save_settings(Settings(theme="amber",unicode=False),path); settings=load_settings(path)
  self.assertEqual(settings.theme,"amber"); self.assertFalse(settings.unicode)
