"""Persistent terminal preferences."""
from __future__ import annotations
import json
from dataclasses import asdict,dataclass
from pathlib import Path
@dataclass
class Settings: theme:str="cyan"; unicode:bool=True
def load_settings(path: str|Path=".ascii_nn_settings.json")->Settings:
 p=Path(path)
 if not p.exists(): return Settings()
 try: return Settings(**json.loads(p.read_text(encoding="utf-8")))
 except (json.JSONDecodeError,TypeError): return Settings()
def save_settings(settings:Settings,path: str|Path=".ascii_nn_settings.json")->None: Path(path).write_text(json.dumps(asdict(settings),indent=2),encoding="utf-8")
