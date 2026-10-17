"""Local (git-ignored) speaker names and chat path for the archival experiment scripts."""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_p = _ROOT / "persona" / "local_speakers.json"
_d = json.load(open(_p, encoding="utf-8")) if _p.exists() else {}
S = _d.get("partner", "S")
T = _d.get("target", "T")
CHAT = Path(_d.get("chat_path", str(_ROOT / "chathistory.txt")))
ROOT = _ROOT
