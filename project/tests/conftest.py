"""
Pytest configuration: mock heavy ML dependencies so the test suite
runs without GPU libraries installed.
"""
import sys
import os
from unittest.mock import MagicMock, patch

# ---- Add project root to path BEFORE any src import ----
# This fixes "ModuleNotFoundError: No module named 'src'"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- Stub out heavy ML libs before any src import ----
_torch = MagicMock()
_torch.cuda.is_available.return_value = False
_torch.device.return_value = "cpu"
sys.modules.setdefault("torch", _torch)
sys.modules.setdefault("torch.nn", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())

for mod in [
    "whisper", "pyannote", "pyannote.audio",
    "resemblyzer", "resemblyzer.hparams",
    "noisereduce", "librosa", "soundfile",
    "numpy", "sklearn", "sklearn.cluster",
    "resemblyzer.audio",
]:
    sys.modules.setdefault(mod, MagicMock())