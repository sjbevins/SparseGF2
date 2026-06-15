"""Pytest configuration: make the tests/ directory importable.

Lets test modules ``import _stim_parity`` directly instead of going
through a relative import.
"""

import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
