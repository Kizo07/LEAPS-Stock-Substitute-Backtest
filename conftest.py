"""Make the repo importable so `python -m pytest` works from anywhere."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
