"""Pytest bootstrap: ensure the repo root is importable so tests can `import trader_core`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
