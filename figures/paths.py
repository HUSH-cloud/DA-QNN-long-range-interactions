"""Locations of results, data and output PDFs for the figure scripts, each overridable by an environment variable."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("DAQNN_RESULTS", REPO / "results"))
DATA = Path(os.environ.get("DAQNN_DATA", REPO / "data"))
THERMAL = Path(os.environ.get("DAQNN_THERMAL", RESULTS / "thermalization"))
OUT = Path(os.environ.get("DAQNN_FIGS", REPO / "figures" / "out"))

SEEDS_10 = [42, 123, 456, 789, 1024, 2024, 2025, 2718, 3141, 5678]
SEEDS_20 = SEEDS_10 + [1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 10010]
