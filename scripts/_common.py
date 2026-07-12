#!/usr/bin/env python3
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(stage: str):
    os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-m", "flowpro.workflow", stage, *sys.argv[1:]], cwd=ROOT, check=True)

