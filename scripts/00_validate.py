#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowpro.config import load_config
from flowpro.validate import validate_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FlowPRO assets and dependencies")
    parser.add_argument("--config", default="configs/flowpro.json")
    parser.add_argument("--hardware", action="store_true")
    args = parser.parse_args()
    checks = validate_project(load_config(args.config), require_hardware=args.hardware)
    for check in checks:
        print(f"{'OK' if check.ok else 'MISSING':7} {check.name:28} {check.detail}")
    if not all(check.ok for check in checks):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
