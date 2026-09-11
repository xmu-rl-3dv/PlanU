#!/usr/bin/env python

"""Compatibility entry point for the repository-level package definition."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parent.parent
runpy.run_path(str(ROOT / "setup.py"), run_name="__main__")
