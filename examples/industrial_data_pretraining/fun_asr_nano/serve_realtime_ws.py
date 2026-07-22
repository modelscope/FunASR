#!/usr/bin/env python3
"""Compatibility entry point for the packaged realtime WebSocket server."""

import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[3]
if (repo_root / "funasr").is_dir() and str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from funasr.bin.realtime_ws import cli_main


if __name__ == "__main__":
    cli_main()
