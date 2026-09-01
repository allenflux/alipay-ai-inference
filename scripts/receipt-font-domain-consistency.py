#!/usr/bin/env python3
"""Source-tree wrapper for the font-domain consistency sidecar CLI."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from transfer_receipt_ai.font_domain_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
