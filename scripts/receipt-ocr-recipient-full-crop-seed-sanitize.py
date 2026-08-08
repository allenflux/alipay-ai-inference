#!/usr/bin/env python
"""Publish an attested analysis-only v13 full-crop warmstart seed."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transfer_receipt_ai.recipient_full_crop_seed_sanitizer import main


if __name__ == "__main__":
    main()
