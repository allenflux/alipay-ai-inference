#!/usr/bin/env python
"""Seal or verify a full-crop pilot source for the guarded v14 route."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transfer_receipt_ai.recipient_full_crop_candidate_source import main


if __name__ == "__main__":
    main()
