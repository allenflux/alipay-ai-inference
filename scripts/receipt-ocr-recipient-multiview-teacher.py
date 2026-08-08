#!/usr/bin/env python
"""Export train-only recipient multiview teacher records from a checkout."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transfer_receipt_ai.recipient_multiview_teacher_export import main


if __name__ == "__main__":
    main()
