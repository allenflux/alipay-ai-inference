#!/usr/bin/env python3
"""Run the read-only recipient held-out error-slice diagnostic from a checkout.

The hyphenated filename is deliberate: it is convenient to type in the RDP
PowerShell session without depending on a package console-script installation.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from transfer_receipt_ai.recipient_slice_report import main


if __name__ == "__main__":
    main()
