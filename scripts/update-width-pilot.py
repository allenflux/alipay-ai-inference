"""Synchronize the one guarded-pilot fix on a 4090 host without Git.

The remote development machine used for this pilot does not expose ``git`` on
its PowerShell PATH.  This intentionally small bootstrapper downloads the
already-reviewed main-branch implementation, validates its identifying guard,
and retains a timestamped local backup before replacement.  It is only for
the transient width-1536 pilot recovery; normal development remains Git-based.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


SOURCE_URL = (
    "https://raw.githubusercontent.com/irenixf/Hx.AI.py/main/"
    "src/transfer_receipt_ai/ocr_unified.py"
)
TARGET = Path("src") / "transfer_receipt_ai" / "ocr_unified.py"
IDENTIFYING_GUARD = b"strictly-wider exception"


def main() -> None:
    if not TARGET.is_file():
        raise FileNotFoundError(f"Pilot source target is missing: {TARGET}")
    with urlopen(SOURCE_URL, timeout=30) as response:
        downloaded = response.read()
    if IDENTIFYING_GUARD not in downloaded:
        raise RuntimeError("Downloaded source is missing the width-expansion guard; refusing replacement")
    backup_suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = TARGET.with_name(f"{TARGET.name}.before-width-fix-{backup_suffix}")
    backup.write_bytes(TARGET.read_bytes())
    TARGET.write_bytes(downloaded)
    print(f"updated={TARGET}")
    print(f"backup={backup}")


if __name__ == "__main__":
    main()
