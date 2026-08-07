"""Print a strict-JSON form of a training or evaluation summary.

Python's default ``json.dumps`` permits ``NaN`` and ``Infinity`` tokens, while
Windows PowerShell's ``ConvertFrom-Json`` correctly rejects those non-standard
tokens.  This small, dependency-free bridge accepts historic summaries and
turns every non-finite float into JSON ``null`` for the guarded 4090 runner.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _normalize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    # Windows PowerShell 5.1's ``Set-Content -Encoding UTF8`` emits a BOM,
    # while Python-generated evidence is plain UTF-8.  ``utf-8-sig`` accepts
    # both without weakening JSON parsing.
    with args.summary.open("r", encoding="utf-8-sig") as stream:
        payload = json.load(stream, parse_constant=lambda _value: None)
    print(json.dumps(_normalize(payload), ensure_ascii=True, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
