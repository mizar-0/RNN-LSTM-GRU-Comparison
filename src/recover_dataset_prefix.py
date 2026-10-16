"""Recover complete records from a locally interrupted JSON-array download.

This is only for creating an initial, validated training subset while a complete
official download is unavailable. It never overwrites the source file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    decoder = json.JSONDecoder()
    try:
        decoder.raw_decode(raw.decode("utf-8"))
        recovered = raw
    except json.JSONDecodeError as error:
        boundary = raw.rfind(b"}", 0, error.pos)
        if boundary < 0:
            raise RuntimeError("No complete record before the interrupted position.") from error
        recovered = raw[:boundary + 1] + b"]"
    records = json.loads(recovered)
    if not isinstance(records, list) or not records:
        raise RuntimeError("Recovery did not produce a non-empty record list.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(recovered)
    print(f"Recovered {len(records)} complete records to {args.output}")


if __name__ == "__main__":
    main()
