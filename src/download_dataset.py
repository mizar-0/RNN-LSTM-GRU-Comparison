"""Download the official Zenodo dataset atomically."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


URL = "https://zenodo.org/api/records/5722527/files/limerick_dataset_oedilf_v3.json/content"
EXPECTED_MD5 = "c086272b5164f4e2f2df9a9f870dfd57"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/limerick_dataset_oedilf_v3.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    request = urllib.request.Request(URL, headers={"User-Agent": "FGAI-coursework/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as file:
        shutil.copyfileobj(response, file)
    checksum = hashlib.md5(temporary.read_bytes()).hexdigest()
    if checksum != EXPECTED_MD5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch: expected {EXPECTED_MD5}, received {checksum}")
    temporary.replace(args.output)
    print(f"Saved verified dataset to {args.output}")


if __name__ == "__main__":
    main()
