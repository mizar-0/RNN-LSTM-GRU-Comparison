"""Run the final, context-64 comparison serially on one GPU."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "src" / "train_simple_rnn.py"
DATA = ROOT / "data" / "limerick_dataset_oedilf_v3.json"
MODELS = ("rnn", "birnn", "lstm", "gru")


def main() -> None:
    for model in MODELS:
        output = ROOT / "results" / f"final_context64_{model}"
        command = [
            sys.executable, str(TRAINER),
            "--model-kind", model,
            "--data", str(DATA),
            "--output", str(output),
            "--sample-size", "0",
            "--context", "64",
            "--epochs", "40",
            "--patience", "5",
            "--batch-size", "512",
            "--seed", "42",
        ]
        print(f"Starting final context-64 {model} run.", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        print(f"Completed final context-64 {model} run.", flush=True)


if __name__ == "__main__":
    main()
