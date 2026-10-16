"""Recompute final test metrics from saved context-64 checkpoints.

This script performs no training and writes no model artefacts.  It verifies
that each saved best-validation checkpoint reproduces the metrics recorded by
the final run, using the same deterministic data split and vocabulary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import train_simple_rnn as training


RUNS = {
    "rnn": "final_context64_rnn",
    "birnn": "final_context64_birnn",
    "lstm": "final_context64_lstm",
    "gru": "final_context64_gru",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def nearly_equal(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    poems, _ = training.load_poems(args.data, sample_size=0, seed=42)
    train_poems, _, test_poems = training.split_poems(poems)
    stoi, vocabulary = training.make_vocabulary(train_poems, vocab_size=10_000, min_frequency=3)
    x_test, y_test = training.make_examples(test_poems, stoi, context=64)

    results: list[dict[str, object]] = []
    for kind, folder in RUNS.items():
        run_dir = args.results / folder
        saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        checkpoint = run_dir / "checkpoints" / f"{kind}_best.pt"
        model = training.RNNLanguageModel(kind, len(vocabulary), 128, 256, 0.3).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        observed = training.evaluate(model, x_test, y_test, device, args.batch_size)
        expected = saved["test"]
        matches = all(nearly_equal(observed[name], expected[name]) for name in expected)
        results.append(
            {
                "model": saved["model"],
                "checkpoint": str(checkpoint),
                "matches_metrics_json": matches,
                "expected": expected,
                "observed": observed,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(json.dumps({"device": str(device), "all_match": all(row["matches_metrics_json"] for row in results), "runs": results}, indent=2))


if __name__ == "__main__":
    main()
