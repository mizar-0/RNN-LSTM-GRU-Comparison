"""Evaluate form, rhyme, repetition, diversity, and copying in the generation sweep."""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np
import pronouncing

import train_simple_rnn as training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, default=Path("results/sweep/sweep.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/sweep"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def lexical(token: str) -> bool:
    return token not in {"<bos>", "<eos>", "<eol>", "<pad>", "<unk>", ",", ".", ";", ":", "!", "?", "-"}


def lines(tokens: list[str]) -> list[list[str]]:
    result, current = [], []
    for token in tokens:
        if token in {"<bos>", "<eos>"}:
            continue
        if token == "<eol>":
            if current:
                result.append(current); current = []
        else:
            current.append(token)
    if current:
        result.append(current)
    return result


def last_word(line: list[str]) -> str | None:
    words = [token for token in line if lexical(token)]
    return words[-1] if words else None


def rhymes(first: str, second: str) -> bool:
    return first == second or second in pronouncing.rhymes(first)


def rhyme_metrics(poem_lines: list[list[str]]) -> tuple[float, float]:
    if len(poem_lines) != 5:
        return float("nan"), 0.0
    endings = [last_word(line) for line in poem_lines]
    coverage = sum(word is not None and bool(pronouncing.phones_for_word(word)) for word in endings) / 5
    if coverage < 1:
        return float("nan"), coverage
    a, a2, b, b2, a3 = endings
    score = sum((rhymes(a, a2), rhymes(b, b2), rhymes(a, a3), rhymes(a2, a3))) / 4
    return float(score), coverage


def ngrams(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    words = [token for token in tokens if lexical(token)]
    return [tuple(words[index:index + size]) for index in range(len(words) - size + 1)]


def normalise_line(line: list[str]) -> str:
    return " ".join(token for token in line if lexical(token))


def bootstrap(values: list[float], samples: int, rng: np.random.Generator) -> tuple[float, float]:
    array = np.asarray([value for value in values if not np.isnan(value)], dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan")
    means = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).tolist())


def score_record(record: dict, training_lines: set[str]) -> dict:
    tokens = record["tokens"]
    poem_lines = lines(tokens)
    rhyme_score, rhyme_coverage = rhyme_metrics(poem_lines)
    words = [token for token in tokens if lexical(token)]
    tri = ngrams(tokens, 3)
    repeated_tri = sum(count - 1 for count in collections.Counter(tri).values() if count > 1) / len(tri) if tri else 0.0
    immediate = sum(words[index] == words[index - 1] for index in range(1, len(words))) / max(1, len(words) - 1)
    poem_line_text = [normalise_line(line) for line in poem_lines if len([token for token in line if lexical(token)]) >= 3]
    copied = sum(line in training_lines for line in poem_line_text) / len(poem_line_text) if poem_line_text else 0.0
    return {**record, "terminated": float(bool(tokens and tokens[-1] == "<eos>")), "five_lines": float(len(poem_lines) == 5), "rhyme_score": rhyme_score, "rhyme_coverage": rhyme_coverage, "repeated_trigram_rate": repeated_tri, "immediate_repeat_rate": immediate, "copied_line_rate": copied, "word_count": len(words), "normalised_poem": " / ".join(normalise_line(line) for line in poem_lines)}


def aggregate(records: list[dict], samples: int, rng: np.random.Generator) -> dict:
    all_tokens = [token for record in records for token in record["tokens"]]
    row = {"n": len(records)}
    for name in ("terminated", "five_lines", "rhyme_score", "rhyme_coverage", "repeated_trigram_rate", "immediate_repeat_rate", "copied_line_rate", "word_count"):
        values = [record[name] for record in records]
        valid = [value for value in values if not np.isnan(value)]
        row[name] = float(np.mean(valid)) if valid else float("nan")
        row[f"{name}_ci_low"], row[f"{name}_ci_high"] = bootstrap(values, samples, rng)
    row["distinct_1"] = len(set(ngrams(all_tokens, 1))) / len(ngrams(all_tokens, 1))
    row["distinct_2"] = len(set(ngrams(all_tokens, 2))) / len(ngrams(all_tokens, 2))
    row["distinct_3"] = len(set(ngrams(all_tokens, 3))) / len(ngrams(all_tokens, 3))
    row["unique_poem_share"] = len(set(record["normalised_poem"] for record in records)) / len(records)
    return row


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    poems, _ = training.load_poems(args.data, sample_size=0, seed=42)
    train_poems, _, test_poems = training.split_poems(poems)
    training_lines = {normalise_line(line) for poem in train_poems for line in lines(poem) if len([token for token in line if lexical(token)]) >= 3}
    records = [json.loads(line) for line in args.sweep.read_text(encoding="utf-8").splitlines()]
    scored = [score_record(record, training_lines) for record in records]
    by_cell: dict[tuple[str, float], list[dict]] = collections.defaultdict(list)
    for record in scored:
        by_cell[(record["model"], record["temperature"])].append(record)
    rng = np.random.default_rng(2026)
    summary = []
    for (model, temperature), cell in sorted(by_cell.items()):
        summary.append({"model": model, "temperature": temperature, **aggregate(cell, args.bootstrap_samples, rng)})
    human_records = [score_record({"tokens": poem}, training_lines) for poem in test_poems[:300]]
    summary.append({"model": "human_reference", "temperature": "", **aggregate(human_records, args.bootstrap_samples, rng)})
    fields = list(summary[0])
    with (args.output / "sweep_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    with (args.output / "per_poem_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for record in scored:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output / "evaluation_notes.md").write_text("# Sweep evaluation\n\nRhyme score is the mean of the A-A, B-B, A-final, and second-A-final rhyme checks, evaluated only for exactly five-line poems whose five line-ending words are all covered by CMU Pronouncing Dictionary. Unterminated poems are retained and count as failures for termination and five-line form. Bootstrap intervals are 95% percentile intervals with 2,000 resamples.\n", encoding="utf-8")
    print(f"Scored {len(scored)} generated poems across {len(by_cell)} model-temperature cells.")


if __name__ == "__main__":
    main()
