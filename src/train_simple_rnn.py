"""Train and evaluate a word-level Simple RNN limerick language model.

The script deliberately splits by poem before making next-token windows, so a
single poem cannot contribute windows to more than one split.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


TOKEN_RE = re.compile(r"[a-z0-9']+|[,.;:!?\-]")
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eol>", "<eos>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/simple_rnn"))
    parser.add_argument("--model-kind", choices=("rnn", "birnn", "lstm", "gru"), default="rnn")
    parser.add_argument("--sample-size", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context", type=int, default=20)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--min-frequency", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(poem: str) -> list[str]:
    tokens = ["<bos>"]
    for line in poem.strip().splitlines():
        line_tokens = TOKEN_RE.findall(line.lower().strip())
        if line_tokens:
            tokens.extend(line_tokens)
            tokens.append("<eol>")
    return tokens + ["<eos>"]


def load_poems(path: Path, sample_size: int, seed: int) -> tuple[list[list[str]], dict[str, int]]:
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a non-empty JSON list of dataset records.")
    required = {"limerick", "is_limerick", "author", "id"}
    missing = required - set(records[0])
    if missing:
        raise ValueError(f"Dataset schema is missing fields: {sorted(missing)}")

    selected = [record for record in records if record.get("is_limerick") and record.get("limerick")]
    random.Random(seed).shuffle(selected)
    if sample_size:
        selected = selected[:sample_size]
    poems = [tokenize(record["limerick"]) for record in selected]
    if len(poems) < 100:
        raise ValueError("Too few filtered poems to train a model.")
    metadata = {
        "all_records": len(records),
        "filtered_limericks": sum(bool(record.get("is_limerick")) for record in records),
        "selected_poems": len(poems),
        "authors_in_selected_set": len({record.get("author") for record in selected}),
    }
    return poems, metadata


def split_poems(poems: list[list[str]]) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    n = len(poems)
    train_end, validation_end = int(0.8 * n), int(0.9 * n)
    return poems[:train_end], poems[train_end:validation_end], poems[validation_end:]


def make_vocabulary(train_poems: list[list[str]], vocab_size: int, min_frequency: int) -> tuple[dict[str, int], list[str]]:
    counts = collections.Counter(token for poem in train_poems for token in poem)
    vocabulary = list(SPECIAL_TOKENS)
    vocabulary.extend(
        token
        for token, frequency in counts.most_common()
        if frequency >= min_frequency and token not in vocabulary
    )
    vocabulary = vocabulary[:vocab_size]
    return {token: index for index, token in enumerate(vocabulary)}, vocabulary


def make_examples(poems: list[list[str]], stoi: dict[str, int], context: int) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id, unknown_id = stoi["<pad>"], stoi["<unk>"]
    inputs: list[list[int]] = []
    targets: list[int] = []
    for poem in poems:
        ids = [pad_id] * context + [stoi.get(token, unknown_id) for token in poem]
        for index in range(context, len(ids)):
            inputs.append(ids[index - context:index])
            targets.append(ids[index])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class RNNLanguageModel(nn.Module):
    def __init__(self, model_kind: str, vocabulary_size: int, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.model_kind = model_kind
        self.hidden_dim = hidden_dim
        self.bidirectional = model_kind == "birnn"
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        cell = {"rnn": nn.RNN, "birnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[model_kind]
        self.rnn = cell(embedding_dim, hidden_dim, batch_first=True, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim * (2 if self.bidirectional else 1), vocabulary_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(inputs))
        hidden_states, final_states = self.rnn(embedded)
        if self.bidirectional:
            # Both states summarize only the supplied, pre-target context window.
            features = torch.cat((final_states[-2], final_states[-1]), dim=1)
        else:
            features = hidden_states[:, -1]
        return self.output(self.dropout(features))


@torch.no_grad()
def evaluate(model: nn.Module, features: torch.Tensor, targets: torch.Tensor, device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    for start in range(0, len(features), batch_size):
        inputs = features[start:start + batch_size].to(device, non_blocking=True)
        labels = targets[start:start + batch_size].to(device, non_blocking=True)
        logits = model(inputs)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        top5 = logits.topk(k=5, dim=1).indices
        correct_top1 += (top5[:, 0] == labels).sum().item()
        correct_top5 += (top5 == labels[:, None]).any(dim=1).sum().item()
    loss = total_loss / len(features)
    return {"loss": loss, "perplexity": math.exp(loss), "top1_accuracy": correct_top1 / len(features), "top5_accuracy": correct_top5 / len(features)}


@torch.no_grad()
def generate(model: nn.Module, stoi: dict[str, int], itos: list[str], context: int, device: torch.device, seed_tokens: list[str], temperature: float = 0.8, max_tokens: int = 60) -> list[str]:
    model.eval()
    ids = [stoi.get(token, stoi["<unk>"]) for token in seed_tokens]
    output = list(seed_tokens)
    for _ in range(max_tokens):
        current_context = ([stoi["<pad>"]] * context + ids)[-context:]
        logits = model(torch.tensor([current_context], dtype=torch.long, device=device))[0]
        logits[stoi["<pad>"]] = -float("inf")
        logits[stoi["<unk>"]] = -float("inf")
        next_id = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).item()
        token = itos[next_id]
        ids.append(next_id)
        output.append(token)
        if token == "<eos>":
            break
    return output


def render_tokens(tokens: list[str]) -> str:
    lines: list[str] = []
    current: list[str] = []
    for token in tokens:
        if token in {"<bos>", "<eos>"}:
            continue
        if token == "<eol>":
            lines.append(" ".join(current).replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?"))
            current = []
        else:
            current.append(token)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    poems, dataset_metadata = load_poems(args.data, args.sample_size, args.seed)
    train_poems, validation_poems, test_poems = split_poems(poems)
    stoi, itos = make_vocabulary(train_poems, args.vocab_size, args.min_frequency)
    x_train, y_train = make_examples(train_poems, stoi, args.context)
    x_validation, y_validation = make_examples(validation_poems, stoi, args.context)
    x_test, y_test = make_examples(test_poems, stoi, args.context)

    model = RNNLanguageModel(args.model_kind, len(itos), args.embedding_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    checkpoint = checkpoint_dir / f"{args.model_kind}_best.pt"
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for inputs, labels in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * len(labels)
        validation = evaluate(model, x_validation, y_validation, device, args.batch_size * 4)
        row = {"epoch": epoch, "train_loss": total_train_loss / len(x_train), "validation_loss": validation["loss"], "validation_perplexity": validation["perplexity"]}
        history.append(row)
        print(json.dumps(row))
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_metrics = evaluate(model, x_test, y_test, device, args.batch_size * 4)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "model": {"rnn": "SimpleRNN", "birnn": "BidirectionalRNN", "lstm": "LSTM", "gru": "GRU"}[args.model_kind],
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "seed": args.seed,
        "context_window": args.context,
        "vocabulary_size": len(itos),
        "parameter_count": parameter_count,
        "epochs_completed": len(history),
        "training_seconds": training_seconds,
        "seconds_per_epoch": training_seconds / len(history),
        "dataset": dataset_metadata,
        "examples": {"train": len(x_train), "validation": len(x_validation), "test": len(x_test)},
        "test": test_metrics,
    }
    with (args.output / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with (args.output / "vocabulary.json").open("w", encoding="utf-8") as file:
        json.dump(itos, file, ensure_ascii=False)
    sample_seeds = [["<bos>", "there", "once", "was", "a"], ["<bos>", "a", "young", "man", "from"], ["<bos>", "an", "old", "woman", "who"]]
    with (args.output / "generated_samples.md").open("w", encoding="utf-8") as file:
        file.write(f"# {metadata['model']} samples\n\n")
        for temperature in (0.5, 0.8, 1.0, 1.3):
            file.write(f"## Temperature {temperature}\n\n")
            for seed_tokens in sample_seeds:
                torch.manual_seed(args.seed)
                file.write(f"Seed: `{' '.join(seed_tokens)}`\n\n")
                file.write(render_tokens(generate(model, stoi, itos, args.context, device, seed_tokens, temperature)) + "\n\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
