"""Generate a reproducible temperature sweep from the final context-64 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import train_simple_rnn as training


RUNS = {"rnn": "final_context64_rnn", "birnn": "final_context64_birnn", "lstm": "final_context64_lstm", "gru": "final_context64_gru"}
TEMPERATURES = (0.3, 0.5, 0.8, 1.0, 1.3, 1.5)
PROMPTS = (("there", "once", "was", "a"), ("a", "young", "man", "from"), ("an", "old", "woman", "who"), ("a", "clever", "young", "poet"), ("in", "a", "small", "town"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--samples-per-cell", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


@torch.no_grad()
def generate_batch(model: torch.nn.Module, sequences: list[list[str]], stoi: dict[str, int], vocabulary: list[str], temperature: float, max_new_tokens: int, device: torch.device) -> list[list[str]]:
    context = 64
    ids = [[stoi["<bos>"]] + [stoi.get(token, stoi["<unk>"]) for token in prompt] for prompt in sequences]
    outputs = [list(sequence) for sequence in ids]
    ended = [False] * len(ids)
    prohibited = [stoi["<pad>"], stoi["<unk>"], stoi["<bos>"]]
    model.eval()
    for _ in range(max_new_tokens):
        inputs = []
        for sequence in outputs:
            inputs.append(([stoi["<pad>"]] * context + sequence)[-context:])
        logits = model(torch.tensor(inputs, dtype=torch.long, device=device))
        logits[:, prohibited] = -float("inf")
        sampled = torch.multinomial(torch.softmax(logits / temperature, dim=1), 1).squeeze(1).tolist()
        for index, token_id in enumerate(sampled):
            if not ended[index]:
                outputs[index].append(token_id)
                ended[index] = token_id == stoi["<eos>"]
        if all(ended):
            break
    return [[vocabulary[token] for token in output] for output in outputs]


def main() -> None:
    args = parse_args()
    if args.samples_per_cell % len(PROMPTS):
        raise ValueError("samples-per-cell must divide evenly across prompts.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    poems, _ = training.load_poems(args.data, sample_size=0, seed=42)
    train_poems, _, _ = training.split_poems(poems)
    stoi, vocabulary = training.make_vocabulary(train_poems, 10_000, 3)
    output_dir = args.results / "sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "sweep.jsonl"
    configuration = {"temperatures": TEMPERATURES, "prompts": [" ".join(prompt) for prompt in PROMPTS], "samples_per_model_temperature": args.samples_per_cell, "max_new_tokens": args.max_new_tokens, "generation_seed": args.seed, "context_window": 64, "models": RUNS}
    (output_dir / "sweep_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    with output.open("w", encoding="utf-8") as handle:
        for model_index, (kind, folder) in enumerate(RUNS.items()):
            checkpoint = args.results / folder / "checkpoints" / f"{kind}_best.pt"
            model = training.RNNLanguageModel(kind, len(vocabulary), 128, 256, 0.3).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            for temperature_index, temperature in enumerate(TEMPERATURES):
                prompts = [list(PROMPTS[index % len(PROMPTS)]) for index in range(args.samples_per_cell)]
                torch.manual_seed(args.seed + model_index * 100 + temperature_index)
                generated = generate_batch(model, prompts, stoi, vocabulary, temperature, args.max_new_tokens, device)
                for sample_index, (prompt, tokens) in enumerate(zip(prompts, generated)):
                    handle.write(json.dumps({"model": kind, "temperature": temperature, "sample_index": sample_index, "prompt": prompt, "tokens": tokens}, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"completed {kind} at temperature {temperature}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
