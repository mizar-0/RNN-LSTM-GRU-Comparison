"""Create dependency-free SVG figures and a summary table for final context-64 runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


RUNS = (("rnn", "Simple RNN", "#4C78A8"), ("birnn", "BiRNN", "#F58518"), ("lstm", "LSTM", "#54A24B"), ("gru", "GRU", "#B279A2"))
FONT = "font-family='Arial, sans-serif'"


def load_run(results_dir: Path, key: str) -> tuple[dict, list[dict[str, str]]]:
    run_dir = results_dir / f"final_context64_{key}"
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with (run_dir / "history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    if not history:
        raise ValueError(f"No history found for {key}.")
    return metrics, history


def start_svg(width: int, height: int, title: str) -> list[str]:
    return [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img' aria-labelledby='title desc'>", f"<title id='title'>{title}</title>", f"<desc id='desc'>{title} for the final context-64 language-model experiments.</desc>", "<rect width='100%' height='100%' fill='white'/>"]


def finish_svg(parts: list[str], output: Path) -> None:
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def pos(value: float, low: float, high: float, begin: float, end: float) -> float:
    return begin + (value - low) / (high - low) * (end - begin)


def ticks(low: float, high: float) -> list[float]:
    return [low + (high - low) * index / 4 for index in range(5)]


def draw_axes(parts: list[str], x0: float, y0: float, x1: float, y1: float, low: float, high: float, epochs: int) -> None:
    parts.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#333'/><line x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}' stroke='#333'/>")
    for value in ticks(low, high):
        y = pos(value, low, high, y1, y0)
        parts.append(f"<line x1='{x0}' y1='{y:.1f}' x2='{x1}' y2='{y:.1f}' stroke='#d8d8d8'/><text x='{x0-7}' y='{y+4:.1f}' text-anchor='end' font-size='10' {FONT}>{value:.2f}</text>")
    for epoch in sorted({1, max(1, epochs // 2), epochs}):
        x = x0 if epochs == 1 else x0 + (x1 - x0) * (epoch - 1) / (epochs - 1)
        parts.append(f"<text x='{x:.1f}' y='{y1+16}' text-anchor='middle' font-size='10' {FONT}>{epoch}</text>")


def polyline(values: list[float], x0: float, x1: float, y0: float, y1: float, low: float, high: float) -> str:
    return " ".join(f"{(x0 if len(values)==1 else x0+(x1-x0)*index/(len(values)-1)):.1f},{pos(value, low, high, y1, y0):.1f}" for index, value in enumerate(values))


def make_learning_curves(runs: list[dict], output_dir: Path) -> None:
    parts = start_svg(1200, 800, "Training and validation convergence")
    parts.append(f"<text x='600' y='32' text-anchor='middle' font-size='20' font-weight='bold' {FONT}>Training and validation convergence (context window = 64)</text>")
    for index, run in enumerate(runs):
        col, row = index % 2, index // 2
        ox, oy = 55 + col * 575, 65 + row * 355
        x0, x1, y0, y1 = ox + 58, ox + 530, oy + 38, oy + 285
        train = [float(item["train_loss"]) for item in run["history"]]
        validation = [float(item["validation_loss"]) for item in run["history"]]
        low, high = min(train + validation), max(train + validation)
        pad = max(0.02, (high - low) * .10)
        low, high = low - pad, high + pad
        parts.append(f"<text x='{ox+294}' y='{oy+15}' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>{run['label']} ({len(train)} epochs)</text>")
        draw_axes(parts, x0, y0, x1, y1, low, high, len(train))
        parts.append(f"<polyline points='{polyline(train,x0,x1,y0,y1,low,high)}' fill='none' stroke='{run['color']}' stroke-width='2.5'/><polyline points='{polyline(validation,x0,x1,y0,y1,low,high)}' fill='none' stroke='{run['color']}' stroke-width='2.5' stroke-dasharray='7 5'/>")
        best = min(range(len(validation)), key=validation.__getitem__)
        bx = x0 if len(validation) == 1 else x0 + (x1-x0) * best / (len(validation)-1)
        by = pos(validation[best], low, high, y1, y0)
        parts.append(f"<circle cx='{bx:.1f}' cy='{by:.1f}' r='4' fill='black'/><text x='{x0}' y='{y0-9}' font-size='10' {FONT}>Loss</text><text x='{(x0+x1)/2}' y='{y1+34}' text-anchor='middle' font-size='10' {FONT}>Epoch</text>")
        parts.append(f"<line x1='{x1-158}' y1='{y0+12}' x2='{x1-135}' y2='{y0+12}' stroke='{run['color']}' stroke-width='2.5'/><text x='{x1-130}' y='{y0+16}' font-size='10' {FONT}>Training</text><line x1='{x1-80}' y1='{y0+12}' x2='{x1-57}' y2='{y0+12}' stroke='{run['color']}' stroke-width='2.5' stroke-dasharray='7 5'/><text x='{x1-52}' y='{y0+16}' font-size='10' {FONT}>Validation</text>")
    finish_svg(parts, output_dir / "learning_curves.svg")


def draw_bars(parts: list[str], ox: int, title: str, ylabel: str, labels: list[str], values: list[float], colors: list[str], fmt: str) -> None:
    x0, x1, y0, y1 = ox + 58, ox + 355, 95, 329
    maximum = max(values) * 1.18
    parts.append(f"<text x='{(x0+x1)/2}' y='71' text-anchor='middle' font-size='14' font-weight='bold' {FONT}>{title}</text><line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#333'/><line x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}' stroke='#333'/>")
    for value in ticks(0, maximum):
        y = pos(value, 0, maximum, y1, y0)
        parts.append(f"<line x1='{x0}' y1='{y:.1f}' x2='{x1}' y2='{y:.1f}' stroke='#d8d8d8'/><text x='{x0-6}' y='{y+4:.1f}' text-anchor='end' font-size='9' {FONT}>{value:.1f}</text>")
    band = (x1-x0)/len(values)
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        bx, bw = x0+index*band+band*.18, band*.64
        by = pos(value, 0, maximum, y1, y0)
        parts.append(f"<rect x='{bx:.1f}' y='{by:.1f}' width='{bw:.1f}' height='{y1-by:.1f}' fill='{color}'/><text x='{bx+bw/2:.1f}' y='{by-6:.1f}' text-anchor='middle' font-size='10' {FONT}>{format(value,fmt)}</text><text x='{bx+bw/2:.1f}' y='{y1+15}' text-anchor='middle' font-size='9' {FONT}>{label}</text>")
    parts.append(f"<text x='{x0-43}' y='{(y0+y1)/2}' text-anchor='middle' transform='rotate(-90 {x0-43} {(y0+y1)/2})' font-size='10' {FONT}>{ylabel}</text>")


def make_bar_figure(runs: list[dict], output: Path, title: str, panels: list[tuple[str, str, list[float], str]]) -> None:
    parts = start_svg(1250, 410, title)
    parts.append(f"<text x='625' y='30' text-anchor='middle' font-size='20' font-weight='bold' {FONT}>{title}</text>")
    labels, colors = [run["label"] for run in runs], [run["color"] for run in runs]
    for index, panel in enumerate(panels):
        draw_bars(parts, 30+index*405, panel[0], panel[1], labels, panel[2], colors, panel[3])
    finish_svg(parts, output)


def write_summary(runs: list[dict], output: Path) -> None:
    fields = ["model","epochs_completed","best_validation_epoch","best_validation_perplexity","test_loss","test_perplexity","test_top1_accuracy_pct","test_top5_accuracy_pct","parameter_count","seconds_per_epoch","training_minutes"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for run in runs:
            metrics, history = run["metrics"], run["history"]
            best = min(history, key=lambda row: float(row["validation_loss"]))
            writer.writerow({"model":run["label"],"epochs_completed":len(history),"best_validation_epoch":best["epoch"],"best_validation_perplexity":f"{float(best['validation_perplexity']):.4f}","test_loss":f"{metrics['test']['loss']:.6f}","test_perplexity":f"{metrics['test']['perplexity']:.4f}","test_top1_accuracy_pct":f"{100*metrics['test']['top1_accuracy']:.4f}","test_top5_accuracy_pct":f"{100*metrics['test']['top5_accuracy']:.4f}","parameter_count":metrics["parameter_count"],"seconds_per_epoch":f"{metrics['seconds_per_epoch']:.4f}","training_minutes":f"{metrics['training_seconds']/60:.4f}"})


def main() -> None:
    results_dir, output_dir = Path("results"), Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for key, label, color in RUNS:
        metrics, history = load_run(results_dir, key); runs.append({"label":label,"color":color,"metrics":metrics,"history":history})
    make_learning_curves(runs, output_dir)
    make_bar_figure(runs, output_dir / "test_metrics.svg", "Held-out next-word prediction performance", [("Test perplexity (lower is better)","Perplexity",[r["metrics"]["test"]["perplexity"] for r in runs],".2f"),("Test top-1 accuracy","Accuracy (%)",[100*r["metrics"]["test"]["top1_accuracy"] for r in runs],".2f"),("Test top-5 accuracy","Accuracy (%)",[100*r["metrics"]["test"]["top5_accuracy"] for r in runs],".2f")])
    make_bar_figure(runs, output_dir / "compute_cost.svg", "Model capacity and training cost", [("Model size","Parameters (millions)",[r["metrics"]["parameter_count"]/1e6 for r in runs],".2f"),("Mean time per epoch","Seconds",[r["metrics"]["seconds_per_epoch"] for r in runs],".1f"),("Total training time","Minutes",[r["metrics"]["training_seconds"]/60 for r in runs],".1f")])
    write_summary(runs, output_dir / "summary_table.csv")
    print(f"Wrote figures and summary table to {output_dir}")


if __name__ == "__main__":
    main()
