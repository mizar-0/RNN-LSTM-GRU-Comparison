"""Create an 80-poem blinded manual-scoring pack and a separate answer key."""
from __future__ import annotations
import csv
import json
import random
from pathlib import Path

TEMPERATURES = (0.3, 0.5, 0.8, 1.3)
MODELS = ("rnn", "birnn", "lstm", "gru")

def render(tokens: list[str]) -> str:
    lines, current = [], []
    for token in tokens:
        if token in {"<bos>", "<eos>"}: continue
        if token == "<eol>":
            lines.append(" ".join(current)); current=[]
        else: current.append(token)
    if current: lines.append(" ".join(current))
    return "\n".join(lines)

def main() -> None:
    root=Path('results/sweep'); records=[json.loads(line) for line in (root/'sweep.jsonl').read_text(encoding='utf-8').splitlines()]
    rng=random.Random(2026); chosen=[]
    for model in MODELS:
        for temperature in TEMPERATURES:
            pool=[r for r in records if r['model']==model and r['temperature']==temperature]
            chosen.extend(rng.sample(pool,5))
    rng.shuffle(chosen)
    sheet=[]; key=[]
    for index, record in enumerate(chosen,1):
        blind_id=f"P{index:03d}"
        sheet.append({'blind_id':blind_id,'poem':render(record['tokens']),'coherence_1_to_5':'','contextual_consistency_1_to_5':'','comments':''})
        key.append({'blind_id':blind_id,'model':record['model'],'temperature':record['temperature'],'sample_index':record['sample_index'],'prompt':' '.join(record['prompt'])})
    with (root/'blinded_manual_scoring.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=sheet[0]); writer.writeheader(); writer.writerows(sheet)
    with (root/'blinded_manual_scoring_key.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=key[0]); writer.writeheader(); writer.writerows(key)
    (root/'manual_scoring_rubric.md').write_text('# Blinded scoring rubric\n\nScore each dimension from 1 (very poor) to 5 (excellent).\n\n- **Coherence:** grammatical, readable progression with an intelligible overall meaning.\n- **Contextual consistency:** remains related to the supplied prompt and does not introduce contradictory characters, places, or events.\n\nScore poems without seeing model or temperature. Do not open `blinded_manual_scoring_key.csv` until scoring is complete. Generated text may contain crude user-submitted-limerick-style language.\n',encoding='utf-8')
    print('Created blinded 80-poem scoring pack.')
if __name__=='__main__': main()
