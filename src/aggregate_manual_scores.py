"""Unblind completed manual scores and aggregate them by model and temperature."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def ci(values: list[float], rng: np.random.Generator) -> tuple[float, float]:
    a=np.asarray(values,float); means=rng.choice(a,size=(10000,len(a)),replace=True).mean(axis=1)
    return tuple(np.percentile(means,[2.5,97.5]))

def main() -> None:
    root=Path('results/sweep')
    scores={row[0]:row for row in json.loads(Path('tmp/manual_scores.json').read_text(encoding='utf-8'))}
    key={row['blind_id']:row for row in csv.DictReader((root/'blinded_manual_scoring_key.csv').open(encoding='utf-8'))}
    records=[]
    for blind_id,row in scores.items():
        if blind_id not in key: raise ValueError(f'Missing key for {blind_id}')
        coherence, consistency=float(row[2]),float(row[3])
        records.append({**key[blind_id], 'coherence':coherence, 'contextual_consistency':consistency, 'manual_mean':(coherence+consistency)/2})
    with (root/'manual_scores_unblinded.csv').open('w',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=records[0]); writer.writeheader(); writer.writerows(records)
    grouped=defaultdict(list)
    for r in records: grouped[(r['model'],float(r['temperature']))].append(r)
    rng=np.random.default_rng(2026); summary=[]
    for (model,temperature), values in sorted(grouped.items()):
        row={'model':model,'temperature':temperature,'n':len(values)}
        for metric in ('coherence','contextual_consistency','manual_mean'):
            v=[item[metric] for item in values]; low,high=ci(v,rng)
            row.update({metric:np.mean(v),f'{metric}_ci_low':low,f'{metric}_ci_high':high})
        summary.append(row)
    fields=list(summary[0])
    with (root/'manual_score_summary.csv').open('w',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    best=max(summary,key=lambda r:r['manual_mean'])
    print(json.dumps({'scored_poems':len(records),'cells':len(summary),'best_manual_cell':best},indent=2))
if __name__=='__main__': main()
