"""Export SVG plots for the evaluated temperature sweep."""
from __future__ import annotations
import csv
from pathlib import Path

MODELS = (("rnn", "Simple RNN", "#4C78A8"), ("birnn", "BiRNN", "#F58518"), ("lstm", "LSTM", "#54A24B"), ("gru", "GRU", "#B279A2"))
FONT = "font-family='Arial, sans-serif'"

def p(v, a, b, x0, x1): return x0 + (v-a)/(b-a)*(x1-x0)
def start(w,h,title): return [f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' role='img'><title>{title}</title><rect width='100%' height='100%' fill='white'/>"]
def end(parts,path): path.write_text("\n".join(parts+["</svg>"]),encoding="utf-8")
def axes(parts,x0,y0,x1,y1,ylabel,title,lo,hi):
    parts.append(f"<text x='{(x0+x1)/2}' y='{y0-18}' text-anchor='middle' font-size='14' font-weight='bold' {FONT}>{title}</text><line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#333'/><line x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}' stroke='#333'/>")
    for i in range(5):
        v=lo+(hi-lo)*i/4; y=p(v,lo,hi,y1,y0)
        parts.append(f"<line x1='{x0}' y1='{y:.1f}' x2='{x1}' y2='{y:.1f}' stroke='#ddd'/><text x='{x0-7}' y='{y+4:.1f}' text-anchor='end' font-size='9' {FONT}>{v:.2f}</text>")
    for t in (0.3,0.8,1.3):
        x=p(t,.3,1.5,x0,x1); parts.append(f"<text x='{x:.1f}' y='{y1+16}' text-anchor='middle' font-size='9' {FONT}>{t:.1f}</text>")
    parts.append(f"<text x='{(x0+x1)/2}' y='{y1+34}' text-anchor='middle' font-size='10' {FONT}>Temperature</text><text x='{x0-41}' y='{(y0+y1)/2}' text-anchor='middle' transform='rotate(-90 {x0-41} {(y0+y1)/2})' font-size='10' {FONT}>{ylabel}</text>")
def path(points): return " ".join(f"{x:.1f},{y:.1f}" for x,y in points)

def main():
    root=Path('results/sweep'); rows=list(csv.DictReader((root/'sweep_metrics.csv').open()))
    cells={(r['model'],float(r['temperature'])):r for r in rows if r['model']!='human_reference'}
    temps=(.3,.5,.8,1.0,1.3,1.5)
    panels=(('five_lines','Five-line form','Share of poems',0,1),('rhyme_score','AABBA rhyme score','Mean score',0,.5),('distinct_2','Distinct-2','Unique bigrams / bigrams',0,1),('repeated_trigram_rate','Repeated trigram rate','Rate',0,.04))
    parts=start(1200,800,'Temperature effects on form and diversity'); parts.append(f"<text x='600' y='32' text-anchor='middle' font-size='20' font-weight='bold' {FONT}>Temperature effects on generated limericks</text>")
    for idx,(metric,title,ylabel,lo,hi) in enumerate(panels):
        col,row=idx%2,idx//2; x0=90+col*590; x1=x0+450; y0=95+row*350; y1=y0+220; axes(parts,x0,y0,x1,y1,ylabel,title,lo,hi)
        for model,label,color in MODELS:
            values=[float(cells[(model,t)][metric]) for t in temps]; pts=[(p(t,.3,1.5,x0,x1),p(v,lo,hi,y1,y0)) for t,v in zip(temps,values)]
            parts.append(f"<polyline points='{path(pts)}' fill='none' stroke='{color}' stroke-width='2.4'/>")
            for x,y in pts: parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.2' fill='{color}'/>")
        if idx==0:
            for n,(_,label,color) in enumerate(MODELS): parts.append(f"<line x1='{x1-270+n*68}' y1='{y0+13}' x2='{x1-252+n*68}' y2='{y0+13}' stroke='{color}' stroke-width='3'/><text x='{x1-248+n*68}' y='{y0+17}' font-size='9' {FONT}>{label}</text>")
    end(parts,root/'temperature_effects.svg')
    parts=start(760,560,'Quality-diversity frontier'); parts.append(f"<text x='380' y='32' text-anchor='middle' font-size='20' font-weight='bold' {FONT}>Quality-diversity frontier</text>"); x0,y0,x1,y1=100,80,700,470
    parts.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#333'/><line x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}' stroke='#333'/>")
    for i in range(6):
        v=i/5; x=p(v,0,1,x0,x1); y=p(v,0,1,y1,y0); parts.append(f"<line x1='{x}' y1='{y0}' x2='{x}' y2='{y1}' stroke='#ddd'/><text x='{x}' y='{y1+17}' text-anchor='middle' font-size='9' {FONT}>{v:.1f}</text><line x1='{x0}' y1='{y}' x2='{x1}' y2='{y}' stroke='#ddd'/><text x='{x0-7}' y='{y+4}' text-anchor='end' font-size='9' {FONT}>{v:.1f}</text>")
    parts.append(f"<text x='400' y='510' text-anchor='middle' font-size='11' {FONT}>Distinct-2 (higher diversity)</text><text x='35' y='275' text-anchor='middle' transform='rotate(-90 35 275)' font-size='11' {FONT}>AABBA rhyme score (higher form quality)</text>")
    for model,label,color in MODELS:
        points=[(p(float(cells[(model,t)]['distinct_2']),0,1,x0,x1),p(float(cells[(model,t)]['rhyme_score']),0,1,y1,y0)) for t in temps]
        parts.append(f"<polyline points='{path(points)}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        for (x,y),t in zip(points,temps): parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}'/><text x='{x+6:.1f}' y='{y-6:.1f}' font-size='8' {FONT}>{t:.1f}</text>")
    human=next(r for r in rows if r['model']=='human_reference'); hx=p(float(human['distinct_2']),0,1,x0,x1); hy=p(float(human['rhyme_score']),0,1,y1,y0); parts.append(f"<rect x='{hx-5:.1f}' y='{hy-5:.1f}' width='10' height='10' fill='black'/><text x='{hx-8:.1f}' y='{hy-9:.1f}' text-anchor='end' font-size='9' {FONT}>Held-out human reference</text>")
    for n,(_,label,color) in enumerate(MODELS): parts.append(f"<line x1='{110+n*140}' y1='540' x2='{130+n*140}' y2='540' stroke='{color}' stroke-width='3'/><text x='{135+n*140}' y='544' font-size='10' {FONT}>{label}</text>")
    end(parts,root/'quality_diversity_frontier.svg')
if __name__=='__main__': main()
