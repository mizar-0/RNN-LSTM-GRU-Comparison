"""Create the blinded Claude LLM-judge score temperature plot."""
from __future__ import annotations
import csv
from pathlib import Path

MODELS=(('rnn','Simple RNN','#4C78A8'),('birnn','BiRNN','#F58518'),('lstm','LSTM','#54A24B'),('gru','GRU','#B279A2'))
TEMPS=(.3,.5,.8,1.3)
def p(v,a,b,x0,x1): return x0+(v-a)/(b-a)*(x1-x0)
def main():
 root=Path('results/sweep'); rows=list(csv.DictReader((root/'manual_score_summary.csv').open())); data={(r['model'],float(r['temperature'])):r for r in rows}; x0,x1,y0,y1=100,700,85,450
 out=["<svg xmlns='http://www.w3.org/2000/svg' width='780' height='540' viewBox='0 0 780 540' role='img'><title>Blinded Claude LLM judge scores by temperature</title><rect width='100%' height='100%' fill='white'/>","<text x='390' y='32' text-anchor='middle' font-family='Arial' font-size='20' font-weight='bold'>Blinded Claude LLM judge scores</text>",f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#333'/><line x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}' stroke='#333'/>"]
 for i in range(6):
  v=i; y=p(v,0,5,y1,y0); out.append(f"<line x1='{x0}' y1='{y:.1f}' x2='{x1}' y2='{y:.1f}' stroke='#ddd'/><text x='{x0-8}' y='{y+4:.1f}' text-anchor='end' font-family='Arial' font-size='10'>{v}</text>")
 for t in TEMPS:
  x=p(t,.3,1.3,x0,x1); out.append(f"<text x='{x:.1f}' y='{y1+18}' text-anchor='middle' font-family='Arial' font-size='10'>{t:.1f}</text>")
 out.append("<text x='400' y='495' text-anchor='middle' font-family='Arial' font-size='11'>Temperature</text><text x='35' y='270' text-anchor='middle' transform='rotate(-90 35 270)' font-family='Arial' font-size='11'>Mean LLM judge score (1-5)</text>")
 for n,(model,label,color) in enumerate(MODELS):
  pts=[]
  for t in TEMPS:
   r=data[(model,t)]; v=float(r['manual_mean']); lo=float(r['manual_mean_ci_low']); hi=float(r['manual_mean_ci_high']); x=p(t,.3,1.3,x0,x1); y=p(v,0,5,y1,y0); pts.append((x,y)); out.append(f"<line x1='{x:.1f}' y1='{p(lo,0,5,y1,y0):.1f}' x2='{x:.1f}' y2='{p(hi,0,5,y1,y0):.1f}' stroke='{color}' stroke-width='1.5'/><line x1='{x-5:.1f}' y1='{p(lo,0,5,y1,y0):.1f}' x2='{x+5:.1f}' y2='{p(lo,0,5,y1,y0):.1f}' stroke='{color}'/><line x1='{x-5:.1f}' y1='{p(hi,0,5,y1,y0):.1f}' x2='{x+5:.1f}' y2='{p(hi,0,5,y1,y0):.1f}' stroke='{color}'/>")
  out.append(f"<polyline points='{' '.join(f'{x:.1f},{y:.1f}' for x,y in pts)}' fill='none' stroke='{color}' stroke-width='2.5'/>")
  for x,y in pts: out.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}'/>")
  out.append(f"<line x1='{115+n*145}' y1='520' x2='{135+n*145}' y2='520' stroke='{color}' stroke-width='3'/><text x='{140+n*145}' y='524' font-family='Arial' font-size='10'>{label}</text>")
 out.append("<text x='700' y='70' text-anchor='end' font-family='Arial' font-size='10'>Points: mean; whiskers: 95% bootstrap CI; n=5 per cell</text></svg>")
 (root/'manual_score_results.svg').write_text('\n'.join(out),encoding='utf-8')
if __name__=='__main__': main()
