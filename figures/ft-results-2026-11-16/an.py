import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t as T
R = Path("data/results/nano5-2026-11-16")
rows=[]
for f in R.glob("eps*/*/lam*/metrics.jsonl"):
    m=re.match(r"lam([0-9.]+)(?:_seed(\d+))?$", f.parent.name)
    ex={}
    for g in f.parent.glob("mauve_*.jsonl"):
        for l in g.read_text().splitlines():
            e=json.loads(l); ex.setdefault(e["round"],{}).update({k:v for k,v in e.items() if k.startswith("mauve_")})
    for l in f.read_text().splitlines():
        r=json.loads(l); r.update(ex.get(r["round"],{}))
        r.update(eps=f.parent.parent.parent.name, arm=f.parent.parent.name, lam=float(m.group(1)), seed=int(m.group(2) or 0))
        rows.append(r)
D=pd.DataFrame(rows)
D=D.groupby(["eps","arm","lam","seed","round"],sort=False).tail(1)  # duplicated rounds (ftt5): keep last line
D.to_csv("all_rounds.csv", index=False)
last=D.sort_values("round").groupby(["eps","arm","lam","seed"]).tail(1)
K=["jsd_blind_mean","jsd_other_mean","jsd_mean","mauve","mauve_gpt2-large"]
print(last.groupby(["eps","arm","lam"])[K+["round"]].agg(["mean","count"]).round(4).to_string())
def ci(x):
    n=len(x); return x.mean(), (T.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n) if n>1 else np.nan), n
out=[]
for (eps,arm,lam),g in last.groupby(["eps","arm","lam"]):
    if arm in ("maple","classifier"): continue
    base = "maple" if lam==0 else arm.split("-")[0]
    b=last[(last.eps==eps)&(last.arm==base)&(last.lam==lam)].set_index("seed")
    gg=g.set_index("seed"); s=gg.index.intersection(b.index)
    for k in K:
        m,c,n=ci(gg.loc[s,k]-b.loc[s,k]); out.append(dict(eps=eps,arm=arm,lam=lam,metric=k,mean=m,ci95=c,n=n))
P=pd.DataFrame(out); P.to_csv("paired_vs_t5.csv",index=False)
print(P.assign(cell=[f"{m:+.3f}±{c:.3f}" for m,c in zip(P["mean"],P.ci95)]).pivot_table(index=["eps","arm","lam"],columns="metric",values="cell",aggfunc="first")[K].to_string())
