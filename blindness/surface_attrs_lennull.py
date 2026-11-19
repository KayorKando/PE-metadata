"""Length-shadow check for surface_attrs_pilot.py: null = real test texts resampled (with replacement,
within 50-word bins) to each survivor set's word-count histogram. If an attribute's drift were only a
side effect of length drift, its JSD would equal the null's. excess = observed JSD - null JSD."""
import sys, glob, os, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
OUT = "figures/surface-attrs-pilot-2026-11-17"
F_test = pd.read_csv(f"{OUT}/test_features.csv")
L = pd.read_csv(f"{OUT}/jsd_long.csv")
src = open(os.path.join(os.path.dirname(__file__), "surface_attrs_pilot.py")).read()
ns = {}; exec(src.split("test = pd.read_csv")[0], ns)   # helpers only
jsdist, hist, feats = ns["jsdist"], ns["hist"], ns["feats"]
names = [c for c in F_test.columns if c != "word_count"]
edges = {c: (np.array([0.5]) if c == "has_stat" else np.unique(np.quantile(F_test[c], np.linspace(0.1, 0.9, 9)))) for c in names}
H_test = {c: hist(F_test[c].to_numpy(), edges[c]) for c in names}
wbin_t = (F_test["word_count"] // 50).astype(int).to_numpy()
pools = {b: np.where(wbin_t == b)[0] for b in np.unique(wbin_t)}
avail = np.array(sorted(pools))
rng = np.random.default_rng(1)
rows = []
for arm, pat in [("t5", "maple/lam0*"), ("ft", "maple-ft/lam0*")]:
    for d in sorted(glob.glob(f"results/nano5-2026-11-05/eps4.0/{pat}")):
        for f in sorted(glob.glob(f"{d}/round_*.csv")):
            r = int(re.search(r"round_(\d+)", f).group(1))
            if r not in (0, 3): continue
            wc = pd.read_csv(f)["text"].astype(str).str.split().str.len().to_numpy()
            wb = (wc // 50).astype(int)
            wb = avail[np.abs(avail[None, :] - wb[:, None]).argmin(1)]   # nearest bin with real texts
            for rep in range(10):
                idx = np.concatenate([rng.choice(pools[b], size=(wb == b).sum(), replace=True) for b in np.unique(wb)])
                for c in names:
                    rows.append(dict(arm=arm, seed=os.path.basename(d), round=r, attr=c, rep=rep,
                                     null=jsdist(hist(F_test[c].to_numpy()[idx], edges[c]), H_test[c])))
N = pd.DataFrame(rows).groupby(["arm", "seed", "round", "attr"], as_index=False)["null"].mean()
X = L.merge(N, on=["arm", "seed", "round", "attr"])
X["excess"] = X["jsd"] - X["null"]
X.to_csv(f"{OUT}/jsd_length_null.csv", index=False)
out = []
for (arm, c), g in X.groupby(["arm", "attr"]):
    p = g.pivot(index="seed", columns="round", values="excess").dropna()
    q = g.pivot(index="seed", columns="round", values="null").dropna()
    o = g.pivot(index="seed", columns="round", values="jsd").dropna()
    d = p[3] - p[0]
    out.append(dict(arm=arm, attr=c, obs_r0=o[0].mean(), obs_r3=o[3].mean(), null_r0=q[0].mean(), null_r3=q[3].mean(),
                    excess_r0=p[0].mean(), excess_r3=p[3].mean(), d_excess=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d))))
O = pd.DataFrame(out).sort_values(["arm", "attr"])
O.to_csv(f"{OUT}/summary_length_null.csv", index=False)
pd.set_option("display.width", 250)
print(O.round(3).to_string(index=False))
