"""Pilot (2026-11-17): do style/surface attributes that nobody labelled drift under the t5 vote?

Attribute list fixed BEFORE looking at any number. All are measured from the text
(no annotator). Bins = deciles of the real test set (binary attrs: 2 bins).
Metric = Jensen-Shannon distance, natural log (same as vote_correction.jsd).
Arms: MAPLE t5 vote (maple/lam0*) and MAPLE ft vote (maple-ft/lam0*), eps 4, 2026-11-05 run.
"""
import re, glob, os, sys
import numpy as np, pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures/surface-attrs-pilot-2026-11-17"
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*|\d[\d.,]*")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
HEDGE = re.compile(r"\b(may|might|could|possibly|potentially|likely|unlikely|suggest\w*|indicat\w*|appear\w*|putative\w*)\b", re.I)
FIRST = re.compile(r"\b(we|our|us|ours)\b", re.I)
PASSIVE = re.compile(r"\b(is|are|was|were|be|been|being)\s+(\w+ly\s+)?\w+ed\b", re.I)
ACRO = re.compile(r"\b[A-Z]{2,}[A-Za-z0-9-]*\b")
STAT = re.compile(r"(\bp\s*[<=>]\s*0?\.\d|\d\s*%|±|\bCI\b|confidence interval|\bodds ratio\b|\bhazard ratio\b)", re.I)

def mattr(toks, w=50):
    toks = [t.lower() for t in toks]
    if len(toks) <= w:
        return len(set(toks)) / max(len(toks), 1)
    return float(np.mean([len(set(toks[i:i + w])) / w for i in range(0, len(toks) - w + 1, 5)]))

def feats(text):
    text = str(text)
    toks = WORD.findall(text)
    n = max(len(toks), 1)
    sents = [s for s in SENT.split(text.strip()) if s.strip()]
    ns = max(len(sents), 1)
    alpha = [t for t in toks if t[0].isalpha()]
    return dict(
        mean_sent_len=n / ns,
        numeric_rate=sum(any(c.isdigit() for c in t) for t in toks) / n,
        has_stat=float(bool(STAT.search(text))),
        hedge_rate=len(HEDGE.findall(text)) / n,
        first_person_rate=len(FIRST.findall(text)) / n,
        mattr=mattr(toks),
        mean_word_len=float(np.mean([len(t) for t in alpha])) if alpha else 0.0,
        acronym_rate=len(ACRO.findall(text)) / n,
        passive_per_sent=len(PASSIVE.findall(text)) / ns,
        word_count=len(text.split()),
    )

def jsdist(p, q):
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0))
    return float(np.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0)))

def hist(x, edges, w=None):
    idx = np.searchsorted(edges, x, side="right")
    return np.bincount(idx, weights=w, minlength=len(edges) + 1).astype(float)

def length_weights(wc, wc_edges, real_hist):
    """Importance weights that give the survivors the real word_count distribution
    (decile bins of the real set). Bins with no survivors are dropped."""
    b = np.searchsorted(wc_edges, wc, side="right")
    cnt = np.bincount(b, minlength=len(wc_edges) + 1).astype(float)
    target = real_hist / real_hist.sum()
    w = np.where(cnt[b] > 0, target[b] / np.maximum(cnt[b], 1), 0.0)
    covered = target[cnt > 0].sum()
    return w, covered

test = pd.read_csv(f"{ROOT}/nano5-2026-11-16/test_ref_qwen.csv")
F_test = pd.DataFrame([feats(t) for t in test["text"]])
names = [c for c in F_test.columns]
edges = {}
for c in names:
    if c == "has_stat":
        edges[c] = np.array([0.5])
    elif c == "word_count":
        edges[c] = np.arange(25, 1000, 50)          # 50-word bins as in the project
    else:
        edges[c] = np.unique(np.quantile(F_test[c], np.linspace(0.1, 0.9, 9)))
H_test = {c: hist(F_test[c].to_numpy(), edges[c]) for c in names}
WC_EDGES = np.unique(np.quantile(F_test["word_count"], np.linspace(0.1, 0.9, 9)))
WC_REAL = hist(F_test["word_count"].to_numpy(), WC_EDGES)

rng = np.random.default_rng(0)
floor = {c: np.mean([jsdist(hist(F_test[c].to_numpy()[rng.integers(0, len(F_test), len(F_test))], edges[c]), H_test[c])
                     for _ in range(50)]) for c in names}

rows, means = [], []
for arm, pat in [("t5", "maple/lam0*"), ("ft", "maple-ft/lam0*")]:
    for d in sorted(glob.glob(f"{ROOT}/nano5-2026-11-05/eps4.0/{pat}")):
        seed = os.path.basename(d)
        for f in sorted(glob.glob(f"{d}/round_*.csv")):
            r = int(re.search(r"round_(\d+)", f).group(1))
            F = pd.DataFrame([feats(t) for t in pd.read_csv(f)["text"]])
            w, cov = length_weights(F["word_count"].to_numpy(), WC_EDGES, WC_REAL)
            for c in names:
                rows.append(dict(arm=arm, seed=seed, round=r, attr=c, jsd=jsdist(hist(F[c].to_numpy(), edges[c]), H_test[c]),
                                 jsd_lenw=jsdist(hist(F[c].to_numpy(), edges[c], w), H_test[c]), len_mass_covered=cov))
                means.append(dict(arm=arm, seed=seed, round=r, attr=c, mean=F[c].mean()))
L = pd.DataFrame(rows); M = pd.DataFrame(means)
os.makedirs(OUT, exist_ok=True)
L.to_csv(f"{OUT}/jsd_long.csv", index=False); M.to_csv(f"{OUT}/means_long.csv", index=False)
F_test.to_csv(f"{OUT}/test_features.csv", index=False)

# summary: round 0 vs round 3 (all seeds of both arms reach round 3), paired by seed
rlast = 3
S = []
for c in names:
    rec = dict(attr=c, floor=floor[c], rho_wc=F_test[c].rank().corr(F_test["word_count"].rank()),
               real_mean=F_test[c].mean())
    for arm in ["t5", "ft"]:
        a = L[(L.arm == arm) & (L.attr == c)].pivot(index="seed", columns="round", values="jsd")
        a = a.dropna(subset=[0, rlast])
        d = a[rlast] - a[0]
        rec[f"{arm}_r0"] = a[0].mean(); rec[f"{arm}_r{rlast}"] = a[rlast].mean()
        rec[f"{arm}_d"] = d.mean(); rec[f"{arm}_d_se"] = d.std(ddof=1) / np.sqrt(len(d)); rec[f"{arm}_n"] = len(d)
        mm = M[(M.arm == arm) & (M.attr == c) & (M["round"] == rlast)]["mean"].mean()
        rec[f"{arm}_mean_r{rlast}"] = mm
        b = L[(L.arm == arm) & (L.attr == c)].pivot(index="seed", columns="round", values="jsd_lenw").dropna(subset=[0, rlast])
        db = b[rlast] - b[0]
        rec[f"{arm}_lenw_r0"] = b[0].mean(); rec[f"{arm}_lenw_r{rlast}"] = b[rlast].mean()
        rec[f"{arm}_lenw_d"] = db.mean(); rec[f"{arm}_lenw_d_se"] = db.std(ddof=1) / np.sqrt(len(db))
    S.append(rec)
S = pd.DataFrame(S)
S.to_csv(f"{OUT}/summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(S.round(3).to_string(index=False))
print("length mass covered (min over runs):", L["len_mass_covered"].min().round(3))
