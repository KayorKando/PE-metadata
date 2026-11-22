"""Collect the results of downstream/ntp.py: <root>/**/result.json and per_text.csv.

A "train set" is one synthetic (or real) set of texts: a PE run at one round, or one random sample of a CSV.
A "setting" is (tree, arm, lambda, round, variant); its train sets differ by the PE seed (--csv sets: the sample
seed). If a train set was trained with several training seeds, their mean is used as that train set's value.

Written to --out (default: <root>):
  ntp_runs.csv      every result.json as one row
  ntp_settings.csv  per setting: mean, sd and n over PE seeds of ntp_acc, ntp_acc_pred, loss, ppl_maple and of the
                    train-set size (texts, tokens)
  ntp_paired.csv    paired differences to the baseline (--baseline, lambda = 0: the t5 vote), same PE seed, same
                    variant: mean, sd, n and the half-width of the 95% t-interval. Arms of the baseline's tree are
                    compared at the same round. Other trees of the same eps (Aug-PE) are compared "last vs last"
                    (each at its own last round) and at every round both have.
  ntp_groups.csv    the same means and paired differences inside groups of the test set:
                      category, ref_<attr>   tail / head / rare classes (see tail_classes)
                      length                 short / middle / long test texts (50-word bins, see length_groups)
                      position               position of the predicted token: 0-127, 128-255, 256-383, 384-511
  ntp_noise.csv     train sets with several training seeds: sd over them (the noise floor of one number)
  NTP.md            the main tables

Metrics: ntp_acc and ppl_maple use MAPLE's formulas (ntp.metrics); inside the position groups only ntp_acc_pred
exists (correct / predicted tokens).

    python downstream/summarize_ntp.py --root $PE_DATA/ntp --test_ref $PE_DATA/quota_sweep/test_ref_qwen.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from downstream.ntp import POS_EDGES, SEQ_LEN, metrics  # noqa: E402

KEY = ["tree", "arm", "lam", "round", "variant"]
METRICS = ["ntp_acc", "ntp_acc_pred", "loss", "ppl_maple"]
SIZE = ["n_train_texts", "n_train_tokens", "final_train_loss"]
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t975(df):
    return float("nan") if df < 1 else (T975[df - 1] if df <= len(T975) else 1.96)


# ---------------- groups of the test set ----------------
def smallest_classes(labels, mass=0.20):
    """The smallest classes, taken in order of size while their total share stays at most `mass`."""
    share = labels.value_counts(normalize=True)
    share = share.iloc[np.lexsort((share.index.astype(str), share.values))]      # by share, ties by name
    out, cum = [], 0.0
    for c, p in share.items():
        if cum + p > mass:
            break
        cum += p
        out.append(c)
    return out


def tail_classes(labels, mass=0.20, min_n=10):
    """Tail classes as in figures/tail-counts-2026-11-18: the smallest classes summing to at most 20 % of the test
    set, each with at least min_n test texts."""
    cnt = labels.value_counts()
    return [c for c in smallest_classes(labels, mass) if cnt[c] >= min_n]


def class_groups(labels, mass=0.20, min_n=10):
    """Per text: 'tail', 'head', or 'rare' (among the smallest classes but with fewer than min_n test texts).
    None if there is no tail class."""
    small, tail = smallest_classes(labels, mass), tail_classes(labels, mass, min_n)
    if not tail:
        return None
    return np.where(labels.isin(tail), "tail", np.where(labels.isin(small), "rare", "head"))


def length_groups(bins, mass=0.20):
    """Per text: 'middle' = the 50-word bins that hold the bulk of the test set (all but the smallest bins, which
    sum to at most 20 %), 'short' / 'long' = bins below / above them."""
    small = set(smallest_classes(bins, mass))
    head = sorted(set(bins) - small)
    lo, hi = head[0], head[-1]
    return np.where(bins < lo, "short", np.where(bins > hi, "long", "middle"))


def test_groups(test, min_n=10):
    """{grouping: array of group names, one per test text}."""
    out = {}
    for col in [c for c in test.columns if c == "category" or (c.startswith("ref_") and c != "ref_word_count")]:
        g = class_groups(test[col].astype(str), min_n=min_n)
        if g is not None:
            out[col] = g
    if "ref_word_count" in test:
        bins = test["ref_word_count"].astype(float)
    else:                                                  # MAPLE's bins: round(words / 50) * 50
        bins = pd.Series(np.round(test["text"].astype(str).str.split().str.len() / 50) * 50)
    out["length"] = length_groups(bins)
    return out


def pos_names():
    edges = [0, *POS_EDGES, SEQ_LEN]
    return [f"{a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:])]


def group_rows(pt, groups):
    """Metrics of one run inside every group: rows (grouping, group, n_test_texts, n_test_tokens, metrics...)."""
    rows = []
    for name, g in groups.items():
        for val in pd.unique(g):
            m = metrics(pt[g == val])
            rows.append(dict(grouping=name, group=val, **m))
    for b, val in enumerate(pos_names()):
        k, c = pt[f"k{b}"].sum(), pt[f"c{b}"].sum()
        if k:
            rows.append(dict(grouping="position", group=val, n_test_texts=int((pt[f"k{b}"] > 0).sum()),
                             n_test_tokens=int(k), ntp_acc_pred=float(c / k)))
    return rows


# ---------------- tables ----------------
def load(root):
    rows = []
    for f in sorted(Path(root).rglob("result.json")):
        r = json.loads(f.read_text())
        r["dir"] = str(f.parent)
        rows.append(r)
    R = pd.DataFrame(rows)
    if len(R):
        R["lam"] = pd.to_numeric(R["lam"], errors="coerce")
        R["round"] = pd.to_numeric(R["round"], errors="coerce")
    return R


def per_set(R, cols, extra=()):
    """One row per train set: the mean over its training seeds."""
    by = KEY + ["seed", "kind", *extra]
    S = R.groupby(by, dropna=False)[cols].mean().reset_index()
    return S.merge(R.groupby(by, dropna=False)["is_last"].all().reset_index(), on=by)


def settings(S, cols, extra=()):
    by = KEY + ["kind", *extra]
    g = S.groupby(by, dropna=False)
    out = g[cols].mean()
    out = out.join(g[cols].std(ddof=1).add_suffix("_sd"))
    out.insert(0, "n", g.size())
    out.insert(0, "is_last", g["is_last"].all())
    return out.reset_index()


def paired(S, cols, baseline, extra=()):
    """Paired differences to the baseline arm (lambda 0), by PE seed. See the module docstring."""
    S = S[S.kind == "pe"]
    rows = []

    def add(d, cmp):
        if d.empty:
            return
        for keys, g in d.groupby(["tree", "arm", "lam", "variant", "round", "round_base", *extra], dropna=False):
            for c in cols:
                x = (g[c] - g[c + "_base"]).dropna()
                if len(x) == 0:
                    continue
                sd = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
                rows.append(dict(zip(["tree", "arm", "lam", "variant", "round", "round_base", *extra], keys), comparison=cmp,
                                 is_last=bool(g["is_last"].all()), metric=c, diff=float(x.mean()), sd=sd, n=len(x),
                                 ci95=t975(len(x) - 1) * sd / np.sqrt(len(x)), mean=float(g.loc[x.index, c].mean()),
                                 base_mean=float(g.loc[x.index, c + "_base"].mean())))

    base_all = S[(S.arm == baseline) & (S.lam == 0)]
    for tree in sorted(base_all.tree.unique()):
        base = base_all[base_all.tree == tree]
        b = base[["variant", "seed", "round", "is_last", *extra, *cols]].rename(
            columns={c: c + "_base" for c in cols} | {"round": "round_base", "is_last": "is_last_base"})
        same = S[(S.tree == tree) & ~((S.arm == baseline) & (S.lam == 0))]
        d = same.merge(b, left_on=["variant", "seed", "round", *extra], right_on=["variant", "seed", "round_base", *extra])
        for t, g in d.groupby("round"):
            add(g, f"round {int(t)}")
        eps = tree.split("/")[-1]
        other = S[(S.tree != tree) & (S.tree.str.split("/").str[-1] == eps) & ~S.tree.isin(base_all.tree.unique())]
        d = other[other.is_last].merge(b[b.is_last_base], on=["variant", "seed", *extra])
        add(d, "last vs last")
        d = other.merge(b, left_on=["variant", "seed", "round", *extra], right_on=["variant", "seed", "round_base", *extra])
        for t, g in d.groupby("round"):
            add(g, f"round {int(t)}")
    return pd.DataFrame(rows)


def noise(R):
    g = R.groupby(["train_set", "variant"])
    out = g[["ntp_acc", "loss"]].agg(["mean", "std"])
    out.columns = [f"{a}_{'sd' if b == 'std' else b}" for a, b in out.columns]
    out.insert(0, "n_train_seeds", g.size())
    return out[out.n_train_seeds > 1].reset_index()


# ---------------- report ----------------
def fmt(x, nd=4):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def pm(a, b, nd=4):
    return fmt(a, nd) + (f" ± {fmt(b, nd)}" if fmt(b, nd) else "")


def arm_name(r):
    lam = "" if pd.isna(r.lam) else f" λ{r.lam:g}"
    return f"{r.arm}{lam}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def report(R, ST, P, G, N, baseline, root):
    L = [f"# NTP downstream ({root})", "",
         f"{len(R)} runs, {R.train_set.nunique()} train sets, test set: {int(R.n_test_texts.iloc[0])} texts "
         f"({', '.join(sorted(R.test_sha.unique()))}). Baseline for Δ: `{baseline}` λ0 (t5 vote), same PE seed, "
         "same variant; ± after Δ = half-width of the 95% t-interval; ± after a mean = sd over PE seeds.",
         "`ntp_acc` = MAPLE's formula (correct / all tokens). Variants: `full` = all texts; `tok<B>` = texts "
         "subsampled to B training tokens; `len<L>` = texts cut to L tokens.", ""]
    pe = ST[ST.kind == "pe"]

    def delta(r, cmp, metric="ntp_acc"):
        if P.empty:
            return ""
        p = P[(P.tree == r.tree) & (P.arm == r.arm) & (P.variant == r.variant) & (P.metric == metric)
              & (P.comparison == cmp) & ((P.lam == r.lam) | (P.lam.isna() & pd.isna(r.lam)))
              & ((P["round"] == r["round"]))]
        return "" if p.empty else f"{p['diff'].item():+.4f} ± {fmt(p.ci95.item())} (n={p.n.item()})"

    for title, sel in [("Last round", pe[pe.is_last]), ("Round 0", pe[(pe["round"] == 0) & ~pe.is_last])]:
        for (eps, variant), g in sel.assign(eps=sel.tree.str.split("/").str[-1]).groupby(["eps", "variant"]):
            rows = []
            for _, r in g.sort_values(["tree", "arm", "lam"]).iterrows():
                is_base_tree = ((pe.tree == r.tree) & (pe.arm == baseline) & (pe.lam == 0)).any()
                cmp = f"round {int(r['round'])}" if is_base_tree else ("last vs last" if r.is_last else f"round {int(r['round'])}")
                rows.append([arm_name(r), int(r["round"]), int(r.n), f"{r.n_train_texts:.0f}", f"{r.n_train_tokens / 1000:.0f}k",
                             pm(r.ntp_acc, r.ntp_acc_sd), delta(r, cmp), pm(r.loss, r.loss_sd, 3), delta(r, cmp, "loss")])
            L += [f"## {title}, {eps}, variant `{variant}`", "",
                  md_table(["arm", "round", "seeds", "train texts", "train tokens", "ntp_acc", "Δ ntp_acc", "loss", "Δ loss"],
                           rows), ""]
    ref = ST[ST.kind == "csv"]
    if len(ref):
        rows = [[r.arm, r.variant, int(r.n), f"{r.n_train_texts:.0f}", f"{r.n_train_tokens / 1000:.0f}k",
                 pm(r.ntp_acc, r.ntp_acc_sd), pm(r.loss, r.loss_sd, 3)] for _, r in ref.sort_values(["arm", "variant"]).iterrows()]
        L += ["## Reference train sets (random rows of a CSV; ± = sd over sample seeds)", "",
              md_table(["set", "variant", "samples", "train texts", "train tokens", "ntp_acc", "loss"], rows), ""]
    if len(N):
        rows = [[r.train_set, r.variant, int(r.n_train_seeds), fmt(r.ntp_acc_mean), fmt(r.ntp_acc_sd), fmt(r.loss_sd, 3)]
                for _, r in N.iterrows()]
        L += ["## Noise of one number: sd over training seeds (same train set)", "",
              md_table(["train set", "variant", "training seeds", "ntp_acc", "sd ntp_acc", "sd loss"], rows), ""]
    if len(G):
        last = G[(G.kind == "pe") & G.is_last & G.grouping.isin(["length", "category", "position"])]
        for (eps, variant), g in last.assign(eps=last.tree.str.split("/").str[-1]).groupby(["eps", "variant"]):
            rows = []
            for _, r in g.sort_values(["grouping", "group", "tree", "arm", "lam"]).iterrows():
                m = "ntp_acc_pred" if r.grouping == "position" else "ntp_acc"
                d = "" if pd.isna(r.get(f"d_{m}")) else f"{r[f'd_{m}']:+.4f} ± {fmt(r[f'ci95_{m}'])}"
                rows.append([r.grouping, r.group, int(r.n_test_texts), arm_name(r), pm(r[m], r[f"{m}_sd"]), d])
            L += [f"## Inside groups of the test set: last round, {eps}, variant `{variant}`", "",
                  "Position groups: correct / predicted tokens. Other groupings (the ref_ attributes) are in ntp_groups.csv.", "",
                  md_table(["grouping", "group", "test texts", "arm", "accuracy", "Δ vs baseline"], rows), ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="output root of downstream/ntp.py, e.g. $PE_DATA/ntp")
    p.add_argument("--test_ref", required=True, help="the test CSV of the runs, with category / ref_<attr> columns")
    p.add_argument("--out", default=None)
    p.add_argument("--baseline", default="maple", help="arm of the t5 vote (its lambda = 0 runs)")
    p.add_argument("--min_group", type=int, default=10, help="a tail class needs this many test texts")
    args = p.parse_args(argv)
    out = Path(args.out or args.root)
    R = load(args.root)
    if R.empty:
        sys.exit(f"no result.json under {args.root}")
    test = pd.read_csv(args.test_ref)
    if R.n_test_texts.nunique() != 1 or R.n_test_texts.iloc[0] != len(test) or R.test_sha.nunique() != 1:
        sys.exit(f"the runs under {args.root} do not share one test set of {len(test)} texts "
                 f"(n_test_texts {sorted(R.n_test_texts.unique())}, {R.test_sha.nunique()} test sets)")
    out.mkdir(parents=True, exist_ok=True)
    cols = METRICS + SIZE
    S = per_set(R, cols)
    ST = settings(S, cols)
    P = paired(S, METRICS, args.baseline)
    N = noise(R)

    groups = test_groups(test, args.min_group)
    rows = []
    for r in R.itertuples():
        pt = pd.read_csv(Path(r.dir) / "per_text.csv")
        base = {k: getattr(r, k) for k in KEY + ["seed", "kind", "is_last", "train_seed"]}
        rows += [dict(base, **g) for g in group_rows(pt, groups)]
    GR = pd.DataFrame(rows)
    gcols = ["ntp_acc", "ntp_acc_pred", "loss"]
    GS = per_set(GR, gcols + ["n_test_texts", "n_test_tokens"], extra=("grouping", "group"))
    G = settings(GS, gcols, extra=("grouping", "group"))
    G = G.merge(GS.groupby(["grouping", "group"])[["n_test_texts", "n_test_tokens"]].first().reset_index(),
                on=["grouping", "group"])
    GP = paired(GS, gcols, args.baseline, extra=("grouping", "group"))
    if len(GP):                 # one row per setting and group: the comparison of the main tables
        is_base_tree = GP.tree.isin(S[(S.arm == args.baseline) & (S.lam == 0)].tree.unique())
        GP = GP[is_base_tree | (GP.comparison == "last vs last") | ~GP.is_last]
        on = ["tree", "arm", "lam", "variant", "round", "grouping", "group"]
        for m in gcols:         # NaN keys (lambda of Aug-PE) match each other in a pandas merge
            sub = GP[GP.metric == m].drop_duplicates(on)[on + ["diff", "ci95"]]
            G = G.merge(sub.rename(columns={"diff": f"d_{m}", "ci95": f"ci95_{m}"}), on=on, how="left")

    R.drop(columns=["dir"]).to_csv(out / "ntp_runs.csv", index=False)
    ST.to_csv(out / "ntp_settings.csv", index=False)
    P.to_csv(out / "ntp_paired.csv", index=False)
    G.to_csv(out / "ntp_groups.csv", index=False)
    N.to_csv(out / "ntp_noise.csv", index=False)
    (out / "NTP.md").write_text(report(R, ST, P, G, N, args.baseline, args.root))
    print(f"{len(R)} runs, {len(ST)} settings -> {out}/NTP.md, ntp_runs.csv, ntp_settings.csv, ntp_paired.csv, "
          "ntp_groups.csv, ntp_noise.csv")


if __name__ == "__main__":
    main()
