"""Dataset layer: which MAPLE dataset a process works on, and everything about it that the PE code needs.

The dataset is set by the environment variable PE_DATASET: biorxiv (default) or openreview. Each process
works on one dataset; the Nano5 scripts export PE_DATASET and point PE_DATA at that dataset's own data root
(which holds a DATASET marker file, see check_data_root).

    METADATA            the columns MAPLE puts in its generation prompts (the AIM metadata)
    LABEL_ATTRS         attributes fixed by a class-conditional vote (OpenReview: area x recommendation =
                        PE.LABEL_ID, as MAPLE's openreview/MAPLE/MAPLE.py and PE.py); empty for bioRxiv
    EPS_SETTINGS        eps -> (T, noise multiplier) of the MAPLE runs (T + 1 votes: rounds 0..T)
    AUGPE_EPS_SETTINGS  the same for the Aug-PE baseline (all of eps to the vote)
    load_private / load_test / load_val / load_aim   MAPLE's CSVs, cleaned the same way for every script

OpenReview specifics (MAPLE repo, openreview/):
  - the private train CSV calls the length column "word count"; it is renamed to word_count.
  - the AIM CSVs have no PE.LABEL_ID: it is mapped from (area, recommendation) with the private set's
    mapping; AIM rows with a pair the private set does not have (73 of 8,396 at eps 4) get no id and are
    dropped (MAPLE samples round-0 metadata only for the private set's label ids, so they are never used).
  - the number of survivors per label id is MAPLE's get_num_sample_per_target_id: proportional to the
    private label counts (MAPLE's convention, kept as is).
  - MAPLE ships no annotation prompt for OpenReview (only Gemini labels); openreview_prompt() below is ours,
    in the style of MAPLE's bioRxiv prompt, over the schema in blindness/schema.py.
  - eps 4 and 2: MAPLE's script runs T = 2 (3 votes). We run T = 4 (5 votes, as on bioRxiv) with the noise
    scaled by sqrt(5/3), which keeps the PE budget rho_PE exactly (Gaussian composition: rho = votes / (2 sigma^2)).
    eps 1: MAPLE's T = 2. eps inf: T = 9, no noise.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from blindness.schema import DATASET, SCHEMA

LABEL_ID = "PE.LABEL_ID"


def rescale_noise(sigma, votes_from, votes_to):
    """Noise multiplier for votes_to Gaussian votes with the same total zCDP budget as votes_from votes at sigma."""
    return float(sigma * np.sqrt(votes_to / votes_from))


_CFG = {
    "biorxiv": dict(
        dir="biorxiv",
        metadata=["primary_research_area", "model_organism", "experimental_approach", "dominant_data_type",
                  "research_focus_scale", "disease_mention", "sample_size", "research_goal", "word_count"],
        label_attrs=[],
        # MAPLE main_experiment.sh, MAPLE.py runs
        eps={"4.0": (4, 2.5787580697142025), "2.0": (4, 4.7482491352519505), "1.0": (2, 6.869194837509232),
             "inf": (9, 0.0)},
        # MAPLE main_experiment.sh, PE.py (Aug-PE) runs
        augpe_eps={"4.0": (9, 3.214362874574832), "2.0": (9, 5.872135064401613), "1.0": (9, 10.889585799108582),
                   "inf": (9, 0.0)},
        blind=["word_count", "sample_size", "dominant_data_type"],      # Experiment 1 (k-NN, round-0 pool)
        ft_backbone="biomedbert",
        annot_max_tokens=512,
        private="biorxiv_train_metadata.csv", test="biorxiv_test_metadata.csv", val="biorxiv_valid_metadata.csv",
        aim="biorxiv_train_metadata_aim_eps{eps}_(1,9).csv",
    ),
    "openreview": dict(
        dir="openreview",
        metadata=["area", "recommendation", "primary_strength", "primary_weakness",
                  "keywords_group_1", "keywords_group_2", "keywords_group_3", "word_count"],
        label_attrs=["area", "recommendation"],
        # MAPLE privacy notebook, OpenReview: 3 votes at 1.876 (eps 4), 3.420 (eps 2), 6.325 (eps 1)
        eps={"4.0": (4, rescale_noise(1.876414555237503, 3, 5)), "2.0": (4, rescale_noise(3.4202124076206792, 3, 5)),
             "1.0": (2, 6.325387702318415), "inf": (9, 0.0)},
        # MAPLE openreview/MAPLE/main_experiment.sh, PE.py runs
        augpe_eps={"4.0": (9, 3.000778920378073), "2.0": (9, 5.418702035826035), "1.0": (9, 9.935410662820507),
                   "inf": (9, 0.0)},
        blind=None,                     # chosen on the round-0 pool: blindness/pick_blind.py -> $PE_DATA/blind.json
        ft_backbone="scibert",
        annot_max_tokens=768,           # 12 attributes with long option strings
        private="train_dataset_with_metadata_v4.csv", test="test_dataset_with_metadata_v4.csv",
        val="val_dataset_with_metadata_v4.csv", aim="openreview_train_metadata_aim_eps{eps}_(1,9).csv",
    ),
}
CFG = _CFG[DATASET]
METADATA = CFG["metadata"]
LABEL_ATTRS = CFG["label_attrs"]
CLASS_CONDITIONAL = bool(LABEL_ATTRS)
EPS_SETTINGS = CFG["eps"]
AUGPE_EPS_SETTINGS = CFG["augpe_eps"]
DEFAULT_BLIND = CFG["blind"]
FT_BACKBONE = CFG["ft_backbone"]
ANNOT_MAX_TOKENS = CFG["annot_max_tokens"]
# attributes the blind screen may pick: the AIM attributes the quota has a target for, minus the vote's classes
BLIND_CANDIDATES = [a for a in METADATA if a not in LABEL_ATTRS]


# ---------------- files ----------------
def data_dir(maple_dir):
    return Path(maple_dir) / CFG["dir"]


def maple_code(maple_dir):
    """MAPLE's code dir of this dataset (MAPLE.py, PE.py, util.py)."""
    return data_dir(maple_dir) / "MAPLE"


def _read(path):
    df = pd.read_csv(path)
    return df.rename(columns={"word count": "word_count"})


def _with_text(df):
    return df[df["text"].notna()].reset_index(drop=True)


@lru_cache(maxsize=None)
def _label_map(maple_dir):
    priv = _read(data_dir(maple_dir) / CFG["private"])
    m = priv[LABEL_ATTRS + [LABEL_ID]].drop_duplicates()
    if m[LABEL_ATTRS].duplicated().any():
        raise ValueError(f"{LABEL_ID} is not a function of {LABEL_ATTRS} in the private set")
    return {tuple(r[:-1]): int(r[-1]) for r in m.itertuples(index=False)}


def add_label_id(df, maple_dir):
    """OpenReview: PE.LABEL_ID from (area, recommendation); rows with a pair the private set lacks are dropped."""
    if not CLASS_CONDITIONAL or LABEL_ID in df:
        return df
    mp = _label_map(str(maple_dir))
    ids = [mp.get(tuple(r)) for r in df[LABEL_ATTRS].itertuples(index=False)]
    df = df.assign(**{LABEL_ID: ids})
    return df[df[LABEL_ID].notna()].astype({LABEL_ID: int}).reset_index(drop=True)


def load_private(maple_dir):
    """The private set (rows with text). OpenReview: with PE.LABEL_ID and word_count."""
    return _with_text(_read(data_dir(maple_dir) / CFG["private"]))


def load_test(maple_dir):
    return _with_text(_read(data_dir(maple_dir) / CFG["test"]))


def load_val(maple_dir):
    """The validation set as MAPLE reads it (no filtering): the donated in-context examples come from it."""
    return _read(data_dir(maple_dir) / CFG["val"])


def load_aim(maple_dir, eps):
    """The metadata MAPLE samples its round-0 prompts from: AIM output at eps, or the private set's own labels
    at eps = inf (non-private reference). OpenReview: with PE.LABEL_ID."""
    if eps == "inf":
        df = _read(data_dir(maple_dir) / CFG["private"])
    else:
        df = _read(data_dir(maple_dir) / CFG["aim"].format(eps=f"{float(eps):.1f}"))
    return add_label_id(df, maple_dir)


# ---------------- class-conditional PE (MAPLE / Aug-PE on OpenReview) ----------------
def num_sample_per_id(priv_ids, num_sample):
    """MAPLE's Voting_histogram.get_num_sample_per_target_id: survivors per label id, proportional to the private
    label counts, largest remainder. Same order of ids (first appearance) and the same argsort as MAPLE."""
    id_dict = {}
    for index, i in enumerate(list(priv_ids)):
        id_dict.setdefault(i, []).append(index)
    id_list = list(id_dict.keys())
    count_list = np.array([len(id_dict[i]) for i in id_list])
    target = count_list / np.sum(count_list) * num_sample
    num = np.floor(target).astype(int)
    left = num_sample - np.sum(num)
    ids = np.argsort(target - num)[::-1]
    num[ids[:left]] += 1
    assert np.sum(num) == num_sample
    if np.any(num == 0):
        raise ValueError("num_samples is so small that the number of samples for some label ids is zero.")
    return {int(i): int(n) for i, n in zip(id_list, num)}


def class_vote(X_priv, priv_groups, X_cand, cand_groups):
    """Vote histogram over the candidates. Without groups: every private point votes for its nearest candidate.
    With groups (OpenReview): only among the candidates of its own label id (MAPLE's per-PE.LABEL_ID vote)."""
    from vote_correction import nearest_pool_index
    if priv_groups is None:
        return np.bincount(nearest_pool_index(X_priv, X_cand), minlength=len(X_cand)).astype(float)
    hist = np.zeros(len(X_cand))
    priv_groups, cand_groups = np.asarray(priv_groups), np.asarray(cand_groups)
    for g in np.unique(priv_groups):
        ci, pi = np.flatnonzero(cand_groups == g), np.flatnonzero(priv_groups == g)
        if len(ci) == 0:
            continue          # cannot happen in MAPLE (every id has (L + 1) K_g candidates); its votes are lost
        hist[ci] += np.bincount(nearest_pool_index(X_priv[pi], X_cand[ci]), minlength=len(ci))
    return hist


def select_per_class(scores, groups, k_per_class, select_fn):
    """select_fn(local_scores, local_index, K) -> (local selected indices, info); run per label id with its K_g,
    concatenated in k_per_class order (MAPLE: per target id, then pd.concat). Returns (indices, summed info)."""
    sel, info = [], {"shortfall": 0, "moved": 0}
    groups = np.asarray(groups)
    for g, k in k_per_class.items():
        ci = np.flatnonzero(groups == g)
        if len(ci) == 0:
            continue
        s, inf = select_fn(np.asarray(scores)[ci], ci, k)
        sel.append(ci[s])
        for key in info:
            info[key] += int(inf.get(key, 0))
    return (np.concatenate(sel) if sel else np.zeros(0, int)), info


# ---------------- annotation prompt ----------------
def openreview_prompt(text):
    """Annotation prompt for OpenReview reviews (MAPLE ships none), in the style of MAPLE's bioRxiv prompt."""
    lines = []
    notes = {
        "area": "The research area of the reviewed paper.",
        "recommendation": "The reviewer's overall recommendation (score) for the paper.",
        "primary_contribution_type": "The main type of contribution of the reviewed paper, as the review describes it.",
        "primary_strength": "The main strength of the paper that the reviewer points out.",
        "primary_weakness": "The main weakness of the paper that the reviewer points out.",
        "novelty_assessment": "How the reviewer judges the novelty of the paper.",
        "reviewer_tone": "The overall tone of the review.",
        "clarity_and_presentation_quality": "How the reviewer judges the clarity and presentation of the paper.",
        "actionability_of_feedback": "How actionable the reviewer's feedback is for the authors.",
        "keywords_group_1": "The keyword from this list that best fits the paper's topic, or Other if none fits.",
        "keywords_group_2": "The keyword from this list that best fits the paper's topic, or Other if none fits.",
        "keywords_group_3": "The keyword from this list that best fits the paper's topic, or Other if none fits.",
    }
    attrs = list(SCHEMA)
    for i, a in enumerate(attrs):
        comma = "," if i < len(attrs) - 1 else ""
        lines.append(f'  "{a}": "<{"|".join(SCHEMA[a])}>"{comma} // {notes.get(a, "")}')
    schema = "\n".join(lines)
    return (
        "You are an expert assistant for analyzing machine learning peer reviews. Your task is to carefully read a "
        "peer review of a paper submitted to ICLR (from OpenReview) and extract the specified features according "
        "to the schema provided.\n\n"
        "Output exactly one JSON object with no extra text or explanations.\n\n"
        "**CRITICAL INSTRUCTION 1:** For any field in the JSON schema that lists specific options (e.g., "
        "\"<Option1|Option2|...>\"), you MUST select one of the provided options exactly as it is written. Do not "
        "invent, alter, or combine options. Failure to use an exact option from the list will be considered an "
        "error.\n\n"
        "**CRITICAL INSTRUCTION 2:** Ensure the value chosen for a field is appropriate for that field's specific "
        "definition. Do not use an option from one field (e.g., 'Good (Mostly clear)' from "
        "`clarity_and_presentation_quality`) as the value for another field.\n\n"
        f"Use this schema:\n```json\n{{\n{schema}\n}}\n```\n\n"
        f"**Review to analyze:**\n\n{text}\n\n**Your output (JSON only):**\n"
    )


# ---------------- data root ----------------
def check_data_root(root=None):
    """Refuse to mix datasets: $PE_DATA/DATASET names the dataset of a data root (no file = biorxiv, the old root)."""
    root = Path(root or os.environ.get("PE_DATA", "."))
    marker = root / "DATASET"
    have = marker.read_text().strip() if marker.exists() else "biorxiv"
    if have != DATASET:
        raise SystemExit(f"PE_DATASET={DATASET} but the data root {root} holds '{have}' data: point PE_DATA "
                         f"(NANO5_DATA on Nano5) at the {DATASET} root (openreview: bash openreview/nano5_setup.sh)")
