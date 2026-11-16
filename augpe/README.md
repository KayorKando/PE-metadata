# Aug-PE drift (Experiment: is drift caused by the vote or by MAPLE's metadata prompts?)

Claim to test: metadata drift comes from voting in one embedding (sentence-t5, top-K), not from MAPLE's
metadata prompts. Aug-PE uses no metadata anywhere, so if the same blind attributes (`word_count`, `sample_size`,
`dominant_data_type`) drift here, the cause is the vote.

**Algorithm** = MAPLE's own Aug-PE baseline (MAPLE repo `biorxiv/MAPLE/PE.py` + `main_experiment.sh`); the
prompts are imported from that file, nothing is rewritten:

| item | value |
|---|---|
| RANDOM_API | one fixed prompt, no metadata, no in-context examples; K(L+1) = 14,000 texts |
| VARIATION_API | MAPLE's Aug-PE variation prompt (no metadata, no length request), L = 6 per survivor |
| vote | sentence-t5-base, max_seq 1024, 1-NN (L2) of every private text, + N(0, σ²); top-K = 2,000 |
| ε = 4 | T = 9 rounds, σ = 3.2144 (GDP calibration, as in MAPLE's paper). ε 2 / 1 / ∞: σ = 5.872 / 10.890 / 0, T = 9 |
| LLM | Qwen2.5-7B-Instruct, temperature 1.0, top-p 0.95, max 1024 tokens |
| seeds | 0–4; each seed makes its own round-0 pool (no shared AIM pool exists without metadata) |

**Measurements** (same definitions as `quota_sweep/quota_pe.py`, so numbers line up with the MAPLE runs):
Qwen labels of the K survivors every round; `jsd_<attr>` vs the Qwen-labelled test set (`word_count` measured);
`jsd_gemini_<attr>`; `jsd_blind_mean`, `jsd_other_mean`; MAUVE (sentence-t5). Extra: the **LLM prior** =
2,000 random texts of the round-0 pool before any vote (`prior_metrics.json`, round −1 in the summary), and
`jsd_prior_<attr>` = JSD(survivors, prior) per round. Drift toward the LLM prior shows as `jsd_prior` falling
while `jsd` (vs real data) rises.

## What each result would mean (written before the run)

- **Blind attributes rise or stay flat over rounds while the others fall** → drift is a property of the vote,
  not of MAPLE's prompts. The claim becomes about PE in general.
- **Nothing drifts in Aug-PE** (blind JSD flat and close to the prior) → the drift in MAPLE is "a good start
  (AIM metadata) eroded by the vote"; the paper's claim is narrower.
- **word_count drifts but content attributes do not** → the length effect is from the variation prompt (Aug-PE's
  has no length request), check the length distribution per round before reading it as a vote effect.
- Aug-PE runs 9 rounds and MAPLE 4: compare at round 4 as well as at the last round (both are in `DRIFT.md`).

## Run on Nano5 (login node, repo root)

Needs what the MAPLE runs already use: `$PE_DATA/MAPLE` (with `biorxiv/MAPLE/PE.py`), the `maple` conda env,
`quota_sweep/test_ref_qwen.csv`, and the sentence-t5 embedding cache of the private set.

```bash
git pull
bash augpe/nano5_augpe_submit.sh                  # eps 4.0, seeds 0-4 -> 1 job, 5 H100, TIME 3 h
# third-party MAUVE (gpt2-large) on every round, after the PE job (1 GPU is enough: 5 runs x 10 rounds)
MAUVE_ROOT=/work/$USER/PE-metadata/augpe/eps4.0 RUNS_GLOB='seed*' \
  sbatch --dependency=afterok:<augpe job> quota_sweep/nano5_mauve.sbatch
# summary (the PE job also writes it at its end, without gpt2-large MAUVE)
python augpe/summarize_augpe.py --augpe_root /work/$USER/PE-metadata/augpe/eps4.0 \
  --maple_root /work/$USER/PE-metadata/quota_sweep/eps4.0/maple
```

Progress: `tail -f /work/$USER/logs/augpe-eps4.0-<jobid>.out`, per seed `/work/$USER/logs/augpe-<jobid>-seed<s>.log`.
Expected time per seed: round-0 pool (14k texts) + 10 rounds of ~6–7 min ≈ 75–90 min on one H100.

## Outputs

Per seed in `$PE_DATA/augpe/eps4.0/seed<s>/`: `pool.csv`, `prior.csv`, `prior_metrics.json`, `round_<t>.csv`,
`metrics.jsonl`, `qwen_cache.csv`, `config.json`, `DONE`.
Summary (`summarize_augpe.py`): `DRIFT.md`, `drift_curves.png`, `drift_attrs.png`, and the data behind them:
`drift_runs.csv` (every run x round), `drift_rounds.csv` (mean / sd / n), `drift_change.csv` (round 0 → last,
paired by seed, 95% t-interval).
