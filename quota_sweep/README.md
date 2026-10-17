# Quota sweep: lambda sweep for the blind-attribute quota (Experiment 2)

MAPLE PE (bioRxiv, ε = 4 by default: T = 4 rounds, noise σ = 2.58, K = 2000, L = 6) with one change in the
selection step (`quota.py`), on the blind attributes B (default `word_count`, `sample_size`, `dominant_data_type`, from Experiment 1).

**Quota space = one-way marginals of B (raking, default `--cell_mode raking`).** For each blind attribute the target marginal is

$$m_k(v) \;\propto\; \hat P_k(v)^{\lambda}\, R_{t,k}(v)^{1-\lambda}$$

($\hat P_k$ from the AIM records, $R_{t,k}$ = marginal of MAPLE's top-K). Joint-cell weights start from MAPLE's top-K composition
(smoothed with 5% of the candidate composition) and are fitted to all three marginals by iterative proportional fitting; a cell that
runs out of candidates is frozen at its supply and the rest of that value's target is spread over the other cells (feasible quotas,
`capped_cells` in the log). Then largest-remainder rounding, argmax by vote inside each joint cell, water-filling for rounding leftovers.
λ = 0 is exactly MAPLE. No merged cells, no `__other__`.

The first version (`--cell_mode joint`: the 29 most frequent joint cells + `__other__`) put 47% of the AIM mass into `__other__`,
where the quota does nothing; it is kept only for reference.

Schedule `constant` (default): $\lambda_t = \lambda$; `linear`: $\lambda_t = \lambda\,t/T$.
No refill yet: `shortfall` now only counts rounding leftovers; whether the supply binds shows up as `capped_cells` and as the gap
between `marg_jsd_vs_phat_loop_labels` and 0 at λ = 1.

Round 0 reuses the pre-vote pool from `gen_round0.py`, so all λ runs start from the same candidates.

## Annotators (`annotators.py`, the choice is still open)

- `qwen`: local Qwen (shares the generator's vLLM engine), MAPLE prompt, guided JSON, greedy. **Ground truth for now.**
- `inherited`: requested AIM metadata carried by each row (MAPLE's labels).
- `classifier`: the best per-attribute classifier from `reannotate/` (`reannotate/select_annotator.py` → `best_annotator.json`, pass it with `--classifier_spec`). It predicts the quota's blind content attributes; the other attributes keep inherited labels (the quota does not use them).

`--loop_annotator` = labels used by the quota; evaluation always uses `--eval_annotator` (qwen). `word_count` is always measured.

## Measured every round on the K survivors (`metrics.jsonl`)

- `mauve`: MAPLE's MAUVE (sentence-t5 features) vs the 2,000-text test set
- `jsd_<attr>`: JSD vs the test set labelled by the same annotator (Qwen; length measured); `jsd_gemini_<attr>`: vs MAPLE's Gemini labels of the test set
- `jsd_mean` (9 attributes), `jsd_blind_mean` (B), `jsd_other_mean` (the rest)
- `marg_jsd_vs_phat_{loop,eval}_labels`: mean over B of JSD(survivors' marginal, $\hat P_k$), with the in-loop and the evaluation labels
- `moved` (survivors that differ from MAPLE's top-K), `shortfall`, `lam_t`

## Run (weekend)

```bash
tmux new -d -s sweep "export PE_DATA=$PE_DATA HF_HOME=$HF_HOME; bash ~/PE-metadata/quota_sweep/run_lambda_sweep.sh"
tail -f $PE_DATA/quota_sweep/eps4.0/logs/*.log
cat $PE_DATA/quota_sweep/eps4.0/SWEEP.md          # updated after every finished run; frontier.png, curves.png
```

Options: `LAMS="0 0.25 0.5 0.75 1"`, `ANNOTATORS="qwen inherited"`, `SEEDS="0 1"`, `SCHEDULE=linear`, `EPS=1.0`
(needs that ε's round-0 pool), `GPUS=1,2`. One run per GPU at a time; finished runs are skipped, unfinished ones resume.

## Re-annotating + sweep in one 24-hour job (3 GPUs)

`run_reannotate_then_quota_sweep.sh` (repo root): GPUs A, B run the re-annotating experiment (Qwen labels → 7 classifiers with weights saved → report → best classifier per blind attribute), while GPU C prepares the test reference and starts sweep jobs that need no classifier. Then all three GPUs take sweep jobs from one priority list: MAPLE baseline (λ = 0, run once and shared by every annotator), classifier λ ∈ {1, 0.5, 0.75, 0.25}, qwen λ ∈ {1, 0.5, 0.75, 0.25}, inherited λ ∈ {1, 0.5}, a second seed for the baseline and classifier λ ∈ {1, 0.5} and qwen λ = 1, then the remaining inherited and classifier seed-1 points, all with constant λ. Only after these, if time is left: the linear schedule for classifier λ ∈ {1, 0.5, 0.75, 0.25}, qwen λ ∈ {1, 0.5}, inherited λ = 1 (results under `<annotator>-linear/`). No new job starts in the last `MARGIN` (3) hours.

```bash
tmux new -d -s all "export PE_DATA=$PE_DATA HF_HOME=$HF_HOME; bash ~/PE-metadata/run_reannotate_then_quota_sweep.sh"
tail -f $PE_DATA/pipeline/logs/*.log
```
