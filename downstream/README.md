# Downstream: next-token prediction on bioRxiv (BERT-NTP)

MAPLE's downstream task for bioRxiv (paper Fig. 3–4): train `prajjwal1/bert-small` as a causal language model on a
synthetic set, then measure next-token accuracy on the 2,000 real test abstracts.
`downstream/ntp.py` runs it for every arm; `downstream/summarize_ntp.py` makes the tables.

## What is the same as MAPLE, and what is added

| item | MAPLE (`biorxiv/utility_eval/`, transformers 4.55.4) | here |
|---|---|---|
| model | bert-small, `BertLMHeadModel`, `is_decoder = True` | the same |
| train set | survivors of the last PE round, text only | the same (`full`); also round 0, token-matched subsets, real texts, the round-0 pool |
| recipe | 512 tokens, 64 × 8 texts per update, 50 epochs (200 updates), AdamW lr 3e-4 constant, weight decay 0.01, no clipping, fp32, seed 42, last weights | the same; the loop is written out and tested against the transformers `Trainer` (same weights) |
| metric | correct next tokens / all tokens (`ntp_eval.py`), perplexity | the same formulas (`ntp_acc`, `ppl_maple`); also per predicted token (`ntp_acc_pred`, `loss`) |
| output | two numbers per run | plus one row per test text (`per_text.csv`) and the training-token count |

Three additions, each for one question:

1. **Per-text results.** Any split of the test set (tail / head classes, short / long texts, token position) is a sum
   over rows of `per_text.csv`. No new training is needed for the tail analysis.
2. **Training tokens and a token-matched variant.** Survivors of the t5 vote are longer than those of the ft vote
   (ε 4, round 4, seed 0: 363 vs 202 words on average), so 2,000 of them hold more training text. `full` is MAPLE's
   protocol. `tok<B>` trains every arm on a random subset of its texts with the same token total B (the smallest
   train set of the first call, kept in `ntp/token_budget.json`) and the same number of updates.
3. **Reference rows and noise.** `real` = 2,000 random private training texts (upper bound, not private);
   `pool` = 2,000 random texts of the round-0 pool before any vote. `TRAIN_SEEDS` repeats the training with other
   seeds (order of the texts, dropout): the spread is the noise of one number.

One difference from MAPLE's script: `is_decoder` is given when the model is loaded. MAPLE sets it on the config after
loading; with transformers 4.55.4 a batch in which every text reaches 512 tokens is then trained without the causal
mask. On bioRxiv this is rare (it needs a 16-text batch of only such texts), so the numbers stay comparable. Even
if 90 % of the texts reached 512 tokens, it would be about 10 of the 1,600 batches of a run.

## Run on Nano5 (login node, repo root)

```bash
cd ~/PE-metadata && git pull
bash downstream/nano5_ntp_setup.sh          # once: downloads bert-small (~120 MB), checks the env and the data

# 1. sanity first (1 GPU, 20 trainings): t5 vote, ft vote, Aug-PE, real texts; last round; MAPLE's protocol only
SOURCES="maple augpe real" RUNS_GLOB="maple/lam0* maple-ft/lam0* seed*" ROUNDS=last VARIANTS=full \
  sbatch --job-name=ntp-sanity --time=01:00:00 downstream/nano5_ntp.sbatch
cat /work/$USER/PE-metadata/ntp/NTP.md      # when it is done; see "What to check first" below

# 2. everything at eps 4: all arms, round 0 and last round, full + token-matched, real, pool (step 1 is not redone)
sbatch --gres=gpu:4 --cpus-per-task=32 --mem=192G downstream/nano5_ntp.sbatch

# 3. noise of one number: two more training seeds for two runs
SOURCES=maple RUNS_GLOB="maple/lam0 maple-ft/lam0" ROUNDS=last VARIANTS=full TRAIN_SEEDS="43 44" \
  sbatch --job-name=ntp-noise --time=00:30:00 downstream/nano5_ntp.sbatch

# 4. other eps (t5 and ft vote; no Aug-PE runs there: that source is skipped)
for e in 2.0 1.0 inf; do
  EPS=$e sbatch --job-name=ntp-eps$e --gres=gpu:2 --cpus-per-task=16 --mem=96G downstream/nano5_ntp.sbatch
done

# summary by hand (every job also writes it at its end)
module load miniconda3/24.11.1 && eval "$(conda shell.bash hook)" && conda activate /work/$USER/envs/maple
python downstream/summarize_ntp.py --root /work/$USER/PE-metadata/ntp \
  --test_ref /work/$USER/PE-metadata/quota_sweep/test_ref_qwen.csv

# results to copy back (small: no model weights; ~100 KB per training)
tar czf ~/ntp_results.tgz -C /work/$USER/PE-metadata ntp
```

Settings of the job (environment variables): `EPS`, `SOURCES` (maple, augpe, real, pool), `RUNS_GLOB`, `ROUNDS`
(`last`, `all`, numbers; `"0 4 last"` adds Aug-PE at MAPLE's last round), `VARIANTS` (`full`, `tok`, `len<L>`),
`TRAIN_SEEDS`, `SAMPLE_SEEDS`, `TOKEN_BUDGET`. They are listed at the top of `nano5_ntp.sbatch`.
One shard per GPU. A job that stops (time limit) is resumed by submitting the same line again: a training with a
`result.json` is skipped. Progress: `/work/$USER/logs/<job name>-<jobid>.out`, per shard
`/work/$USER/logs/ntp-<jobid>-shard<i>.log` (one line per training, with its seconds).

Time: not measured yet. A training is 1,600 batches of bert-small plus 2,000 test texts; the sanity job prints the
seconds per training, which gives the time of step 2 (about 270 trainings at ε 4).

## What to check first (step 1)

- **MAPLE's numbers.** `maple λ0` and `augpe` at ε 4 should lie in the range of MAPLE's Fig. 3 (next-token accuracy
  0.26–0.34) with MAPLE above Aug-PE. If not, stop and compare with MAPLE's own script before reading anything else.
- **`real` is the best row.** A synthetic set should not beat 2,000 real texts.
- **Train sets have 2,000 texts** (`train texts` column). Quota runs made before the survivor-shortfall fix
  (2026-11-19) have 1,924–1,999.
- **Training tokens** per arm (`train tokens` column, and `ntp/token_counts.csv` after step 2): how much longer the
  t5 arm is in bert-small tokens, and how many texts are cut at 512.

## Output

```
$PE_DATA/ntp/
  quota_sweep/eps4.0/<arm>/<run>/round<t>/<variant>-ts<seed>/{result.json, per_text.csv}
  augpe/eps4.0/seed<s>/round<t>/<variant>-ts<seed>/...
  real/n2000_s<k>/...      pool_eps4.0/n2000_s<k>/...
  token_counts.csv, token_budget.json
  NTP.md, ntp_runs.csv, ntp_settings.csv, ntp_paired.csv, ntp_groups.csv, ntp_noise.csv
```

- `result.json`: the train set (texts, tokens, texts at the 512 limit, mean words), the training (updates, epochs,
  first and final loss, seconds), the test metrics, versions and the git commit.
- `per_text.csv`: `idx` (row of the test CSV), `n_tokens` (with [CLS] and [SEP]), `n_correct`, `nll` (summed loss of
  the `n_tokens − 1` predicted tokens), `k0..k3` / `c0..c3` (predicted / correct tokens at positions 0–127,
  128–255, 256–383, 384–511).
- `ntp_settings.csv`: mean, sd, n over PE seeds per (tree, arm, λ, round, variant).
- `ntp_paired.csv`: differences to the t5 vote (`maple`, λ 0), paired by PE seed, with the 95% t-interval. Aug-PE is
  compared "last vs last" (each at its own last round) and at every round both have.
- `ntp_groups.csv`: the same inside groups of the test set:
  `category` (bioRxiv's own field, no annotator) and every `ref_<attribute>` (Qwen labels): tail / head / rare, where
  tail = the smallest classes summing to at most 20 % of the test set, each with at least 10 test texts (the rule
  of `figures/tail-counts-2026-11-18`); rare = small classes with fewer than 10 test texts;
  `length`: short / middle / long test texts (50-word bins; middle = the bins holding the bulk of the test set);
  `position`: position of the predicted token.
- `ntp_noise.csv`: sd over training seeds of one train set.

## How to read `full` against `tok<B>`

- `full`: what a user of the synthetic set gets (2,000 texts, as released). This is the number comparable with MAPLE.
- `tok<B>`: the same amount of training text for every arm. If an arm wins under `full` and loses or ties under
  `tok<B>`, its gain came from longer texts, not from better text.
- The `position` groups show where a gain sits: an arm trained on longer texts sees more tokens at positions above
  256, where few test tokens are (the real abstracts have about 220 words).

## Tests

`python -m pytest tests/test_ntp.py` (tiny random BERT, CPU, ~10 s; the comparison with the `Trainer` needs
`accelerate` and is skipped without it). They check: the model is causal and padding changes nothing; the evaluation
equals MAPLE's `ntp_eval.py` formulas; the training loop ends with the same weights as `transformers.Trainer` called
with MAPLE's arguments; token-matched subsets; run discovery, `last` round, shards, resuming; the summary's means,
paired differences and groups; the job script.
