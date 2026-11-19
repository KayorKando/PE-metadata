# OpenReview runs (second dataset)

The same pipeline as bioRxiv, switched by one environment variable: `PE_DATASET=openreview`. Every script reads it
(`blindness/schema.py`, `pe_data.py`); bioRxiv stays the default and its behaviour is unchanged.

## What is different from bioRxiv

| item | OpenReview |
|---|---|
| data | MAPLE `openreview/`: 8,396 private reviews, 2,798 val, 2,798 test, AIM metadata at ε 1 / 2 / 4 |
| schema | 12 attributes (`blindness/schema.py`): area, recommendation (MAPLE Listing 2) and 10 more; word_count measured |
| prompts | MAPLE `openreview/MAPLE/MAPLE.py` (8 metadata columns) and `PE.py` (Aug-PE) |
| vote | **class-conditional**: a private review votes only for candidates of its own `PE.LABEL_ID` (area × recommendation, 49 ids); K_g survivors per id, from the private label counts (MAPLE's rule; K_g = 7 … 198) |
| ε → (T, σ), MAPLE | 4: (4, 2.422), 2: (4, 4.415), 1: (2, 6.325), ∞: (9, 0). MAPLE runs T = 2 at ε 4 / 2; T = 4 with σ × √(5/3) keeps ρ_PE |
| ε → (T, σ), Aug-PE | 4: (9, 3.001), 2: (9, 5.419), 1: (9, 9.935) (MAPLE `main_experiment.sh`) |
| blind attributes | not fixed: worst 3 by k-NN κ (k = 5, within label id) on the ε = 4 round-0 pool, among the AIM attributes minus area / recommendation → `$NANO5_DATA/blind.json`, used at every ε |
| quota | inside each label id, target = AIM marginals of that id; loop labels from Qwen (`ANNOTATOR=qwen`) |
| ft vote | SciBERT (`allenai/scibert_scivocab_uncased`), normsoftmax, same recipe; ftt5 = sentence-t5 |
| annotator | Qwen with our prompt (`pe_data.openreview_prompt`; MAPLE ships none), guided JSON |
| metrics | as bioRxiv, plus `jsd_label_mean` (area, recommendation); these two are left out of `jsd_other_mean` |
| MAUVE | in the loop on sentence-t5 (as MAPLE); third-party: `Alibaba-NLP/gte-base-en-v1.5`, CLS, 2048 tokens |

Notes: the AIM files have no `PE.LABEL_ID`; it is mapped from (area, recommendation) and AIM rows with a pair the
private set lacks (73 at ε 4) are dropped (MAPLE never samples them). MAPLE's Aug-PE baseline (`PE.py`) puts no label
in the prompt; the label enters only through the per-id vote (Aug-PE's paper puts it in the prompt).

## Nano5 (login node, repo root)

```bash
cd ~/PE-metadata && git pull
hfsquota                                       # /work limit 100 GB; OpenReview adds ~5-10 GB
bash openreview/nano5_setup.sh                 # once: /work/$USER/PE-openreview, MAPLE/openreview, SciBERT, gte
python3 openreview/hf_cache_slim.py --delete  # frees ONNX / Flax copies in hf_cache (gte 1.6 GB, SciBERT 0.4 GB)

export PE_DATASET=openreview NANO5_DATA=/work/$USER/PE-openreview      # every command below needs these two

# 1. prep per eps: round-0 pool, Qwen labels, SciBERT ft embedder; the eps-4 job also makes the
#    Qwen-labelled test set and blind.json (so every PE job also waits for $P4)
P4=$(EPS=4.0 sbatch --parsable --job-name=or-prep-eps4.0 --time=04:00:00 quota_sweep/nano5_prep_eps.sbatch)
P2=$(EPS=2.0 sbatch --parsable --job-name=or-prep-eps2.0 --time=04:00:00 quota_sweep/nano5_prep_eps.sbatch)
P1=$(EPS=1.0 sbatch --parsable --job-name=or-prep-eps1.0 --time=04:00:00 quota_sweep/nano5_prep_eps.sbatch)
PI=$(EPS=inf sbatch --parsable --job-name=or-prep-epsinf --time=04:00:00 quota_sweep/nano5_prep_eps.sbatch)
TR=$(EPS=4.0 VARIANTS=ftt5 sbatch --parsable --job-name=or-vemb-train --time=02:00:00 --dependency=afterok:$P4 \
     vote_embedding/nano5_train.sbatch)
echo $P4 $P2 $P1 $PI $TR                       # keep these ids if you log out before step 2

# 2. PE runs (5 seeds = 1 job of 5 H100 per vote embedding x lambda)
S="0 1 2 3 4"
EPS=4.0 AFTER=$P4     GROUP_BY=lam VOTE_EMBS="t5 ft" LAMS=0   SEEDS="$S" TIME=02:30:00 bash quota_sweep/nano5_pe_submit.sh
EPS=4.0 AFTER=$P4:$TR GROUP_BY=lam VOTE_EMBS=ftt5    LAMS=0   SEEDS="$S" TIME=02:30:00 bash quota_sweep/nano5_pe_submit.sh
EPS=4.0 AFTER=$P4 ANNOTATOR=qwen GROUP_BY=lam VOTE_EMBS=t5 LAMS=0.5 SEEDS="$S" TIME=03:30:00 bash quota_sweep/nano5_pe_submit.sh
EPS=2.0 AFTER=$P4:$P2 GROUP_BY=lam VOTE_EMBS="t5 ft" LAMS=0   SEEDS="$S" TIME=02:30:00 bash quota_sweep/nano5_pe_submit.sh
EPS=1.0 AFTER=$P4:$P1 GROUP_BY=lam VOTE_EMBS="t5 ft" LAMS=0   SEEDS="$S" TIME=01:30:00 bash quota_sweep/nano5_pe_submit.sh
EPS=inf AFTER=$P4:$PI GROUP_BY=lam VOTE_EMBS="t5 ft" LAMS=0   SEEDS="$S" TIME=04:00:00 bash quota_sweep/nano5_pe_submit.sh
EPS=4.0 AFTER=$P4 SEEDS="$S" TIME=05:00:00 bash augpe/nano5_augpe_submit.sh                 # Aug-PE, T = 9

# 3. third-party MAUVE (gte) after the OpenReview jobs; afterany: the scorer is resumable and scores what finished
DEP=$(squeue -u $USER -h -o "%i %j" | awk '$2 ~ /^or-/ {print $1}' | paste -sd:)    # empty if all are done
GTE=Alibaba-NLP/gte-base-en-v1.5
for e in 4.0 2.0 1.0 inf; do
  EPS=$e MODEL=$GTE sbatch --job-name=or-mauve-eps$e ${DEP:+--dependency=afterany:$DEP} --gres=gpu:2 \
    --cpus-per-task=16 --mem=96G --time=03:00:00 quota_sweep/nano5_mauve.sbatch
done
MAUVE_ROOT=$NANO5_DATA/augpe/eps4.0 RUNS_GLOB='seed*' MODEL=$GTE sbatch --job-name=or-mauve-augpe \
  ${DEP:+--dependency=afterany:$DEP} --time=03:00:00 quota_sweep/nano5_mauve.sbatch

# 4. summaries (the jobs also write them; rerun after MAUVE). pe-metadata env has pandas / matplotlib
module load miniconda3/24.11.1 && eval "$(conda shell.bash hook)" && conda activate /work/$USER/envs/pe-metadata
for e in 4.0 2.0 1.0 inf; do python quota_sweep/summarize.py --root $NANO5_DATA/quota_sweep/eps$e; done
python augpe/summarize_augpe.py --augpe_root $NANO5_DATA/augpe/eps4.0 --maple_root $NANO5_DATA/quota_sweep/eps4.0/maple

unset PE_DATASET NANO5_DATA                    # back to bioRxiv in this shell
```

At most 2 of your jobs run at a time; the rest wait in the queue. A run that hits its time limit resumes after its
last finished round: rerun the same submit line (finished runs are skipped). Progress:
`tail -f /work/$USER/logs/or-pe4.0-t5-lam0-<jobid>.out`, per run `/work/$USER/logs/pe-<jobid>-maple_lam0.log`.

Outputs under `/work/$USER/PE-openreview/`: `round0/eps*/pool.csv`, `reannotate/eps*/pool_qwen.csv`,
`vote_embedding/eps*/normsoftmax[-t5]/`, `quota_sweep/test_ref_qwen.csv`, `blind.json` (+ `blind.csv`, κ per
attribute), `quota_sweep/eps*/{maple,maple-ft,maple-ftt5,qwen}/lam*/`, `augpe/eps4.0/seed*/`.

## Rough cost (H100)

Prep ≈ 2–2.5 h per ε (14k few-shot prompts of ~6.5k tokens, 14k reviews labelled, SciBERT). PE ≈ 10–12 min per
round (12k variations of ~400-word reviews): ε 4 / 2 ≈ 1 h per run, ε 1 ≈ 35 min, ε ∞ ≈ 2 h, quota (Qwen labels of
every candidate) ≈ 1.8 h, Aug-PE ≈ 2.3 h. Total ≈ 90 GPU-hours (55 PE runs, 4 preps, 1 embedder, MAUVE).
