#!/usr/bin/env bash
# Submit the vanilla Aug-PE drift runs on NCHC Nano5: one job, one H100 per seed, seeds in parallel.
# Run on the login node from the repo root. It only checks inputs and calls sbatch once.
#   bash augpe/nano5_augpe_submit.sh                          # eps 4.0, seeds 0-4 -> 1 job, 5 GPUs
#   SEEDS="0 1" EPS=2.0 AFTER=<jobid> bash augpe/nano5_augpe_submit.sh
# Settings: SEEDS ("0 1 2 3 4"), EPS (4.0), TIME (03:00:00: T = 9 rounds + round-0 pool, ~75-90 min expected),
# CPUS_PER_RUN (8; Nano5 allows at most 12 per GPU), MEM_PER_RUN (64, GB), AFTER (job id), PARTITION (normal),
# EXTRA (augpe_pe.py flags). Finished seeds (DONE) are left out; unfinished ones resume after their last round.
# OpenReview: export PE_DATASET=openreview first (see openreview/README.md); TIME=05:00:00 there (longer texts).
set -eo pipefail
SEEDS=${SEEDS:-0 1 2 3 4}
EPS=${EPS:-4.0}
TIME=${TIME:-03:00:00}
CPUS_PER_RUN=${CPUS_PER_RUN:-8}
MEM_PER_RUN=${MEM_PER_RUN:-64}
AFTER=${AFTER:-}
PARTITION=${PARTITION:-normal}
[ -f augpe/nano5_augpe.sbatch ] || { echo "run from the repo root"; exit 1; }
source quota_sweep/nano5_lib.sh
export PE_DATASET=${PE_DATASET:-biorxiv}
PE_DATA=$(nano5_data)
[ "$CPUS_PER_RUN" -le 12 ] || { echo "CPUS_PER_RUN $CPUS_PER_RUN > 12 (Nano5 limit per GPU)"; exit 1; }
need="quota_sweep/test_ref_qwen.csv $(private_csv) MAPLE/$(maple_subdir)/MAPLE/PE.py"
[ "$PE_DATASET" = biorxiv ] || need="$need blind.json"
for f in $need; do
  # with AFTER, the eps-4 prep job may still be making the test reference and blind.json (OpenReview)
  [ -n "$AFTER" ] && [ "$PE_DATASET" != biorxiv ] && [[ $f == quota_sweep/test_ref_qwen.csv || $f == blind.json ]] && continue
  [ -s "$PE_DATA/$f" ] || { echo "missing $PE_DATA/$f (see quota_sweep/nano5_pe_setup.sh; OpenReview: openreview/README.md)"; exit 1; }
done
ROOT=$PE_DATA/augpe/eps$EPS
todo=()
for s in $SEEDS; do
  if [ -f "$ROOT/seed$s/DONE" ]; then echo "  done already: seed$s"; else todo+=("$s"); fi
done
[ ${#todo[@]} -gt 0 ] || { echo "nothing to submit: all seeds are done"; exit 0; }
n=${#todo[@]}
[ "$n" -le 8 ] || { echo "$n seeds would need $n GPUs (> 8 per node): submit fewer SEEDS at a time"; exit 1; }
dep=(); [ -z "$AFTER" ] || dep=(--dependency=afterok:"$AFTER")
export EPS EXTRA=${EXTRA:-} NANO5_DATA=$PE_DATA PE_DATASET SEEDS="${todo[*]}"   # sbatch passes the environment on
echo "job augpe-eps$EPS ($PE_DATASET, $PE_DATA): seeds ${todo[*]} on $n GPU(s), limit $TIME${AFTER:+, after job $AFTER}"
sbatch --job-name="$(job_prefix)augpe-eps$EPS" --gres=gpu:"$n" --cpus-per-task=$((CPUS_PER_RUN * n)) --mem=$((MEM_PER_RUN * n))G \
  --time="$TIME" --partition="$PARTITION" "${dep[@]}" augpe/nano5_augpe.sbatch
