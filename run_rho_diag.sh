#!/usr/bin/env bash
# Why is rho(k-NN, probe) lower on the round-0 pool than on private data?
# Runs the three diagnostics and collects every output in one folder.
#
#   tmux new -d -s rhodiag "bash ~/PE-metadata/run_rho_diag.sh"
#   tail -f $PE_DATA/results/rho_diag_eps4.0/*.log      # progress
#
# Settings (environment variables): EPS (default 4.0), GPUS (default 1,2; first two used),
# PE_ENV (venv), PRIVATE_RES (summary.csv folder of the full-n private run, default results/n0),
# POOL_RES (summary.csv folder of the earlier pool run, default results/round0_eps$EPS).
set -euo pipefail

EPS=${EPS:-4.0}
GPUS=${GPUS:-1,2}
: "${PE_DATA:?set PE_DATA, e.g. export PE_DATA=/data/\$USER/PE-metadata}"
export HF_HOME=${HF_HOME:-/data/$USER/hf_cache}
PE_ENV=${PE_ENV:-/data/$USER/envs/pe-metadata}
PY=$PE_ENV/bin/python
REPO=$(cd "$(dirname "$0")" && pwd)

POOL=$PE_DATA/round0/eps$EPS/pool.csv
PRIVATE_CSV=$PE_DATA/MAPLE/biorxiv/biorxiv_train_metadata.csv
PRIVATE_RES=${PRIVATE_RES:-$PE_DATA/results/n0}
POOL_RES=${POOL_RES:-$PE_DATA/results/round0_eps$EPS}
OUT=$PE_DATA/results/rho_diag_eps$EPS
mkdir -p "$OUT"

IFS=, read -ra G <<< "$GPUS"
G1=${G[0]}; G2=${G[1]:-${G[0]}}
ATTRS=(primary_research_area model_organism experimental_approach dominant_data_type
       research_focus_scale disease_mention sample_size research_goal word_count word_count_requested)
LEN=(word_count word_count_requested)
stamp() { echo "[$(date '+%F %T')] $*"; }

# --- 0. checks ---
[ -x "$PY" ] || { echo "missing $PY (Experiment 1 env)"; exit 1; }
for f in "$POOL" "$PRIVATE_CSV" "$PRIVATE_RES/summary.csv" "$POOL_RES/summary.csv"; do
  [ -f "$f" ] || { echo "missing $f  (set PRIVATE_RES / POOL_RES if your results live elsewhere)"; exit 1; }
done
cd "$REPO"
stamp "repo $(git rev-parse --short HEAD 2>/dev/null || echo '?'), outputs -> $OUT"

# --- 1. is 0.95 vs 0.64 a real difference? (CPU, seconds) ---
stamp "1. rho statistics, private vs pool (random split)"
"$PY" compare_rho.py --a "$PRIVATE_RES/summary.csv" --b "$POOL_RES/summary.csv" \
    --label_a private --label_b pool 2>&1 | tee "$OUT/1_compare_rho_random.log"

# --- 2. twins: random split + twin_rate (GPU G1) and group split (GPU G2), in parallel ---
common=(--csv "$POOL" --n 0 --repeats 5 --attributes "${ATTRS[@]}" --batch_size 256
        --group_cols requested --exclude_from_rho "${LEN[@]}")
stamp "2. twin_rate on GPU $G1, group split on GPU $G2 (in parallel)"
CUDA_VISIBLE_DEVICES=$G1 "$PY" run.py "${common[@]}" --out "$OUT/twinrate" > "$OUT/2a_twinrate.log" 2>&1 &
PA=$!
CUDA_VISIBLE_DEVICES=$G2 "$PY" run.py "${common[@]}" --group_split --out "$OUT/groupsplit" > "$OUT/2b_groupsplit.log" 2>&1 &
PB=$!

# --- 3. label correlation between attributes (CPU, runs while the GPUs work) ---
stamp "3. pairwise NMI of attribute labels"
"$PY" label_mi.py --private "$PRIVATE_CSV" --pool "$POOL" --out "$OUT/label_mi" 2>&1 | tee "$OUT/3_label_mi.log"

fail=0
wait $PA || { echo "twin_rate run failed, see $OUT/2a_twinrate.log"; fail=1; }
wait $PB || { echo "group-split run failed, see $OUT/2b_groupsplit.log"; fail=1; }
[ $fail -eq 0 ] || exit 1
stamp "2. done"
tail -n 25 "$OUT/2a_twinrate.log"
tail -n 25 "$OUT/2b_groupsplit.log"

stamp "2c. rho statistics, private vs pool (group split)"
"$PY" compare_rho.py --a "$PRIVATE_RES/summary.csv" --b "$OUT/groupsplit/summary.csv" \
    --label_a private --label_b pool_groupsplit 2>&1 | tee "$OUT/2c_compare_rho_groupsplit.log"

# --- one file to paste back ---
{
  for f in 1_compare_rho_random 2a_twinrate 2b_groupsplit 2c_compare_rho_groupsplit 3_label_mi; do
    echo "===== $f ====="
    case $f in 2a_*|2b_*) grep -v -E "Batches|it/s\]|%\|" "$OUT/$f.log" | tail -n 25 ;; *) cat "$OUT/$f.log" ;; esac
    echo
  done
} > "$OUT/ALL.txt"
stamp "done. paste $OUT/ALL.txt   (plots: $OUT/{twinrate,groupsplit}/knn_vs_probe.png, $OUT/label_mi/nmi.png)"
