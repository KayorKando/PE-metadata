#!/usr/bin/env bash
# Re-annotating experiment, one command:
#   1. round-0 pool (Qwen + AIM metadata)       -> reuses $PE_DATA/round0/eps$EPS/pool.csv, else runs run_round0.sh
#   2. Qwen as annotator (vLLM, MAPLE prompt)   -> pool on GPU A, a private sample on GPU B, in parallel
#   3. one classifier per attribute: TF-IDF+LR (CPU); SciBERT, phi-MLP (GPU A); BiomedBERT, frozen-BiomedBERT MLP (GPU B); in parallel
#   4. report                                    -> $OUT/results/REPORT.md
#
#   tmux new -d -s reann "bash ~/PE-metadata/reannotate/run_reannotate.sh"
#   tail -f $PE_DATA/reannotate/logs/*.log
#
# Settings: EPS (4.0), GPUS (1,2), PRIVATE_SAMPLE (3000), QWEN (Qwen/Qwen2.5-7B-Instruct),
# MAPLE_ENV, PE_ENV. Finished steps are skipped on a rerun.
set -euo pipefail
EPS=${EPS:-4.0}
GPUS=${GPUS:-1,2}
PRIVATE_SAMPLE=${PRIVATE_SAMPLE:-3000}
QWEN=${QWEN:-Qwen/Qwen2.5-7B-Instruct}
: "${PE_DATA:?set PE_DATA, e.g. export PE_DATA=/data/\$USER/PE-metadata}"
export HF_HOME=${HF_HOME:-/data/$USER/hf_cache}
MAPLE_ENV=${MAPLE_ENV:-/data/$USER/envs/maple}
PE_ENV=${PE_ENV:-/data/$USER/envs/pe-metadata}
REPO=$(cd "$(dirname "$0")/.." && pwd)
POOL=$PE_DATA/round0/eps$EPS/pool.csv
PRIVATE_CSV=$PE_DATA/MAPLE/biorxiv/biorxiv_train_metadata.csv
OUT=$PE_DATA/reannotate/eps$EPS
LOG=$OUT/logs
RES=$OUT/results
mkdir -p "$LOG" "$RES"
IFS=, read -ra G <<< "$GPUS"; GA=${G[0]}; GB=${G[1]:-${G[0]}}
stamp() { echo "[$(date '+%F %T')] $*"; }
cd "$REPO"
stamp "repo $(git rev-parse --short HEAD 2>/dev/null || echo '?'), outputs -> $OUT"

# 1. round-0 pool
if [ ! -s "$POOL" ]; then
  stamp "1. no pool yet: running run_round0.sh"
  EPS=$EPS GPUS=$GPUS bash run_round0.sh
else
  stamp "1. pool exists: $POOL"
fi

# 2. Qwen annotation (pool on GA, private sample on GB)
pids=()
if [ ! -s "$OUT/pool_qwen.csv" ]; then
  stamp "2a. Qwen labels the pool on GPU $GA"
  CUDA_VISIBLE_DEVICES=$GA "$MAPLE_ENV/bin/python" reannotate/annotate_qwen.py --csv "$POOL" \
    --maple_dir "$PE_DATA/MAPLE" --model "$QWEN" --out "$OUT/pool_qwen.csv" > "$LOG/2a_annotate_pool.log" 2>&1 &
  pids+=($!)
fi
if [ ! -s "$OUT/private_qwen.csv" ]; then
  stamp "2b. Qwen labels $PRIVATE_SAMPLE private texts on GPU $GB"
  CUDA_VISIBLE_DEVICES=$GB "$MAPLE_ENV/bin/python" reannotate/annotate_qwen.py --csv "$PRIVATE_CSV" \
    --maple_dir "$PE_DATA/MAPLE" --model "$QWEN" --sample "$PRIVATE_SAMPLE" --out "$OUT/private_qwen.csv" \
    > "$LOG/2b_annotate_private.log" 2>&1 &
  pids+=($!)
fi
for p in "${pids[@]:-}"; do [ -z "$p" ] || wait "$p" || { stamp "annotation failed, see $LOG/2*.log"; exit 1; }; done
stamp "2. done"

# 3. classifiers
common=(--pool_qwen "$OUT/pool_qwen.csv" --private_csv "$PRIVATE_CSV" --private_qwen "$OUT/private_qwen.csv" --out "$RES")
pids=()
run_clf() {  # model gpu
  if [ -s "$RES/$1/metrics.csv" ]; then stamp "3. $1 already done"; return; fi
  stamp "3. $1 (GPU ${2:-none}) -> $LOG/3_$1.log"
  CUDA_VISIBLE_DEVICES=${2:-} "$PE_ENV/bin/python" reannotate/train_classifier.py --model "$1" "${common[@]}" \
    > "$LOG/3_$1.log" 2>&1 &
  pids+=($!)
}
run_clf tfidf ""
run_clf scibert "$GA"
run_clf biomedbert "$GB"
run_clf phi_mlp "$GA"      # MLP on the vote embedding (sentence-t5, cached)
run_clf biomed_mlp "$GB"   # MLP on frozen BiomedBERT (mean-pooled)
fail=0
for p in "${pids[@]:-}"; do [ -z "$p" ] || wait "$p" || fail=1; done
[ $fail -eq 0 ] || stamp "a classifier failed, see $LOG/3_*.log (the report uses the ones that finished)"

# 4. report
"$PE_ENV/bin/python" reannotate/report.py --results "$RES" --pool_qwen "$OUT/pool_qwen.csv" \
  --private_csv "$PRIVATE_CSV" --private_qwen "$OUT/private_qwen.csv" > "$LOG/4_report.log" 2>&1
stamp "done. paste $RES/REPORT.md"
