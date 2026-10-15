#!/usr/bin/env bash
# Experiment 2: sweep lambda for the blind-attribute quota (MAUVE vs metadata JSD).
#
#   tmux new -d -s sweep "export PE_DATA=$PE_DATA HF_HOME=$HF_HOME; bash ~/PE-metadata/experiment2/run_lambda_sweep.sh"
#   tail -f $PE_DATA/exp2/eps4.0/logs/*.log
#   cat $PE_DATA/exp2/eps4.0/SWEEP.md
#
# Settings (environment variables):
#   LAMS        lambda values             (default "0 0.25 0.5 0.75 1"; 0 = MAPLE)
#   ANNOTATORS  in-loop label sources     (default "qwen"; also "inherited")
#   SEEDS       seeds                     (default "0")
#   SCHEDULE    linear | constant         (default linear: lambda_t = lambda * t / T)
#   EPS         4.0 | 2.0 | 1.0           (default 4.0; needs $PE_DATA/round0/eps$EPS/pool.csv)
#   GPUS        one run per GPU at a time (default 1,2)
#   EXTRA       extra flags for quota_pe.py
# Finished runs (DONE) are skipped and unfinished ones resume after their last round.
set -euo pipefail
LAMS=${LAMS:-"0 0.25 0.5 0.75 1"}
ANNOTATORS=${ANNOTATORS:-qwen}
SEEDS=${SEEDS:-0}
SCHEDULE=${SCHEDULE:-linear}
EPS=${EPS:-4.0}
GPUS=${GPUS:-1,2}
EXTRA=${EXTRA:-}
: "${PE_DATA:?set PE_DATA, e.g. export PE_DATA=/data/\$USER/PE-metadata}"
export HF_HOME=${HF_HOME:-/data/$USER/hf_cache}
MAPLE_ENV=${MAPLE_ENV:-/data/$USER/envs/maple}
PE_ENV=${PE_ENV:-/data/$USER/envs/pe-metadata}
REPO=$(cd "$(dirname "$0")/.." && pwd)
POOL=$PE_DATA/round0/eps$EPS/pool.csv
ROOT=$PE_DATA/exp2/eps$EPS
LOG=$ROOT/logs
TEST_REF=$PE_DATA/exp2/test_ref_qwen.csv
mkdir -p "$LOG"
IFS=, read -ra G <<< "$GPUS"
MPY=$MAPLE_ENV/bin/python
stamp() { echo "[$(date '+%F %T')] $*"; }
cd "$REPO"
stamp "repo $(git rev-parse --short HEAD 2>/dev/null || echo '?'), outputs -> $ROOT"

# --- 0. checks ---
[ -s "$POOL" ] || { echo "missing $POOL: run run_round0.sh first (EPS=$EPS)"; exit 1; }
"$MPY" -c "import faiss" 2>/dev/null || { stamp "installing faiss-cpu (MAUVE) into $MAPLE_ENV"; "$MAPLE_ENV/bin/pip" install -q faiss-cpu; }
"$MPY" -m tests.test_quota

# --- 1. test-set reference labels (once, shared by every run) ---
if [ ! -s "$TEST_REF" ]; then
  stamp "1. Qwen labels for the test set -> $TEST_REF"
  CUDA_VISIBLE_DEVICES=${G[0]} "$MPY" experiment2/quota_pe.py --maple_dir "$PE_DATA/MAPLE" --pool_csv "$POOL" \
    --eps "$EPS" --lam 0 --test_ref "$TEST_REF" --out "$ROOT/_prep" --prep_only > "$LOG/1_test_ref.log" 2>&1
fi

# --- 2. jobs, round-robin over GPUs; each GPU runs its jobs one after another ---
jobs=()
for ann in $ANNOTATORS; do for seed in $SEEDS; do for lam in $LAMS; do jobs+=("$ann $seed $lam"); done; done; done
stamp "2. ${#jobs[@]} runs on GPUs ${GPUS}"
worker() {  # gpu index
  local gi=$1 j=0
  for job in "${jobs[@]}"; do
    if [ $((j % ${#G[@]})) -eq "$gi" ]; then
      read -r ann seed lam <<< "$job"
      name="lam${lam}"; [ "$seed" = 0 ] || name="${name}_seed${seed}"
      out="$ROOT/$ann/$name"
      if [ -f "$out/DONE" ]; then stamp "skip $ann/$name (done)"; else
        stamp "run $ann/$name on GPU ${G[$gi]}"
        CUDA_VISIBLE_DEVICES=${G[$gi]} "$MPY" experiment2/quota_pe.py --maple_dir "$PE_DATA/MAPLE" --pool_csv "$POOL" \
          --eps "$EPS" --lam "$lam" --schedule "$SCHEDULE" --loop_annotator "$ann" --seed "$seed" \
          --test_ref "$TEST_REF" --out "$out" $EXTRA > "$LOG/${ann}_${name}.log" 2>&1 \
          && stamp "done $ann/$name" || stamp "FAILED $ann/$name, see $LOG/${ann}_${name}.log"
        "$PE_ENV/bin/python" experiment2/summarize.py --root "$ROOT" > /dev/null 2>&1 || true   # keep SWEEP.md current
      fi
    fi
    j=$((j + 1))
  done
}
pids=()
for gi in "${!G[@]}"; do worker "$gi" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done

# --- 3. summary ---
"$PE_ENV/bin/python" experiment2/summarize.py --root "$ROOT" > "$LOG/3_summary.log" 2>&1 || true
stamp "done. cat $ROOT/SWEEP.md  (figures: frontier.png, curves.png)"
