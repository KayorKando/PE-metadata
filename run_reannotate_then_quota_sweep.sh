#!/usr/bin/env bash
# One 24-hour job on 3 GPUs: the re-annotating experiment, then the quota lambda sweep that uses
# its best classifier as the in-loop annotator.
#
#   tmux new -d -s all "export PE_DATA=$PE_DATA HF_HOME=$HF_HOME; bash ~/PE-metadata/run_reannotate_then_quota_sweep.sh"
#   tail -f $PE_DATA/pipeline/logs/*.log
#
# Timeline
#   GPU A, B : re-annotating = Qwen labels (pool, private sample) -> 7 classifiers (weights saved)
#              -> report -> best classifier per blind attribute -> then join the sweep
#   GPU C    : test-set reference labels -> starts the sweep at once with the jobs that do not need
#              the classifier (MAPLE baseline, qwen / inherited loop labels)
#   CPU      : TF-IDF
# Sweep jobs are taken from one priority list by whichever GPU is free; classifier jobs go first
# as soon as best_annotator.json exists. No new job starts after HOURS - MARGIN hours; a job cut
# off by the deadline resumes on the next invocation (finished ones are skipped).
#
# Settings: GPUS (default 0,1,2: A,B,C), HOURS (24), MARGIN (3), EPS (4.0), SCHEDULE (linear),
#           PRIVATE_SAMPLE (3000), MAPLE_ENV, PE_ENV.
set -uo pipefail
GPUS=${GPUS:-0,1,2}
HOURS=${HOURS:-24}
MARGIN=${MARGIN:-3}
EPS=${EPS:-4.0}
SCHEDULE=${SCHEDULE:-linear}
PRIVATE_SAMPLE=${PRIVATE_SAMPLE:-3000}
: "${PE_DATA:?set PE_DATA, e.g. export PE_DATA=/data/\$USER/PE-metadata}"
USER=${USER:-$(id -un)}
export HF_HOME=${HF_HOME:-/data/$USER/hf_cache}
MAPLE_ENV=${MAPLE_ENV:-/data/$USER/envs/maple}
PE_ENV=${PE_ENV:-/data/$USER/envs/pe-metadata}
MPY=$MAPLE_ENV/bin/python
PPY=$PE_ENV/bin/python
REPO=$(cd "$(dirname "$0")" && pwd)
POOL=$PE_DATA/round0/eps$EPS/pool.csv
PRIVATE_CSV=$PE_DATA/MAPLE/biorxiv/biorxiv_train_metadata.csv
REANN=$PE_DATA/reannotate/eps$EPS
RES=$REANN/results
SPEC=$REANN/best_annotator.json
ROOT=$PE_DATA/quota_sweep/eps$EPS
TEST_REF=$PE_DATA/quota_sweep/test_ref_qwen.csv
LOG=$PE_DATA/pipeline/logs
CLAIMS=$PE_DATA/pipeline/claims
BLIND_CONTENT="sample_size dominant_data_type"     # word_count is measured, needs no classifier
mkdir -p "$LOG" "$RES" "$ROOT"
rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"               # stale claims from an earlier invocation
IFS=, read -ra G <<< "$GPUS"; GA=${G[0]}; GB=${G[1]}; GC=${G[2]}
START=$(date +%s)
stamp() { echo "[$(date '+%F %T')] $*"; }
hours_left() { echo $(( HOURS * 3600 - ($(date +%s) - START) )); }
cd "$REPO"
stamp "repo $(git rev-parse --short HEAD 2>/dev/null || echo '?'), GPUs A=$GA B=$GB C=$GC, budget ${HOURS}h"

# ---------------- 0. checks ----------------
[ -s "$POOL" ] || { echo "missing $POOL: run run_round0.sh first"; exit 1; }
"$MPY" -c "import faiss" 2>/dev/null || { stamp "installing faiss-cpu into $MAPLE_ENV"; "$MAPLE_ENV/bin/pip" install -q faiss-cpu; }
"$MPY" -m tests.test_quota || exit 1

# ---------------- A. re-annotating (GPU A, B) ----------------
phase_a() {
  touch "$REANN/phaseA.running"
  local ok=1
  [ -s "$REANN/pool_qwen.csv" ] || CUDA_VISIBLE_DEVICES=$GA "$MPY" reannotate/annotate_qwen.py --csv "$POOL" \
      --maple_dir "$PE_DATA/MAPLE" --out "$REANN/pool_qwen.csv" > "$LOG/A1_annotate_pool.log" 2>&1 &
  local p1=$!
  [ -s "$REANN/private_qwen.csv" ] || CUDA_VISIBLE_DEVICES=$GB "$MPY" reannotate/annotate_qwen.py --csv "$PRIVATE_CSV" \
      --maple_dir "$PE_DATA/MAPLE" --sample "$PRIVATE_SAMPLE" --out "$REANN/private_qwen.csv" > "$LOG/A2_annotate_private.log" 2>&1 &
  local p2=$!
  wait $p1 || ok=0; wait $p2 || ok=0
  if [ $ok = 0 ] || [ ! -s "$REANN/pool_qwen.csv" ]; then stamp "A. annotation FAILED (see $LOG/A1*, A2*)"; rm -f "$REANN/phaseA.running"; touch "$REANN/phaseA.failed"; return; fi
  stamp "A. Qwen annotation done; training classifiers"
  local common=(--pool_qwen "$REANN/pool_qwen.csv" --private_csv "$PRIVATE_CSV" --private_qwen "$REANN/private_qwen.csv"
                --out "$RES" --save_models)
  clf() {  # model gpu
    [ -s "$RES/$1/metrics.csv" ] && [ -d "$RES/$1/models" ] && return 0
    CUDA_VISIBLE_DEVICES=${2:-} "$PPY" reannotate/train_classifier.py --model "$1" "${common[@]}" > "$LOG/A3_$1.log" 2>&1 \
      || stamp "A. classifier $1 FAILED (see $LOG/A3_$1.log)"
  }
  ( clf tfidf "" ) &
  ( clf phi_mh "$GA"; clf phi_mlp "$GA"; clf scibert "$GA" ) &
  ( clf biomed_mh "$GB"; clf biomed_mlp "$GB"; clf biomedbert "$GB" ) &
  wait
  "$PPY" reannotate/report.py --results "$RES" --pool_qwen "$REANN/pool_qwen.csv" --private_csv "$PRIVATE_CSV" \
      --private_qwen "$REANN/private_qwen.csv" > "$LOG/A4_report.log" 2>&1
  "$PPY" reannotate/select_annotator.py --results "$RES" --attributes $BLIND_CONTENT --out "$SPEC" \
      > "$LOG/A5_select.log" 2>&1 && stamp "A. best annotator: $(tr '\n' ' ' < "$LOG/A5_select.log" | cut -c1-200)" \
      || { stamp "A. selection FAILED (see $LOG/A5_select.log)"; touch "$REANN/phaseA.failed"; }
  rm -f "$REANN/phaseA.running"
}

# ---------------- B. quota sweep: one priority list, any free GPU ----------------
# job = "annotator seed lambda"; maple = the lambda = 0 baseline (quota off; loop labels unused)
JOBS=("maple 0 0"
      "classifier 0 1" "classifier 0 0.5" "classifier 0 0.75" "classifier 0 0.25"
      "qwen 0 1" "qwen 0 0.5" "qwen 0 0.75" "qwen 0 0.25"
      "inherited 0 1" "inherited 0 0.5"
      "maple 1 0" "classifier 1 1" "classifier 1 0.5" "qwen 1 1"
      "inherited 0 0.75" "inherited 0 0.25" "classifier 1 0.75" "classifier 1 0.25")
job_dir() { local n="lam$3"; [ "$2" = 0 ] || n="${n}_seed$2"; echo "$ROOT/$1/$n"; }
ready() { [ "$1" != classifier ] || [ -s "$SPEC" ]; }
waiting_possible() { [ -f "$REANN/phaseA.running" ]; }   # classifier jobs may still become ready

run_job() {  # gpu ann seed lam
  local gpu=$1 ann=$2 seed=$3 lam=$4 out loop extra=()
  out=$(job_dir "$ann" "$seed" "$lam")
  loop=$ann; [ "$ann" = maple ] && loop=inherited
  [ "$ann" = classifier ] && extra=(--classifier_spec "$SPEC")
  stamp "B. run $ann seed=$seed lam=$lam on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu "$MPY" quota_sweep/quota_pe.py --maple_dir "$PE_DATA/MAPLE" --pool_csv "$POOL" \
    --eps "$EPS" --lam "$lam" --schedule "$SCHEDULE" --loop_annotator "$loop" --seed "$seed" \
    --test_ref "$TEST_REF" --out "$out" "${extra[@]}" > "$LOG/B_${ann}_s${seed}_lam${lam}.log" 2>&1 \
    && stamp "B. done $ann seed=$seed lam=$lam" || stamp "B. FAILED $ann seed=$seed lam=$lam (see $LOG/B_${ann}_s${seed}_lam${lam}.log)"
  "$PPY" quota_sweep/summarize.py --root "$ROOT" > /dev/null 2>&1 || true
}

worker() {  # gpu
  local gpu=$1
  while :; do
    if [ "$(hours_left)" -lt $(( MARGIN * 3600 )) ]; then stamp "B. GPU $gpu: less than ${MARGIN}h left, no new job"; return; fi
    local picked="" pending=0
    for job in "${JOBS[@]}"; do
      read -r ann seed lam <<< "$job"
      [ -f "$(job_dir "$ann" "$seed" "$lam")/DONE" ] && continue
      if ! ready "$ann"; then pending=1; continue; fi
      if mkdir "$CLAIMS/${ann}_${seed}_${lam}" 2>/dev/null; then picked="$job"; break; fi
    done
    if [ -n "$picked" ]; then read -r ann seed lam <<< "$picked"; run_job "$gpu" "$ann" "$seed" "$lam"; continue; fi
    if [ $pending = 1 ] && waiting_possible; then sleep 60; continue; fi
    stamp "B. GPU $gpu: nothing left to run"; return
  done
}

phase_a > "$LOG/A_phase.log" 2>&1 &
PA=$!
sleep 5

# GPU C: test-set reference labels, then the sweep
if [ ! -s "$TEST_REF" ]; then
  stamp "B0. Qwen labels for the test set on GPU $GC"
  CUDA_VISIBLE_DEVICES=$GC "$MPY" quota_sweep/quota_pe.py --maple_dir "$PE_DATA/MAPLE" --pool_csv "$POOL" \
    --eps "$EPS" --lam 0 --test_ref "$TEST_REF" --out "$ROOT/_prep" --prep_only > "$LOG/B0_test_ref.log" 2>&1 \
    || { stamp "B0. test reference FAILED (see $LOG/B0_test_ref.log)"; }
fi
worker "$GC" > "$LOG/B_worker_gpu$GC.log" 2>&1 &
WC=$!

wait $PA
stamp "A. finished: $(cat "$LOG/A5_select.log" 2>/dev/null | head -2 | tr '\n' ' ')"
worker "$GA" > "$LOG/B_worker_gpu$GA.log" 2>&1 &
WA=$!
worker "$GB" > "$LOG/B_worker_gpu$GB.log" 2>&1 &
WB=$!
wait $WA $WB $WC

"$PPY" quota_sweep/summarize.py --root "$ROOT" > "$LOG/C_summary.log" 2>&1 || true
stamp "all done. Re-annotating: $RES/REPORT.md ; best annotator: $SPEC ; sweep: $ROOT/SWEEP.md (frontier.png, curves.png)"
