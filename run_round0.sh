#!/usr/bin/env bash
# One command for the round-0 experiment: (env setup if missing) -> generate the
# pre-vote pool on every GPU in GPUS -> merge -> k-NN vs probe blindness.
#
#   tmux new -d -s round0 "bash ~/PE-metadata/run_round0.sh"
#   tail -f $PE_DATA/round0/eps4.0/logs/*.log      # progress
#
# Settings (environment variables): EPS (default 4.0), GPUS (default 1,2),
# MAPLE_ENV, PE_ENV (venv paths). Finished shards are skipped on a rerun.
set -euo pipefail

EPS=${EPS:-4.0}
GPUS=${GPUS:-1,2}
: "${PE_DATA:?set PE_DATA, e.g. export PE_DATA=/data/\$USER/PE-metadata}"
MAPLE_ENV=${MAPLE_ENV:-/data/$USER/envs/maple}
PE_ENV=${PE_ENV:-/data/$USER/envs/pe-metadata}
MAPLE_DIR=$PE_DATA/MAPLE
OUT=$PE_DATA/round0/eps$EPS
LOG=$OUT/logs
RES=$PE_DATA/results/round0_eps$EPS
REPO=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$LOG"
IFS=, read -ra G <<< "$GPUS"
N=${#G[@]}
stamp() { echo "[$(date '+%F %T')] $*"; }

# --- 0. checks and MAPLE env (vLLM + private-evolution) ---
[ -d "$MAPLE_DIR/biorxiv" ] || git clone https://github.com/elichien-google/MAPLE.git "$MAPLE_DIR"
[ -x "$PE_ENV/bin/python" ] || { echo "missing $PE_ENV (Experiment 1 env)"; exit 1; }
if [ ! -x "$MAPLE_ENV/bin/python" ]; then
  stamp "creating $MAPLE_ENV"
  python3 -m venv "$MAPLE_ENV"
  "$MAPLE_ENV/bin/pip" install -q -U pip
  "$MAPLE_ENV/bin/pip" install -q vllm==0.10.1.1
  "$MAPLE_ENV/bin/pip" install -q "private-evolution[text] @ git+https://github.com/microsoft/DPSDA.git" datasets==4.0.0
fi
MPY=$MAPLE_ENV/bin/python
CUDA_VISIBLE_DEVICES=${G[0]} "$MPY" - <<'PY'
import torch
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0), arch)
assert arch in torch.cuda.get_arch_list(), f"this torch build has no {arch} kernels: {torch.cuda.get_arch_list()}"
PY
stamp "downloading Qwen2.5-7B-Instruct once (no race between shards)"
"$MPY" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')" > "$LOG/download.log" 2>&1

# --- 1. generate: one shard per GPU, in parallel ---
pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
for i in "${!G[@]}"; do
  if [ -s "$OUT/pool_shard${i}of$N.csv" ]; then stamp "shard $i already done, skipping"; continue; fi
  stamp "shard $i/$N on GPU ${G[$i]} -> $LOG/gen_shard$i.log"
  ( cd "$REPO" && CUDA_VISIBLE_DEVICES=${G[$i]} exec "$MPY" gen_round0.py --maple_dir "$MAPLE_DIR" \
      --eps "$EPS" --shard "$i" --num_shards "$N" --out "$OUT" ) > "$LOG/gen_shard$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]:-}"; do
  [ -z "$p" ] && continue
  wait "$p" || { stamp "a shard failed, see $LOG/gen_shard*.log"; exit 1; }
done
pids=()
trap - EXIT

# --- 2. merge ---
( cd "$REPO" && "$MPY" gen_round0.py --maple_dir "$MAPLE_DIR" --eps "$EPS" --merge --num_shards "$N" --out "$OUT" ) \
  | tee "$LOG/merge.log"

# --- 3. blindness on the pool ---
stamp "k-NN vs probe -> $RES"
( cd "$REPO" && CUDA_VISIBLE_DEVICES=${G[0]} "$PE_ENV/bin/python" run.py --csv "$OUT/pool.csv" --n 0 --repeats 5 \
    --attributes primary_research_area model_organism experimental_approach dominant_data_type \
                 research_focus_scale disease_mention sample_size research_goal word_count word_count_requested \
    --batch_size 256 --out "$RES" ) 2>&1 | tee "$LOG/blindness.log"

stamp "done. summary: $RES/summary.csv  plot: $RES/knn_vs_probe.png"
