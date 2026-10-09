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
# Install when the env is missing OR incomplete (e.g. an earlier install was interrupted).
if ! "$MAPLE_ENV/bin/python" -c "import torch, vllm, pe, sentence_transformers" >/dev/null 2>&1; then
  stamp "installing MAPLE env in $MAPLE_ENV (vllm, private-evolution); takes a few minutes"
  [ -x "$MAPLE_ENV/bin/python" ] || python3 -m venv "$MAPLE_ENV"
  "$MAPLE_ENV/bin/pip" install -q -U pip
  "$MAPLE_ENV/bin/pip" install vllm==0.10.1.1 --extra-index-url https://download.pytorch.org/whl/cu128
  "$MAPLE_ENV/bin/pip" install "private-evolution[text] @ git+https://github.com/microsoft/DPSDA.git" datasets==4.0.0
fi
MPY=$MAPLE_ENV/bin/python
# vllm 0.10.1.1 pins torch 2.7.1; the PyPI build of that is CUDA 12.6 and has no
# Blackwell (sm_120) kernels. Swap in the CUDA 12.8 build of the same versions.
has_arch() {
  CUDA_VISIBLE_DEVICES=${G[0]} "$MPY" -c "import torch; c=torch.cuda.get_device_capability(0); \
import sys; sys.exit(0 if f'sm_{c[0]}{c[1]}' in torch.cuda.get_arch_list() else 1)" 2>/dev/null
}
if ! has_arch; then
  stamp "torch build lacks this GPU's kernels; reinstalling torch 2.7.1 (CUDA 12.8)"
  "$MAPLE_ENV/bin/pip" install --force-reinstall torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
fi
# vllm 0.10.1.1 calls tokenizer APIs that transformers 5.x removed
# (e.g. all_special_tokens_extended); other packages may have pulled 5.x in.
if ! "$MPY" -c "import transformers, sys; sys.exit(0 if int(transformers.__version__.split('.')[0]) < 5 else 1)"; then
  stamp "transformers >= 5 is incompatible with vllm 0.10.1.1; installing 4.55.4"
  "$MAPLE_ENV/bin/pip" install "transformers==4.55.4"
fi
# vllm 0.10.1.1 pins numba 0.61.2, which only supports numpy <= 2.2; other
# packages may have pulled a newer numpy in. The engine imports numba lazily,
# so it only failed inside the worker process.
if ! "$MPY" -c "import numba" >/dev/null 2>&1; then
  stamp "numba import fails (numpy too new for numba 0.61.2); installing numpy<2.3"
  "$MPY" -c "import numba" 2>&1 | tail -n 2 || true
  "$MAPLE_ENV/bin/pip" install "numpy<2.3" "numba==0.61.2"
fi
# Import the modules the vLLM engine worker loads, so import errors show up here
# instead of as "Engine core initialization failed".
"$MPY" -c "import vllm.v1.worker.gpu_model_runner, vllm.v1.worker.gpu_worker; print('vllm worker imports ok')"
CUDA_VISIBLE_DEVICES=${G[0]} "$MPY" - <<'PY'
import torch
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
import transformers, vllm
print("transformers", transformers.__version__, "vllm", vllm.__version__)
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
