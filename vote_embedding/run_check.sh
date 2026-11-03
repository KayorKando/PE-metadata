#!/usr/bin/env bash
# Step 3 in one command: train both arms (normsoftmax on GPU A, ce on GPU B, in parallel), then the check.
#   tmux new -d -s vemb "bash ~/PE-metadata/vote_embedding/run_check.sh"
#   tail -f $PE_DATA/vote_embedding/eps4.0/logs/*.log
#   cat $PE_DATA/vote_embedding/eps4.0/check/CHECK.md
# Settings: EPS (4.0), GPUS (1,2), PE_ENV. Trained arms are reused on a rerun.
set -euo pipefail
EPS=${EPS:-4.0}
GPUS=${GPUS:-1,2}
: "${PE_DATA:?set PE_DATA, e.g. export PE_DATA=/data/\$USER/PE-metadata}"
export HF_HOME=${HF_HOME:-/data/$USER/hf_cache}
PE_ENV=${PE_ENV:-/data/$USER/envs/pe-metadata}
REPO=$(cd "$(dirname "$0")/.." && pwd)
RA=$PE_DATA/reannotate/eps$EPS
OUT=$PE_DATA/vote_embedding/eps$EPS
LOG=$OUT/logs
mkdir -p "$LOG"
IFS=, read -ra G <<< "$GPUS"; GA=${G[0]}; GB=${G[1]:-${G[0]}}
stamp() { echo "[$(date '+%F %T')] $*"; }
cd "$REPO"
stamp "repo $(git rev-parse --short HEAD 2>/dev/null || echo '?'), outputs -> $OUT"

pids=()
train() {  # loss gpu
  if [ -s "$OUT/$1/heads.pt" ]; then stamp "$1 already trained"; return; fi
  stamp "training $1 on GPU $2 -> $LOG/train_$1.log"
  CUDA_VISIBLE_DEVICES=$2 "$PE_ENV/bin/python" vote_embedding/train_embedder.py --loss "$1" --eps "$EPS" \
    --pool_qwen "$RA/pool_qwen.csv" --out "$OUT/$1" > "$LOG/train_$1.log" 2>&1 &
  pids+=($!)
}
train normsoftmax "$GA"
train ce "$GB"
for p in "${pids[@]:-}"; do [ -z "$p" ] || wait "$p" || { stamp "training failed, see $LOG/train_*.log"; exit 1; }; done

stamp "check -> $LOG/check.log"
CUDA_VISIBLE_DEVICES=$GA "$PE_ENV/bin/python" vote_embedding/check_embedding.py --eps "$EPS" \
  --pool_qwen "$RA/pool_qwen.csv" --private_qwen "$RA/private_qwen.csv" --ft_root "$OUT" > "$LOG/check.log" 2>&1
stamp "done: $OUT/check/CHECK.md"
