#!/bin/bash
#SBATCH --job-name=mauve-rescore
#SBATCH --account=acd115132
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/work/%u/logs/%x-%j.out
# Third-party MAUVE (quota_sweep/mauve_rescore.py) for every finished round of every PE run, one shard of the
# run dirs per GPU, in parallel. Setup once on the login node: bash quota_sweep/nano5_mauve_setup.sh
# From the repo root:
#   sbatch --gres=gpu:4 --cpus-per-task=32 --mem=192G quota_sweep/nano5_mauve.sbatch   # gpt2-large, eps 4.0, all runs, 4 shards
#   MODEL=sentence-t5-base RUNS_GLOB="maple/lam0*" sbatch ...         # sanity: must reproduce the stored mauve
#   sbatch --dependency=afterok:<pe job> quota_sweep/nano5_mauve.sbatch   # after new PE runs (resumable)
# Settings: EPS (4.0), MODEL (gpt2-large), RUNS_GLOB (*/lam*), LAST_ONLY (empty = all rounds), LENGTH_MATCH (1 = length-matched survivors), MAUVE_ROOT (run tree).
# The number of shards = number of GPUs of the job (--gres). Nano5 allows at most 12 CPUs per GPU:
# change --gres, --cpus-per-task and --mem together (default 1 GPU / 8 CPUs). Then: python quota_sweep/summarize.py --root ...
set -eo pipefail
EPS=${EPS:-4.0}
MODEL=${MODEL:-gpt2-large}
RUNS_GLOB=${RUNS_GLOB:-*/lam*}
ENV_NAME=${MAPLE_ENV_NAME:-maple}
REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
export PE_DATA=${NANO5_DATA:-/work/$USER/PE-metadata} HF_HOME=/work/$USER/hf_cache PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ROOT=${MAUVE_ROOT:-$PE_DATA/quota_sweep/eps$EPS}       # MAUVE_ROOT: other run trees, e.g. $PE_DATA/augpe/eps4.0
LOGDIR=/work/$USER/logs; mkdir -p "$LOGDIR"
IFS=',' read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-0}"
n=${#gpus[@]}
threads=$(( ${SLURM_CPUS_PER_TASK:-$((8 * n))} / n ))
echo "===== Job ${SLURM_JOB_ID:-?}: MAUVE with $MODEL on $ROOT/$RUNS_GLOB, $n shard(s), start @ $(date) on $(hostname) ====="
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate "/work/$USER/envs/$ENV_NAME"
pids=()
for i in "${!gpus[@]}"; do
  log=$LOGDIR/mauve-${SLURM_JOB_ID:-x}-shard$i.log
  echo "[$(date '+%T')] shard $i/$n on GPU ${gpus[$i]} -> $log"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} OMP_NUM_THREADS=$threads python quota_sweep/mauve_rescore.py --root "$ROOT" \
    --maple_dir "$PE_DATA/MAPLE" --test_ref "$PE_DATA/quota_sweep/test_ref_qwen.csv" --model "$MODEL" \
    --runs "$RUNS_GLOB" --shard "$i" --num_shards "$n" ${LAST_ONLY:+--last_only} ${LENGTH_MATCH:+--length_match} > "$log" 2>&1 &
  pids+=($!)
done
fail=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "[$(date '+%T')] shard $i FAILED"; fail=1; }; done
python quota_sweep/summarize.py --root "$ROOT" > /dev/null 2>&1 || true
echo "===== Job ${SLURM_JOB_ID:-?} end @ $(date), failed: $fail ====="
exit $fail
