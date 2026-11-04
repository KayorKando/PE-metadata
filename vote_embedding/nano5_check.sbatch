#!/bin/bash
#SBATCH --job-name=vemb-check
#SBATCH --account=acd115132
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/work/%u/logs/%x-%j.out
# Step 3 on NCHC Nano5 (one H100): train normsoftmax, then ce, then run the check. Same as
# run_check.sh, but as a Slurm job, serial on one GPU (cheaper than holding two GPUs).
# Setup first: bash vote_embedding/nano5_setup.sh (login node). Then, from the repo root:
#   sbatch vote_embedding/nano5_check.sbatch
#   tail -f /work/$USER/logs/vemb-check-<jobid>.out          # progress; do not loop squeue
#   cat /work/$USER/PE-metadata/vote_embedding/eps4.0/check/CHECK.md
# Trained arms are reused on a resubmit. Settings: EPS (4.0), ENV_NAME (pe-metadata).
set -eo pipefail
EPS=${EPS:-4.0}
ENV_NAME=${ENV_NAME:-pe-metadata}
REPO=${SLURM_SUBMIT_DIR:-$PWD}
[ -f "$REPO/vote_embedding/train_embedder.py" ] || { echo "submit from the repo root (got $REPO)"; exit 1; }

echo "===== Job $SLURM_JOB_ID start @ $(date) on $(hostname), repo $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) ====="
nvidia-smi -L
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
export PYTHONUNBUFFERED=1
export PE_DATA=/work/$USER/PE-metadata HF_HOME=/work/$USER/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1          # compute nodes have no internet
RA=$PE_DATA/reannotate/eps$EPS
OUT=$PE_DATA/vote_embedding/eps$EPS
cd "$REPO"

for loss in normsoftmax ce; do
  if [ -s "$OUT/$loss/heads.pt" ]; then echo "[$(date '+%T')] $loss already trained"; continue; fi
  echo "[$(date '+%T')] training $loss"
  python vote_embedding/train_embedder.py --loss "$loss" --eps "$EPS" --pool_qwen "$RA/pool_qwen.csv" --out "$OUT/$loss"
done
echo "[$(date '+%T')] check"
python vote_embedding/check_embedding.py --eps "$EPS" --pool_qwen "$RA/pool_qwen.csv" \
  --private_qwen "$RA/private_qwen.csv" --ft_root "$OUT"
echo "===== Job $SLURM_JOB_ID end @ $(date): $OUT/check/CHECK.md ====="
