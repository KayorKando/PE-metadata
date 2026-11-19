#!/bin/bash
#SBATCH --job-name=vemb-train
#SBATCH --account=acd115132
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/work/%u/logs/%x-%j.out
# Train extra normsoftmax vote embedders on the round-0 pool (Qwen labels), serially on one H100.
# Same recipe as the main ft embedder (train_embedder.py defaults); only the backbone or the seed changes:
#   ftt5     backbone sentence-transformers/sentence-t5-base (MAPLE's own vote embedding, fine-tuned)
#   ftt5_lr<X>  the same with backbone learning rate X (heads 10x), e.g. ftt5_lr1e-4
#   fts<N>   BiomedBERT (OpenReview: SciBERT), --seed N (changes the 70/15/15 split and the init)
#   ft       BiomedBERT (OpenReview: SciBERT), seed 0 (the main embedder; normally trained by the prep job)
# From the repo root:   VARIANTS="ftt5 fts1 fts2" sbatch vote_embedding/nano5_train.sbatch
#                       EPS=2.0 VARIANTS=ft sbatch ...      (pool_qwen.csv of that eps must exist)
# Already trained variants (heads.pt present) are skipped. Then submit PE runs with AFTER=<this job id>.
set -eo pipefail
EPS=${EPS:-4.0}
ENV_NAME=${ENV_NAME:-pe-metadata}
VARIANTS=${VARIANTS:-ftt5 fts1 fts2}
REPO=${SLURM_SUBMIT_DIR:-$PWD}
[ -f "$REPO/vote_embedding/train_embedder.py" ] || { echo "submit from the repo root (got $REPO)"; exit 1; }
cd "$REPO"
source quota_sweep/nano5_lib.sh
echo "===== Job ${SLURM_JOB_ID:-?} start @ $(date) on $(hostname), repo $(git rev-parse --short HEAD 2>/dev/null): $VARIANTS ====="
nvidia-smi -L
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate "/work/$USER/envs/$ENV_NAME"
export PE_DATASET=${PE_DATASET:-biorxiv}      # OpenReview: ft = SciBERT (train_embedder.py default per dataset)
export PYTHONUNBUFFERED=1 PE_DATA=$(nano5_data) HF_HOME=/work/$USER/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1          # compute nodes have no internet
POOL=$PE_DATA/reannotate/eps$EPS/pool_qwen.csv
[ -s "$POOL" ] || { echo "missing $POOL"; exit 1; }
for v in $VARIANTS; do
  vemb_dir "$v"
  [ -n "$VEMB_DIR" ] || { echo "$v is not a fine-tuned embedding"; exit 1; }
  out=$PE_DATA/vote_embedding/eps$EPS/$VEMB_DIR
  if [ -s "$out/heads.pt" ]; then echo "[$(date '+%T')] $v already trained ($out)"; continue; fi
  args=()
  case $v in
    ftt5) args=(--model sentence-transformers/sentence-t5-base) ;;
    ftt5_lr*) args=(--model sentence-transformers/sentence-t5-base --lr "${v#ftt5_lr}") ;;
    fts*) args=(--seed "${v#fts}") ;;
  esac
  echo "[$(date '+%T')] training $v -> $out (${args[*]})"
  python vote_embedding/train_embedder.py --loss normsoftmax --eps "$EPS" --pool_qwen "$POOL" --out "$out" "${args[@]}"
done
echo "===== Job ${SLURM_JOB_ID:-?} end @ $(date) ====="
