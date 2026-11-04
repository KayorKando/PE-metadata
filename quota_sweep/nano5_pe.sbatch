#!/bin/bash
#SBATCH --job-name=pe-quota
#SBATCH --account=acd115132
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/work/%u/logs/%x-%A_%a.out
# One PE run per array task (vLLM generation + Qwen eval labels + vote + quota + MAUVE on one H100).
# Submit with quota_sweep/nano5_pe_submit.sh (it sets --array and the grid), not directly.
# Progress: tail -f /work/$USER/logs/pe-quota-<jobid>_<task>.out   (do not loop squeue)
set -eo pipefail
EPS=${EPS:-4.0}
ANNOTATOR=${ANNOTATOR:-classifier}
ENV_NAME=${MAPLE_ENV_NAME:-maple}
REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
export PE_DATA=${NANO5_DATA:-/work/$USER/PE-metadata} HF_HOME=/work/$USER/hf_cache PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1                  # compute nodes have no internet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# this task's (vote_emb, seed, lambda): same order as nano5_pe_submit.sh counts them
jobs=(); for v in $VOTE_EMBS; do for s in $SEEDS; do for l in $LAMS; do jobs+=("$v $s $l"); done; done; done
read -r vemb seed lam <<< "${jobs[$SLURM_ARRAY_TASK_ID]}"
name="lam${lam}"; [ "$seed" = 0 ] || name="${name}_seed${seed}"
# lambda = 0: quota off, loop labels unused -> the shared MAPLE baseline (maple/ or maple-ft/, inherited
# labels, as on the lab cluster); summarize.py shows it as the lambda = 0 point of every curve of that embedding
ann=$ANNOTATOR; dir=$ANNOTATOR
if awk -v l="$lam" 'BEGIN { exit !(l + 0 == 0) }'; then ann=inherited; dir=maple; fi
vflags=(); aflags=()
if [ "$vemb" = ft ]; then dir="${dir}-ft"; vflags=(--vote_emb ft --vote_emb_path "$PE_DATA/vote_embedding/eps$EPS/normsoftmax"); fi
if [ "$ann" = classifier ]; then aflags=(--classifier_spec "$PE_DATA/reannotate/eps$EPS/best_annotator_nano5.json"); fi
ROOT=$PE_DATA/quota_sweep/eps$EPS
out=$ROOT/$dir/$name

echo "===== Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}: $dir/$name, start @ $(date) on $(hostname), repo $(git rev-parse --short HEAD 2>/dev/null) ====="
nvidia-smi -L
if [ -f "$out/DONE" ]; then echo "already done: $out"; exit 0; fi
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
python quota_sweep/quota_pe.py --maple_dir "$PE_DATA/MAPLE" --pool_csv "$PE_DATA/round0/eps$EPS/pool.csv" \
  --eps "$EPS" --lam "$lam" --loop_annotator "$ann" --seed "$seed" \
  --test_ref "$PE_DATA/quota_sweep/test_ref_qwen.csv" --out "$out" "${vflags[@]}" "${aflags[@]}" $EXTRA
python quota_sweep/summarize.py --root "$ROOT" > /dev/null 2>&1 || true      # refresh SWEEP.md
echo "===== done @ $(date): $out ====="
