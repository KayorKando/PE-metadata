#!/bin/bash
#SBATCH --job-name=augpe
#SBATCH --account=acd115132
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/work/%u/logs/%x-%j.out
# Vanilla Aug-PE drift runs (augpe/augpe_pe.py), one seed per GPU, all seeds of the job in parallel.
# Submit with augpe/nano5_augpe_submit.sh: it sets SEEDS and the GPU / CPU / memory request.
# Progress: tail -f /work/$USER/logs/<job name>-<jobid>.out and the per-seed logs /work/$USER/logs/augpe-<jobid>-seed<s>.log
set -eo pipefail
EPS=${EPS:-4.0}
ENV_NAME=${MAPLE_ENV_NAME:-maple}
REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
export PE_DATA=${NANO5_DATA:-/work/$USER/PE-metadata} HF_HOME=/work/$USER/hf_cache PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1                  # compute nodes have no internet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$PE_DATA/augpe/eps$EPS
LOGDIR=/work/$USER/logs; mkdir -p "$LOGDIR" "$ROOT"
read -ra seeds <<< "${SEEDS:?SEEDS not set: submit with augpe/nano5_augpe_submit.sh}"
IFS=',' read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((${#seeds[@]} - 1)))}"
[ "${#gpus[@]}" -ge "${#seeds[@]}" ] || { echo "${#seeds[@]} seeds but only ${#gpus[@]} GPUs"; exit 1; }
threads=$(( ${SLURM_CPUS_PER_TASK:-$((8 * ${#seeds[@]}))} / ${#seeds[@]} ))

echo "===== Job ${SLURM_JOB_ID:-?}: Aug-PE eps $EPS, seeds ${seeds[*]} on GPUs ${gpus[*]}, start @ $(date) on $(hostname)," \
     "repo $(git rev-parse --short HEAD 2>/dev/null) ====="
nvidia-smi -L
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate "/work/$USER/envs/$ENV_NAME"

pids=(); names=()
for i in "${!seeds[@]}"; do
  s=${seeds[$i]}
  out=$ROOT/seed$s
  if [ -f "$out/DONE" ]; then echo "[$(date '+%T')] skip seed$s (done)"; continue; fi
  log=$LOGDIR/augpe-${SLURM_JOB_ID:-x}-seed$s.log
  echo "[$(date '+%T')] start seed$s on GPU ${gpus[$i]} -> $log"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} OMP_NUM_THREADS=$threads python augpe/augpe_pe.py --maple_dir "$PE_DATA/MAPLE" \
    --eps "$EPS" --seed "$s" --test_ref "$PE_DATA/quota_sweep/test_ref_qwen.csv" --out "$out" $EXTRA > "$log" 2>&1 &
  pids+=($!); names+=("seed$s")
done
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[$(date '+%T')] done ${names[$i]}"; else echo "[$(date '+%T')] FAILED ${names[$i]}"; fail=1; fi
done
python augpe/summarize_augpe.py --augpe_root "$ROOT" --maple_root "$PE_DATA/quota_sweep/eps$EPS/maple" > /dev/null 2>&1 || true
echo "===== Job ${SLURM_JOB_ID:-?} end @ $(date), failed: $fail ====="
exit $fail
