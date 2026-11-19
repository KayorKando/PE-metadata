#!/bin/bash
#SBATCH --job-name=pe-quota
#SBATCH --account=acd115132
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/work/%u/logs/%x-%j.out
# Several PE runs in one job, one GPU each, in parallel (vLLM generation + Qwen eval labels + vote + quota
# + MAUVE per run). Submit with quota_sweep/nano5_pe_submit.sh: it sets RUNS ("vemb seed lam;...") and
# the GPU / CPU / memory request (--gres, --cpus-per-task, --mem) to match the number of runs.
# Progress: tail -f /work/$USER/logs/<job name>-<jobid>.out (one line per run start / end) and the per-run
# logs /work/$USER/logs/pe-<jobid>-<run>.log   (do not loop squeue)
set -eo pipefail
EPS=${EPS:-4.0}
ENV_NAME=${MAPLE_ENV_NAME:-maple}
REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
source quota_sweep/nano5_lib.sh
export PE_DATASET=${PE_DATASET:-biorxiv}
export PE_DATA=$(nano5_data) HF_HOME=/work/$USER/hf_cache PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1                  # compute nodes have no internet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$PE_DATA/quota_sweep/eps$EPS
LOGDIR=/work/$USER/logs; mkdir -p "$LOGDIR"
IFS=';' read -ra runs <<< "${RUNS:?RUNS not set: submit with quota_sweep/nano5_pe_submit.sh}"
IFS=',' read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((${#runs[@]} - 1)))}"
[ "${#gpus[@]}" -ge "${#runs[@]}" ] || { echo "${#runs[@]} runs but only ${#gpus[@]} GPUs"; exit 1; }
threads=$(( ${SLURM_CPUS_PER_TASK:-$((8 * ${#runs[@]}))} / ${#runs[@]} ))
dataset_flags "$PE_DATA"

echo "===== Job ${SLURM_JOB_ID:-?} ($PE_DATASET, eps $EPS): ${#runs[@]} run(s) on GPUs ${gpus[*]}, start @ $(date) on $(hostname)," \
     "repo $(git rev-parse --short HEAD 2>/dev/null) ====="
nvidia-smi -L
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate "/work/$USER/envs/$ENV_NAME"

pids=(); names=()
for i in "${!runs[@]}"; do
  read -r vemb seed lam <<< "${runs[$i]}"
  run_dir "$vemb" "$seed" "$lam"
  out=$ROOT/$RUN_DIR
  if [ -f "$out/DONE" ]; then echo "[$(date '+%T')] skip $RUN_DIR (done)"; continue; fi
  vflags=(); aflags=()
  vemb_dir "$vemb"
  [ -z "$VEMB_DIR" ] || vflags=(--vote_emb ft --vote_emb_path "$PE_DATA/vote_embedding/eps$EPS/$VEMB_DIR")
  [ "$RUN_ANN" != classifier ] || aflags=(--classifier_spec "$PE_DATA/reannotate/eps$EPS/best_annotator_nano5.json")
  log=$LOGDIR/pe-${SLURM_JOB_ID:-x}-${RUN_DIR//\//_}.log
  echo "[$(date '+%T')] start $RUN_DIR on GPU ${gpus[$i]} -> $log"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} OMP_NUM_THREADS=$threads python quota_sweep/quota_pe.py --maple_dir "$PE_DATA/MAPLE" \
    --pool_csv "$PE_DATA/round0/eps$EPS/pool.csv" --eps "$EPS" --lam "$lam" --loop_annotator "$RUN_ANN" --seed "$seed" \
    --test_ref "$PE_DATA/quota_sweep/test_ref_qwen.csv" --out "$out" "${vflags[@]}" "${aflags[@]}" "${DATASET_FLAGS[@]}" \
    $EXTRA > "$log" 2>&1 &
  pids+=($!); names+=("$RUN_DIR")
done
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[$(date '+%T')] done ${names[$i]}"; else echo "[$(date '+%T')] FAILED ${names[$i]}"; fail=1; fi
done
python quota_sweep/summarize.py --root "$ROOT" > /dev/null 2>&1 || true      # refresh SWEEP.md
echo "===== Job ${SLURM_JOB_ID:-?} end @ $(date), failed: $fail ====="
exit $fail
