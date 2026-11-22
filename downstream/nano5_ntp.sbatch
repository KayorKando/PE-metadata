#!/bin/bash
#SBATCH --job-name=ntp
#SBATCH --account=acd115132
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/work/%u/logs/%x-%j.out
# Downstream NTP (downstream/ntp.py): bert-small trained on each synthetic set, next-token accuracy on the real
# test set, MAPLE's protocol. One shard of the jobs per GPU, in parallel; resumable (finished jobs are skipped).
# Setup once on the login node: bash downstream/nano5_ntp_setup.sh      Details: downstream/README.md
# From the repo root:
#   sbatch --gres=gpu:4 --cpus-per-task=32 --mem=192G downstream/nano5_ntp.sbatch      # eps 4.0, everything, 4 shards
#   SOURCES=maple RUNS_GLOB="maple/lam0 maple-ft/lam0" ROUNDS=last VARIANTS=full TRAIN_SEEDS="43 44" sbatch ...   # noise
#   EPS=2.0 SOURCES=maple sbatch downstream/nano5_ntp.sbatch
# Settings (environment):
#   EPS (4.0)
#   SOURCES ("maple augpe real pool")  maple = $PE_DATA/quota_sweep/eps$EPS (every arm), augpe = $PE_DATA/augpe/eps$EPS,
#                                      real = 2,000 random private training texts (upper bound, not private),
#                                      pool = 2,000 random texts of the round-0 pool (before any vote)
#   RUNS_GLOB ("*")       globs of run dirs inside the trees, space separated, e.g. "maple*/lam0* seed*"
#   ROUNDS ("0 last")     rounds to train on: last, all, or numbers (e.g. "0 4 last" adds Aug-PE's round 4)
#   VARIANTS ("full tok") full = MAPLE's protocol, tok = token-matched subsets, len<L> = texts cut to L tokens
#   TRAIN_SEEDS (42)      training seeds (42 = MAPLE's)
#   SAMPLE_SEEDS ("0 1 2 3 4")  samples of real / pool        TOKEN_BUDGET (auto)   EXTRA (more flags for ntp.py)
# The number of shards = number of GPUs of the job (--gres). Nano5 allows at most 12 CPUs per GPU: change --gres,
# --cpus-per-task and --mem together. Results: $PE_DATA/ntp, summary: $PE_DATA/ntp/NTP.md.
set -eo pipefail
EPS=${EPS:-4.0}
SOURCES=${SOURCES:-maple augpe real pool}
RUNS_GLOB=${RUNS_GLOB:-*}
ROUNDS=${ROUNDS:-0 last}
VARIANTS=${VARIANTS:-full tok}
TRAIN_SEEDS=${TRAIN_SEEDS:-42}
SAMPLE_SEEDS=${SAMPLE_SEEDS:-0 1 2 3 4}
TOKEN_BUDGET=${TOKEN_BUDGET:-auto}
ENV_NAME=${MAPLE_ENV_NAME:-maple}
REPO=${SLURM_SUBMIT_DIR:-$PWD}
[ -f "$REPO/downstream/ntp.py" ] || { echo "submit from the repo root (got $REPO)"; exit 1; }
cd "$REPO"
source quota_sweep/nano5_lib.sh
export PE_DATASET=biorxiv                      # MAPLE's NTP task is bioRxiv only (OpenReview: classification)
export PE_DATA=$(nano5_data) HF_HOME=/work/$USER/hf_cache PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1          # compute nodes have no internet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -f "$PE_DATA/DATASET" ] && [ "$(cat "$PE_DATA/DATASET")" != biorxiv ]; then
  echo "the data root $PE_DATA holds $(cat "$PE_DATA/DATASET") data: unset NANO5_DATA (NTP is bioRxiv only)"; exit 1
fi
OUT=$PE_DATA/ntp
TEST=$PE_DATA/quota_sweep/test_ref_qwen.csv      # MAPLE's biorxiv_test_metadata.csv texts, same order, plus labels
LOGDIR=${NTP_LOGDIR:-/work/$USER/logs}; mkdir -p "$LOGDIR" "$OUT"

set -f                                          # no filename expansion: RUNS_GLOB holds patterns for ntp.py
roots=(); csvs=()
for s in $SOURCES; do
  case $s in
    maple) roots+=("$PE_DATA/quota_sweep/eps$EPS") ;;
    augpe) if [ -d "$PE_DATA/augpe/eps$EPS" ]; then roots+=("$PE_DATA/augpe/eps$EPS"); else echo "no Aug-PE runs at eps $EPS: skipped"; fi ;;
    real) csvs+=("real=$PE_DATA/MAPLE/biorxiv/biorxiv_train_metadata.csv") ;;
    pool) csvs+=("pool_eps$EPS=$PE_DATA/round0/eps$EPS/pool.csv") ;;
    *) echo "unknown source '$s' (maple, augpe, real, pool)"; exit 1 ;;
  esac
done
args=(--test_csv "$TEST" --out "$OUT" --runs $RUNS_GLOB --rounds $ROUNDS --variants $VARIANTS
      --train_seeds $TRAIN_SEEDS --sample_seeds $SAMPLE_SEEDS --token_budget "$TOKEN_BUDGET")
[ ${#roots[@]} -eq 0 ] || args+=(--pe_root "${roots[@]}")
[ ${#csvs[@]} -eq 0 ] || args+=(--csv "${csvs[@]}")
args+=($EXTRA)
set +f

IFS=',' read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-0}"
n=${#gpus[@]}
threads=$(( ${SLURM_CPUS_PER_TASK:-$((8 * n))} / n ))
echo "===== Job ${SLURM_JOB_ID:-?}: NTP eps $EPS, sources: $SOURCES, runs: $RUNS_GLOB, rounds: $ROUNDS, variants: $VARIANTS," \
     "training seeds: $TRAIN_SEEDS, $n shard(s), start @ $(date) on $(hostname), repo $(git rev-parse --short HEAD 2>/dev/null) ====="
nvidia-smi -L
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate "/work/$USER/envs/$ENV_NAME"

case " $VARIANTS " in *" tok "*)      # token totals of every train set and the shared token budget, once, before the shards
  echo "[$(date '+%T')] counting training tokens -> $OUT/token_counts.csv, $OUT/token_budget.json"
  CUDA_VISIBLE_DEVICES= python downstream/ntp.py "${args[@]}" --count_only ;;
esac
pids=()
for i in "${!gpus[@]}"; do
  log=$LOGDIR/ntp-${SLURM_JOB_ID:-x}-shard$i.log
  echo "[$(date '+%T')] shard $i/$n on GPU ${gpus[$i]} -> $log"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} OMP_NUM_THREADS=$threads python downstream/ntp.py "${args[@]}" \
    --shard "$i" --num_shards "$n" > "$log" 2>&1 &
  pids+=($!)
done
fail=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "[$(date '+%T')] shard $i FAILED (see its log)"; fail=1; }; done
python downstream/summarize_ntp.py --root "$OUT" --test_ref "$TEST" || echo "summary failed (rerun it by hand, see downstream/README.md)"
echo "===== Job ${SLURM_JOB_ID:-?} end @ $(date), failed: $fail ====="
exit $fail
