#!/bin/bash
#SBATCH --job-name=prep-eps
#SBATCH --account=acd115132
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/work/%u/logs/%x-%j.out
# Everything a new privacy level needs before its PE runs, on one H100 (eps 4.0 was done on the lab cluster):
#   1. round-0 pool (gen_round0.py: AIM metadata at this eps, Qwen, MAPLE's prompt)   -> round0/eps$EPS/pool.csv
#   2. Qwen labels of the pool (reannotate/annotate_qwen.py)                          -> reannotate/eps$EPS/pool_qwen.csv
#   3. the ft vote embedder on it (train_embedder.py, normsoftmax, seed 0)            -> vote_embedding/eps$EPS/normsoftmax
# The test reference (quota_sweep/test_ref_qwen.csv) and the private set do not depend on eps and are shared.
# OpenReview (PE_DATASET=openreview): step 3 trains SciBERT, and the eps-$BLIND_EPS job (4.0) also makes
#   2b. the Qwen-labelled test reference (quota_pe.py --prep_only)                    -> quota_sweep/test_ref_qwen.csv
#   2c. the blind attributes, k-NN screen on its round-0 pool (pick_blind.py)          -> blind.json (all eps use it)
# so the PE jobs of every eps wait for that job too (AFTER=<eps-4 prep>:<own prep>). Give OpenReview preps
# --time=04:00:00 (14k few-shot prompts of ~5k tokens + 14k reviews to label + SciBERT).
# eps = inf: true metadata, no vote noise (non-private reference). From the repo root:
#   for e in 1.0 2.0 inf; do EPS=$e sbatch --job-name=prep-eps$e quota_sweep/nano5_prep_eps.sbatch; done
#   EPS=2.0 GROUP_BY=lam VOTE_EMBS="t5 ft" LAMS=0 SEEDS="0 1 2 3 4" AFTER=<prep job> bash quota_sweep/nano5_pe_submit.sh
# Finished steps are skipped on a rerun.
set -eo pipefail
EPS=${EPS:?set EPS (1.0, 2.0 or inf)}
BLIND_EPS=${BLIND_EPS:-4.0}
REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
source quota_sweep/nano5_lib.sh
export PE_DATASET=${PE_DATASET:-biorxiv}
export PE_DATA=$(nano5_data) HF_HOME=/work/$USER/hf_cache PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R0=$PE_DATA/round0/eps$EPS
RA=$PE_DATA/reannotate/eps$EPS
VE=$PE_DATA/vote_embedding/eps$EPS/normsoftmax
mkdir -p "$R0" "$RA"
stamp() { echo "[$(date '+%F %T')] $*"; }
stamp "job ${SLURM_JOB_ID:-?} $PE_DATASET eps $EPS on $(hostname), repo $(git rev-parse --short HEAD 2>/dev/null), data $PE_DATA"
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"

conda activate "/work/$USER/envs/${MAPLE_ENV_NAME:-maple}"
if [ ! -s "$R0/pool.csv" ]; then
  stamp "1. round-0 pool"
  [ -s "$R0/pool_shard0of1.csv" ] || python gen_round0.py --maple_dir "$PE_DATA/MAPLE" --eps "$EPS" --shard 0 --num_shards 1 --out "$R0"
  python gen_round0.py --maple_dir "$PE_DATA/MAPLE" --eps "$EPS" --merge --num_shards 1 --out "$R0"
else stamp "1. pool exists"; fi
if [ ! -s "$RA/pool_qwen.csv" ]; then
  stamp "2. Qwen labels of the pool"
  python reannotate/annotate_qwen.py --csv "$R0/pool.csv" --maple_dir "$PE_DATA/MAPLE" --out "$RA/pool_qwen.csv"
else stamp "2. pool_qwen exists"; fi
if [ "$PE_DATASET" != biorxiv ] && [ "$EPS" = "$BLIND_EPS" ]; then
  TEST_REF=$PE_DATA/quota_sweep/test_ref_qwen.csv
  if [ ! -s "$TEST_REF" ]; then
    stamp "2b. Qwen labels of the test set (+ t5 embedding of the private set, cached)"
    python quota_sweep/quota_pe.py --maple_dir "$PE_DATA/MAPLE" --pool_csv "$R0/pool.csv" --eps "$EPS" --lam 0 \
      --test_ref "$TEST_REF" --out "$PE_DATA/quota_sweep/_prep_test_ref" --prep_only
  else stamp "2b. test reference exists"; fi
  if [ ! -s "$PE_DATA/blind.json" ]; then
    stamp "2c. blind attributes (k-NN screen on the round-0 pool)"
    python blindness/pick_blind.py --pool_csv "$R0/pool.csv" --out "$PE_DATA/blind.json"
  else stamp "2c. blind.json exists: $(python -c "import json; print(json.load(open('$PE_DATA/blind.json'))['blind'])")"; fi
fi
conda deactivate

conda activate "/work/$USER/envs/${ENV_NAME:-pe-metadata}"
if [ ! -s "$VE/heads.pt" ]; then
  stamp "3. ft embedder"
  python vote_embedding/train_embedder.py --loss normsoftmax --eps "$EPS" --pool_qwen "$RA/pool_qwen.csv" --out "$VE"
else stamp "3. embedder exists"; fi
stamp "done"
