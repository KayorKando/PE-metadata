#!/usr/bin/env bash
# One-time setup for the OpenReview runs on NCHC Nano5. LOGIN node (internet), repo root, after the bioRxiv setup
# (quota_sweep/nano5_pe_setup.sh and vote_embedding/nano5_setup.sh: the conda envs maple and pe-metadata exist):
#   cd ~/PE-metadata && bash openreview/nano5_setup.sh
# Light work only, no GPU:
#   1. data root /work/$USER/PE-openreview (own runs, caches, blind.json), marked with a DATASET file so that no
#      script mixes it up with the bioRxiv root; MAPLE is a symlink to the bioRxiv root's MAPLE clone
#   2. MAPLE's openreview/ folder (CSVs, AIM metadata, MAPLE.py, PE.py): copied from GitHub if the clone lacks it
#   3. model weights into /work/$USER/hf_cache: SciBERT (ft vote backbone), gte-base-en-v1.5 (MAUVE encoder, with
#      its remote code), plus sentence-t5-base and Qwen if missing; then an offline load check
# Check `hfsquota` first: the two new models are ~1 GB together.
set -eo pipefail
WORK=/work/$USER
SRC=${SRC:-$WORK/PE-metadata}              # the bioRxiv data root (holds the MAPLE clone)
ROOT=${NANO5_DATA:-$WORK/PE-openreview}
QWEN=${QWEN:-Qwen/Qwen2.5-7B-Instruct}
export HF_HOME=$WORK/hf_cache
[ -f openreview/nano5_setup.sh ] || { echo "run from the repo root"; exit 1; }
mkdir -p "$ROOT/cache" "$ROOT/quota_sweep" "$WORK/logs" "$HF_HOME"

# 1. data root
if [ -s "$ROOT/DATASET" ] && [ "$(cat "$ROOT/DATASET")" != openreview ]; then
  echo "$ROOT/DATASET says $(cat "$ROOT/DATASET"): not an OpenReview root"; exit 1
fi
echo openreview > "$ROOT/DATASET"
if [ ! -e "$ROOT/MAPLE" ]; then
  if [ -d "$SRC/MAPLE" ]; then ln -s "$SRC/MAPLE" "$ROOT/MAPLE"; echo "linked $ROOT/MAPLE -> $SRC/MAPLE"
  else git clone --depth 1 https://github.com/elichien-google/MAPLE.git "$ROOT/MAPLE"; fi
fi

# 2. MAPLE's openreview folder
OR=$ROOT/MAPLE/openreview
if [ ! -s "$OR/train_dataset_with_metadata_v4.csv" ] || [ ! -s "$OR/MAPLE/MAPLE.py" ]; then
  tmp=$(mktemp -d "$WORK/maple-src.XXXX")
  git clone --depth 1 https://github.com/elichien-google/MAPLE.git "$tmp/MAPLE"
  mkdir -p "$OR"
  cp -rn "$tmp/MAPLE/openreview/." "$OR/"
  rm -rf "$tmp"
  echo "copied MAPLE/openreview from GitHub"
fi
for f in train_dataset_with_metadata_v4.csv val_dataset_with_metadata_v4.csv test_dataset_with_metadata_v4.csv \
         "openreview_train_metadata_aim_eps1.0_(1,9).csv" "openreview_train_metadata_aim_eps2.0_(1,9).csv" \
         "openreview_train_metadata_aim_eps4.0_(1,9).csv" MAPLE/MAPLE.py MAPLE/PE.py MAPLE/util.py; do
  [ -s "$OR/$f" ] || { echo "MISSING $OR/$f"; exit 1; }
done
[ -s "$ROOT/MAPLE/biorxiv/utility_eval/compute_mauve.py" ] || { echo "MISSING MAPLE/biorxiv/utility_eval/compute_mauve.py"; exit 1; }
echo "MAPLE openreview data: ok"

# 3. models (download in the maple env: it runs generation, annotation, PE and MAUVE)
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate "$WORK/envs/${MAPLE_ENV_NAME:-maple}"
python - "$QWEN" <<'PY'
import sys
from huggingface_hub import snapshot_download
for repo in ["allenai/scibert_scivocab_uncased", "Alibaba-NLP/gte-base-en-v1.5", "sentence-transformers/sentence-t5-base",
             sys.argv[1]]:
    print("ok", repo, "->", snapshot_download(repo))
PY
# gte's modeling code lives in another repo (Alibaba-NLP/new-impl): one online load caches it for the offline jobs
python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('Alibaba-NLP/gte-base-en-v1.5'); AutoModel.from_pretrained('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True); print('gte remote code cached')"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
from transformers import AutoModel, AutoTokenizer
for m, trust in [("Alibaba-NLP/gte-base-en-v1.5", True), ("allenai/scibert_scivocab_uncased", False)]:
    AutoTokenizer.from_pretrained(m, trust_remote_code=trust); AutoModel.from_pretrained(m, trust_remote_code=trust)
    print("offline load ok:", m)
PY
PE_DATASET=openreview PE_DATA=$ROOT python -c "
import sys; sys.path.insert(0, '$OR/MAPLE'); import MAPLE, PE, util, vllm
sys.path.insert(0, '.'); import pe_data
p = pe_data.load_private('$ROOT/MAPLE'); a = pe_data.load_aim('$ROOT/MAPLE', '4.0')
print('import check ok;', len(p), 'private reviews,', p['PE.LABEL_ID'].nunique(), 'label ids,', len(a), 'AIM rows (eps 4) with a label id')"
conda deactivate
conda activate "$WORK/envs/${ENV_NAME:-pe-metadata}"
HF_HUB_OFFLINE=1 python -c "from transformers import AutoModel; AutoModel.from_pretrained('allenai/scibert_scivocab_uncased'); print('pe-metadata env: SciBERT ok')"
echo
echo "OpenReview root ready: $ROOT. Next: openreview/README.md (export PE_DATASET=openreview, then the prep jobs)."
