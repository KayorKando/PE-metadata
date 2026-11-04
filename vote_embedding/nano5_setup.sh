#!/usr/bin/env bash
# One-time setup for NCHC Nano5. Run on the LOGIN node (cbi-lgn01), from the repo root:
#   cd ~/PE-metadata && bash vote_embedding/nano5_setup.sh
# Login node = internet, light work only. Compute nodes have NO internet, so everything the
# job needs (pip packages, model weights, data) is fetched here, before sbatch.
# What it does: conda env in /home (code + env), HF cache + data + logs in /work, downloads
# BiomedBERT and sentence-t5-base, then lists which data files are still missing.
# No training, no GPU, no polling.
set -eo pipefail
WORK=/work/$USER
ENV_NAME=${ENV_NAME:-pe-metadata}
EPS=${EPS:-4.0}
export HF_HOME=$WORK/hf_cache
PE_DATA=$WORK/PE-metadata
mkdir -p "$WORK/logs" "$HF_HOME" "$PE_DATA/reannotate/eps$EPS/results/biomedbert" "$PE_DATA/cache"

module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda env list | grep "^$ENV_NAME " >/dev/null || conda create -n "$ENV_NAME" python=3.11 -y
conda activate "$ENV_NAME"
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers sentence-transformers scikit-learn pandas scipy matplotlib huggingface_hub

python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", "sentence-transformers/sentence-t5-base"]:
    print("downloaded", repo, "->", snapshot_download(repo))
PY

echo
echo "Data needed in $PE_DATA/reannotate/eps$EPS/ (copy from the lab cluster, e.g. with scp/rsync):"
for f in pool_qwen.csv private_qwen.csv results/biomedbert/metrics.csv; do
  if [ -s "$PE_DATA/reannotate/eps$EPS/$f" ]; then echo "  ok       $f"; else echo "  MISSING  $f"; fi
done
echo "(results/biomedbert/metrics.csv is optional: only the reference column of the annotator table.)"
echo "Then submit from the repo root:  sbatch vote_embedding/nano5_check.sbatch"
