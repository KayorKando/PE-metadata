#!/usr/bin/env bash
# One-time setup for PE runs on NCHC Nano5. Run on the LOGIN node, from the repo root:
#   cd ~/PE-metadata && bash quota_sweep/nano5_pe_setup.sh
# Light work only: conda env /work/$USER/envs/maple (vLLM), model downloads into /work, a Nano5 copy of the
# classifier spec, and a list of the data still missing. No GPU, no PE run, no polling.
# The env mirrors the lab cluster's MAPLE env (vllm 0.10.1.1, transformers 4.55.4, numpy < 2.3).
# If the import check at the end fails, compare with `pip freeze` of the lab env and add what is missing.
set -eo pipefail
WORK=/work/$USER
EPS=${EPS:-4.0}
ENV_NAME=${MAPLE_ENV_NAME:-maple}
QWEN=${QWEN:-Qwen/Qwen2.5-7B-Instruct}
export HF_HOME=$WORK/hf_cache
PE_DATA=$WORK/PE-metadata
mkdir -p "$WORK/logs" "$HF_HOME" "$PE_DATA/cache" "$PE_DATA/quota_sweep/eps$EPS"

module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
# env, pip cache and conda package cache in /work (/home is only 100 GB; vLLM + torch envs are ~10-15 GB each)
export PIP_CACHE_DIR=$WORK/.cache/pip CONDA_PKGS_DIRS=$WORK/.conda/pkgs
ENV_PATH=$WORK/envs/$ENV_NAME
[ -x "$ENV_PATH/bin/python" ] || conda create -p "$ENV_PATH" python=3.11 -y
conda activate "$ENV_PATH"
pip install vllm==0.10.1.1 transformers==4.55.4 "numpy<2.3" sentence-transformers scikit-learn pandas scipy \
  matplotlib joblib faiss-cpu mauve-text "huggingface_hub>=0.34,<1.0" datasets   # hub < 1.0 for transformers 4.55.4
# Private Evolution library (MAPLE imports pe.*): the exact commit of the lab env; --no-deps keeps the pins above
pip install --no-deps "private-evolution @ git+https://github.com/microsoft/DPSDA.git@c109ea5e90cddfd309903e8c877bdea1565fe366"
pip check

python - "$QWEN" <<'PY'
import sys
from huggingface_hub import snapshot_download
for repo in [sys.argv[1], "sentence-transformers/sentence-t5-base", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"]:
    print("downloaded", repo, "->", snapshot_download(repo))
PY

echo
echo "Data needed under $PE_DATA (copy from the lab cluster's \$PE_DATA, same relative paths):"
echo "(from the lab cluster: bash quota_sweep/nano5_copy_data.sh <nano5 account>)"
need() { if [ -e "$PE_DATA/$1" ]; then echo "  ok       $1"; else echo "  MISSING  $1   $2"; fi; }
need MAPLE/biorxiv                                    "(the whole MAPLE repo dir: private/test/AIM CSVs, MAPLE/, utility_eval/)"
need round0/eps$EPS/pool.csv
need quota_sweep/test_ref_qwen.csv                    "(Qwen labels of the test set)"
need reannotate/eps$EPS/best_annotator.json           "(classifier annotator spec)"
need reannotate/eps$EPS/results                       "(only results/<model>/models/ of the models named in the spec)"
need reannotate/eps$EPS/pool_qwen.csv                 "(to train the ft embedder)"
need vote_embedding/eps$EPS/normsoftmax/heads.pt      "(or train it: sbatch vote_embedding/nano5_check.sbatch)"
echo "  optional cache/emb_sentence-t5-base_*.npy (t5 embedding of the private set; recomputed if missing)"

SPEC=$PE_DATA/reannotate/eps$EPS/best_annotator.json
if [ -s "$SPEC" ]; then   # results_dir in the spec is an absolute lab-cluster path: point it at /work
  python - "$SPEC" "$PE_DATA/reannotate/eps$EPS/results" <<'PY'
import json, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
spec["results_dir"] = sys.argv[2]
out = Path(sys.argv[1]).with_name("best_annotator_nano5.json")
out.write_text(json.dumps(spec, indent=2))
print("wrote", out, "models:", {a: v["model"] for a, v in spec["attributes"].items()})
PY
fi
if [ -d "$PE_DATA/MAPLE/biorxiv/MAPLE" ]; then   # import check (imports only, no GPU)
  python -c "import sys; sys.path.insert(0, '$PE_DATA/MAPLE/biorxiv/MAPLE'); import MAPLE, util, vllm, mauve, faiss; print('import check ok')"
fi
echo "Then submit from the repo root:  bash quota_sweep/nano5_pe_submit.sh"
