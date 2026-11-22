#!/usr/bin/env bash
# LOGIN node, once, from the repo root: bash downstream/nano5_ntp_setup.sh
# Makes prajjwal1/bert-small loadable offline from /work/$USER/hf_cache (compute nodes have no internet; ~120 MB),
# checks the conda env (the `maple` env: transformers 4.55.4, the version MAPLE pins) and lists the data that
# downstream/nano5_ntp.sbatch reads. Light work only: no GPU, no training.
set -eo pipefail
WORK=/work/$USER
EPS=${EPS:-4.0}
MODEL=${MODEL:-prajjwal1/bert-small}
export HF_HOME=$WORK/hf_cache
[ -f quota_sweep/nano5_lib.sh ] || { echo "run from the repo root"; exit 1; }
source quota_sweep/nano5_lib.sh
PE_DATASET=biorxiv; PE_DATA=$(nano5_data)
mkdir -p "$HF_HOME" "$WORK/logs"
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate "$WORK/envs/${MAPLE_ENV_NAME:-maple}"
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "(MAPLE pins 4.55.4)")
PY
load="from transformers import AutoModelForCausalLM, AutoTokenizer as T; t = T.from_pretrained('$MODEL'); \
m = AutoModelForCausalLM.from_pretrained('$MODEL', is_decoder=True); \
print('$MODEL: vocab', len(t), '| parameters', sum(p.numel() for p in m.parameters()), '| is_decoder', m.config.is_decoder)"
if HF_HUB_OFFLINE=1 python -c "$load" 2>/dev/null; then
  echo "$MODEL loads offline from $HF_HOME: ok"
else
  echo "downloading $MODEL into $HF_HOME"
  python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL', allow_patterns=['*.json', '*.txt', 'model.safetensors', 'pytorch_model.bin']))"
  HF_HUB_OFFLINE=1 python -c "$load"
fi
echo
echo "Data under $PE_DATA:"
need() { if [ -e "$PE_DATA/$1" ]; then echo "  ok       $1"; else echo "  MISSING  $1   $2"; fi; }
need quota_sweep/test_ref_qwen.csv                       "(test texts and their labels)"
need MAPLE/biorxiv/biorxiv_train_metadata.csv            "(source 'real')"
need round0/eps$EPS/pool.csv                             "(source 'pool')"
need quota_sweep/eps$EPS                                 "(source 'maple': the PE runs)"
need augpe/eps$EPS                                       "(source 'augpe'; skipped by the job if missing)"
n=$(find "$PE_DATA/quota_sweep/eps$EPS" -mindepth 2 -maxdepth 2 -name DONE 2>/dev/null | wc -l)
a=$(find "$PE_DATA/augpe/eps$EPS" -mindepth 2 -maxdepth 2 -name DONE 2>/dev/null | wc -l)
echo "  finished PE runs at eps $EPS: $n (MAPLE tree), $a (Aug-PE)"
echo "Then, from the repo root:  sbatch downstream/nano5_ntp.sbatch   (see downstream/README.md)"
