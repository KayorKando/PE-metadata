#!/usr/bin/env bash
# LOGIN node, once: make gpt2-large loadable offline from /work/$USER/hf_cache for nano5_mauve.sbatch.
# Reuses the copy already in /work/$USER/hf (another project) via a symlink, so no extra /work quota;
# downloads only if there is none (safetensors + tokenizer, ~3.3 GB). Check `hfsquota` first.
set -eo pipefail
WORK=/work/$USER
MODEL=${MODEL:-gpt2-large}
export HF_HOME=$WORK/hf_cache
mkdir -p "$HF_HOME/hub"
for id in "$MODEL" "openai-community/$MODEL"; do
  d=models--${id//\//--}
  for src in "$WORK/hf/hub/$d" "$WORK/hf/$d"; do
    if [ -d "$src" ] && [ ! -e "$HF_HOME/hub/$d" ]; then ln -s "$src" "$HF_HOME/hub/$d"; echo "linked $src"; fi
  done
done
module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate "$WORK/envs/${MAPLE_ENV_NAME:-maple}"
if HF_HUB_OFFLINE=1 python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL'); AutoModel.from_pretrained('$MODEL')" >/dev/null 2>&1; then
  echo "$MODEL loads offline from $HF_HOME: ok"
else
  echo "downloading $MODEL into $HF_HOME"
  python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL', allow_patterns=['*.json', '*.txt', 'model.safetensors']))"
  HF_HUB_OFFLINE=1 python -c "from transformers import AutoModel; AutoModel.from_pretrained('$MODEL'); print('ok')"
fi
