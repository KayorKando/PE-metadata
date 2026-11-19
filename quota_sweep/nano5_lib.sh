# Shared by nano5_pe_submit.sh and nano5_pe.sbatch (sourced, not run).
# Vote embeddings (the first word of a RUNS entry):
#   t5        MAPLE's sentence-t5 (no fine-tuning)
#   ft        BiomedBERT fine-tuned with normsoftmax, embedder seed 0     -> vote_embedding/eps<eps>/normsoftmax
#   fts<N>    the same, embedder trained with seed N (split + init)        -> .../normsoftmax-seed<N>
#   ftt5      sentence-t5 fine-tuned with normsoftmax (same recipe)        -> .../normsoftmax-t5
#   ftt5_lr<X> the same with learning rate X (e.g. ftt5_lr1e-4)           -> .../normsoftmax-t5-lr<X>
# run_dir <vote_emb> <seed> <lam>  -> sets RUN_DIR (relative to the sweep root) and RUN_ANN (loop annotator).
# lambda = 0: quota off, loop labels unused -> the shared MAPLE baseline maple/ (t5) or maple-<vemb>/ with
# inherited labels; summarize.py shows it as the lambda = 0 point of every curve of that embedding.
run_dir() {
  local vemb=$1 seed=$2 lam=$3 name dir
  name="lam${lam}"; [ "$seed" = 0 ] || name="${name}_seed${seed}"
  RUN_ANN=${ANNOTATOR:-classifier}; dir=$RUN_ANN
  if awk -v l="$lam" 'BEGIN { exit !(l + 0 == 0) }'; then RUN_ANN=inherited; dir=maple; fi
  [ "$vemb" = t5 ] || dir="${dir}-${vemb}"
  RUN_DIR="$dir/$name"
}
# vemb_dir <vote_emb>  -> sets VEMB_DIR, the trained embedder's dir name under vote_embedding/eps<eps>/ (empty for t5)
vemb_dir() {
  case $1 in
    t5) VEMB_DIR= ;;
    ft) VEMB_DIR=normsoftmax ;;
    ftt5) VEMB_DIR=normsoftmax-t5 ;;
    ftt5_lr*) VEMB_DIR=normsoftmax-t5-lr${1#ftt5_lr} ;;
    fts[0-9]*) VEMB_DIR=normsoftmax-seed${1#fts} ;;
    *) echo "unknown vote embedding '$1' (t5, ft, fts<N>, ftt5, ftt5_lr<X>)" >&2; return 1 ;;
  esac
}
# ---- dataset (PE_DATASET: biorxiv, the default, or openreview) ----
# nano5_data  -> the data root: NANO5_DATA if set, else /work/$USER/PE-metadata (biorxiv) or /work/$USER/PE-<dataset>
#               (OpenReview's root is made by openreview/nano5_setup.sh; its DATASET file guards against mix-ups)
nano5_data() {
  if [ -n "$NANO5_DATA" ]; then echo "$NANO5_DATA"
  elif [ "${PE_DATASET:-biorxiv}" = biorxiv ]; then echo "/work/$USER/PE-metadata"
  else echo "/work/$USER/PE-${PE_DATASET}"; fi
}
# private_csv -> the dataset's private set, relative to the data root (every PE job reads it)
private_csv() {
  case ${PE_DATASET:-biorxiv} in
    biorxiv) echo MAPLE/biorxiv/biorxiv_train_metadata.csv ;;
    openreview) echo MAPLE/openreview/train_dataset_with_metadata_v4.csv ;;
    *) echo "unknown PE_DATASET '$PE_DATASET' (biorxiv or openreview)" >&2; return 1 ;;
  esac
}
# maple_subdir -> MAPLE's folder of the dataset (biorxiv / openreview)
maple_subdir() { echo "${PE_DATASET:-biorxiv}"; }
# dataset_flags <data root> -> DATASET_FLAGS, extra flags for quota_pe.py / augpe_pe.py:
#   OpenReview has no fixed blind attributes; they come from the round-0 screen (blindness/pick_blind.py)
dataset_flags() {
  DATASET_FLAGS=()
  [ "${PE_DATASET:-biorxiv}" = biorxiv ] || DATASET_FLAGS=(--blind_json "$1/blind.json")
}
# job_prefix -> "" for bioRxiv, "or-" for OpenReview (job names, so squeue / log names show the dataset)
job_prefix() { case ${PE_DATASET:-biorxiv} in biorxiv) ;; openreview) echo "or-" ;; *) echo "${PE_DATASET}-" ;; esac; }
