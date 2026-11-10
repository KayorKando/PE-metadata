# Shared by nano5_pe_submit.sh and nano5_pe.sbatch (sourced, not run).
# Vote embeddings (the first word of a RUNS entry):
#   t5        MAPLE's sentence-t5 (no fine-tuning)
#   ft        BiomedBERT fine-tuned with normsoftmax, embedder seed 0     -> vote_embedding/eps<eps>/normsoftmax
#   fts<N>    the same, embedder trained with seed N (split + init)        -> .../normsoftmax-seed<N>
#   ftt5      sentence-t5 fine-tuned with normsoftmax (same recipe)        -> .../normsoftmax-t5
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
    fts[0-9]*) VEMB_DIR=normsoftmax-seed${1#fts} ;;
    *) echo "unknown vote embedding '$1' (t5, ft, fts<N>, ftt5)" >&2; return 1 ;;
  esac
}
