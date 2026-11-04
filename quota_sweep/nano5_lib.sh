# Shared by nano5_pe_submit.sh and nano5_pe.sbatch (sourced, not run).
# run_dir <vote_emb> <seed> <lam>  -> sets RUN_DIR (relative to the sweep root) and RUN_ANN (loop annotator).
# lambda = 0: quota off, loop labels unused -> the shared MAPLE baseline maple/ or maple-ft/ with inherited
# labels (as on the lab cluster); summarize.py shows it as the lambda = 0 point of every curve of that embedding.
run_dir() {
  local vemb=$1 seed=$2 lam=$3 name dir
  name="lam${lam}"; [ "$seed" = 0 ] || name="${name}_seed${seed}"
  RUN_ANN=${ANNOTATOR:-classifier}; dir=$RUN_ANN
  if awk -v l="$lam" 'BEGIN { exit !(l + 0 == 0) }'; then RUN_ANN=inherited; dir=maple; fi
  [ "$vemb" != ft ] || dir="${dir}-ft"
  RUN_DIR="$dir/$name"
}
