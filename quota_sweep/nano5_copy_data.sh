#!/usr/bin/env bash
# Copy what the Nano5 runs need from the LAB CLUSTER's $PE_DATA to Nano5's /work/<account>/PE-metadata,
# keeping the same relative paths. Run on the lab cluster, from the repo root:
#   bash quota_sweep/nano5_copy_data.sh <nano5 account>            # rsync over ssh (asks password + OTP once)
#   TAR=1 bash quota_sweep/nano5_copy_data.sh                       # if the lab cluster cannot reach Nano5:
#       writes ~/pe_nano5_data.tar; move it via your laptop, then on Nano5: tar xf pe_nano5_data.tar -C /work/$USER/PE-metadata
#   DRY=1 bash quota_sweep/nano5_copy_data.sh                       # only list the files
# Only the classifier models named in best_annotator.json are copied. The ft embedder is trained on Nano5
# (nano5_check.sbatch); embedding caches are recomputed there.
set -eo pipefail
EPS=${EPS:-4.0}
: "${PE_DATA:?set PE_DATA to the lab cluster data dir, e.g. /data/$USER/PE-metadata}"
cd "$PE_DATA"
R=reannotate/eps$EPS
files=(MAPLE round0/eps$EPS/pool.csv quota_sweep/test_ref_qwen.csv
       $R/pool_qwen.csv $R/private_qwen.csv $R/best_annotator.json)
[ -s "$R/results/biomedbert/metrics.csv" ] && files+=("$R/results/biomedbert/metrics.csv")   # optional reference
if [ -s "$R/best_annotator.json" ]; then
  models=$(python3 - "$R/best_annotator.json" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
print(" ".join(sorted({v["model"] for v in spec["attributes"].values()})))
PY
)
  for m in $models; do files+=("$R/results/$m/models"); done
fi
missing=0
for f in "${files[@]}"; do [ -e "$f" ] || { echo "MISSING on the lab cluster: $PE_DATA/$f"; missing=1; }; done
[ $missing -eq 0 ] || exit 1
echo "from $PE_DATA:"; du -sh "${files[@]}" 2>/dev/null | sed 's/^/  /'
if [ -n "$DRY" ]; then exit 0; fi
if [ -n "$TAR" ]; then
  tar cf "$HOME/pe_nano5_data.tar" "${files[@]}"
  echo "wrote $HOME/pe_nano5_data.tar ($(du -h "$HOME/pe_nano5_data.tar" | cut -f1))"
else
  acct=${1:?usage: bash quota_sweep/nano5_copy_data.sh <nano5 account>  (or TAR=1)}
  rsync -avR --partial --progress "${files[@]}" "$acct@nano5.nchc.org.tw:/work/$acct/PE-metadata/"
fi
