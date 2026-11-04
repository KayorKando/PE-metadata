#!/usr/bin/env bash
# Submit PE runs on NCHC Nano5 as one Slurm job array (one run = one array task = one H100).
# Run on the login node from the repo root. It only checks inputs and calls sbatch once.
#   LAMS="0" SEEDS="0" VOTE_EMBS="ft" bash quota_sweep/nano5_pe_submit.sh        # one run
#   LAMS="0 0.5 1" SEEDS="0 1" VOTE_EMBS="ft" AFTER=<jobid> bash quota_sweep/nano5_pe_submit.sh
# Settings: LAMS, SEEDS, VOTE_EMBS (t5 and/or ft), ANNOTATOR (classifier), EPS (4.0), TIME per run (02:00:00;
# a run took ~45 min on the lab GPU), MAX_PAR (4 runs at a time), AFTER (job id that must finish OK first,
# e.g. the nano5_check job that trains the ft embedder), PARTITION (normal), EXTRA (more quota_pe.py flags).
# Runs land in $PE_DATA/quota_sweep/eps$EPS/<annotator>[-ft]/lam*/; lambda = 0 goes to maple[-ft]/ (the shared baseline)
# (same layout as the lab sweep, so quota_sweep/summarize.py works). Finished runs are skipped,
# unfinished ones resume after their last round, so a TIMEOUT just needs a resubmit.
set -eo pipefail
LAMS=${LAMS:-0}
SEEDS=${SEEDS:-0}
VOTE_EMBS=${VOTE_EMBS:-ft}
ANNOTATOR=${ANNOTATOR:-classifier}
EPS=${EPS:-4.0}
TIME=${TIME:-02:00:00}
MAX_PAR=${MAX_PAR:-4}
AFTER=${AFTER:-}
PARTITION=${PARTITION:-normal}
PE_DATA=${NANO5_DATA:-/work/$USER/PE-metadata}
[ -f quota_sweep/nano5_pe.sbatch ] || { echo "run from the repo root"; exit 1; }
for f in round0/eps$EPS/pool.csv quota_sweep/test_ref_qwen.csv MAPLE/biorxiv/biorxiv_train_metadata.csv; do
  [ -s "$PE_DATA/$f" ] || { echo "missing $PE_DATA/$f (see quota_sweep/nano5_pe_setup.sh)"; exit 1; }
done
[ "$ANNOTATOR" != classifier ] || [ -s "$PE_DATA/reannotate/eps$EPS/best_annotator_nano5.json" ] || \
  { echo "missing best_annotator_nano5.json: rerun quota_sweep/nano5_pe_setup.sh after copying the spec"; exit 1; }
[[ " $VOTE_EMBS " != *" ft "* ]] || [ -n "$AFTER" ] || [ -s "$PE_DATA/vote_embedding/eps$EPS/normsoftmax/heads.pt" ] || \
  { echo "no ft embedder yet: sbatch vote_embedding/nano5_check.sbatch first, then pass its id as AFTER=<jobid>"; exit 1; }
n=0; for v in $VOTE_EMBS; do for s in $SEEDS; do for l in $LAMS; do n=$((n + 1)); done; done; done
echo "$n run(s): VOTE_EMBS=[$VOTE_EMBS] SEEDS=[$SEEDS] LAMS=[$LAMS] ANNOTATOR=$ANNOTATOR; 1 GPU each, limit $TIME each," \
  "at most $MAX_PAR at a time${AFTER:+, after job $AFTER}; expect ~$(( (n * 3 + 3) / 4 )) GPU-hours (~45 min per run)"
dep=(); [ -z "$AFTER" ] || dep=(--dependency=afterok:"$AFTER")
export LAMS SEEDS VOTE_EMBS ANNOTATOR EPS EXTRA=${EXTRA:-} NANO5_DATA=$PE_DATA   # sbatch passes the environment on
sbatch --array=0-$((n - 1))%"$MAX_PAR" --time="$TIME" --partition="$PARTITION" "${dep[@]}" quota_sweep/nano5_pe.sbatch
