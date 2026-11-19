#!/usr/bin/env bash
# Submit PE runs on NCHC Nano5, several runs per Slurm job (one H100 per run, all runs of a job in parallel).
# Nano5 lets each user have only 2 jobs RUNNING at a time (any number pending), so packing runs into
# multi-GPU jobs is what makes a big grid finish in reasonable wall time.
# Run on the login node from the repo root. It only checks inputs and calls sbatch (once per job).
#   GROUP_BY=lam VOTE_EMBS=t5 LAMS="0 0.25 0.5 0.75 1" SEEDS="0 1 2 3 4" bash quota_sweep/nano5_pe_submit.sh
#       -> 5 jobs (one per lambda), 5 GPUs each (one per seed)
#   GROUP_BY=all VOTE_EMBS=ft LAMS="0 0.5 1" SEEDS="0 1" AFTER=<jobid> bash quota_sweep/nano5_pe_submit.sh
#       -> 1 job with all 6 runs, 6 GPUs, starts after job <jobid> (e.g. nano5_check, which trains the embedder)
# GROUP_BY: run (1 run per job), lam (1 job per vote embedding x lambda), all (1 job per vote embedding).
# Other settings: ANNOTATOR (classifier), EPS (4.0), TIME per job (02:00:00; a run took ~45 min on the lab GPU,
# runs of a job go in parallel), CPUS_PER_RUN (8), MEM_PER_RUN (64, GB), PARTITION (normal), EXTRA (quota_pe.py flags).
# VOTE_EMBS: t5, ft, fts<N> (ft embedder seed N), ftt5 (fine-tuned sentence-t5); see nano5_lib.sh.
# OpenReview: export PE_DATASET=openreview first (data root /work/$USER/PE-openreview, see openreview/README.md); use
# ANNOTATOR=qwen for lambda > 0 (Qwen labels in the loop, no classifier). AFTER may list several jobs: AFTER=<id1>:<id2>.
# Runs land in $PE_DATA/quota_sweep/eps$EPS/<annotator>[-<vemb>]/lam*/, lambda = 0 in maple[-<vemb>]/. Finished runs
# (DONE) are left out when the jobs are built, unfinished ones resume after their last round: after a
# TIMEOUT or a failure, rerun the same command and only the missing runs are submitted.
set -eo pipefail
LAMS=${LAMS:-0}
SEEDS=${SEEDS:-0}
VOTE_EMBS=${VOTE_EMBS:-ft}
GROUP_BY=${GROUP_BY:-run}
ANNOTATOR=${ANNOTATOR:-classifier}
EPS=${EPS:-4.0}
TIME=${TIME:-02:00:00}
CPUS_PER_RUN=${CPUS_PER_RUN:-8}
MEM_PER_RUN=${MEM_PER_RUN:-64}
AFTER=${AFTER:-}
PARTITION=${PARTITION:-normal}
[ -f quota_sweep/nano5_pe.sbatch ] || { echo "run from the repo root"; exit 1; }
source quota_sweep/nano5_lib.sh
export PE_DATASET=${PE_DATASET:-biorxiv}
PE_DATA=$(nano5_data)
case $GROUP_BY in run|lam|all) ;; *) echo "GROUP_BY must be run, lam or all"; exit 1;; esac
# with AFTER, the prep jobs (nano5_prep_eps.sbatch) may still be making these
from_prep="round0/eps$EPS/pool.csv"
[ "$PE_DATASET" = biorxiv ] || from_prep="$from_prep quota_sweep/test_ref_qwen.csv blind.json"
need="round0/eps$EPS/pool.csv quota_sweep/test_ref_qwen.csv $(private_csv)"
[ "$PE_DATASET" = biorxiv ] || need="$need blind.json"
for f in $need; do
  [ -n "$AFTER" ] && [[ " $from_prep " == *" $f "* ]] && continue
  [ -s "$PE_DATA/$f" ] || { echo "missing $PE_DATA/$f (see quota_sweep/nano5_pe_setup.sh; OpenReview: openreview/README.md)"; exit 1; }
done
nonzero=$(awk -v ls="$LAMS" 'BEGIN { n = split(ls, a, " "); for (i = 1; i <= n; i++) if (a[i] + 0 != 0) { print 1; exit } }')
[ "$ANNOTATOR" != classifier ] || [ -z "$nonzero" ] || [ -s "$PE_DATA/reannotate/eps$EPS/best_annotator_nano5.json" ] || \
  { echo "missing best_annotator_nano5.json: rerun quota_sweep/nano5_pe_setup.sh after copying the spec"; exit 1; }
for v in $VOTE_EMBS; do
  vemb_dir "$v" || exit 1
  [ -z "$VEMB_DIR" ] || [ -n "$AFTER" ] || [ -s "$PE_DATA/vote_embedding/eps$EPS/$VEMB_DIR/heads.pt" ] || \
    { echo "no $v embedder yet ($PE_DATA/vote_embedding/eps$EPS/$VEMB_DIR): train it first" \
           "(vote_embedding/nano5_train.sbatch), then pass its job id as AFTER=<jobid>"; exit 1; }
done
ROOT=$PE_DATA/quota_sweep/eps$EPS

# build the jobs: key -> "vemb seed lam;vemb seed lam;..." (finished runs left out)
declare -A GROUPS_; order=()
for v in $VOTE_EMBS; do for l in $LAMS; do for s in $SEEDS; do
  run_dir "$v" "$s" "$l"
  if [ -f "$ROOT/$RUN_DIR/DONE" ]; then echo "  done already: $RUN_DIR"; continue; fi
  case $GROUP_BY in run) key="$v-lam$l-s$s";; lam) key="$v-lam$l";; all) key="$v";; esac
  [ -n "${GROUPS_[$key]+x}" ] || order+=("$key")
  GROUPS_[$key]="${GROUPS_[$key]:+${GROUPS_[$key]};}$v $s $l"
done; done; done
[ ${#order[@]} -gt 0 ] || { echo "nothing to submit: all runs are done"; exit 0; }

dep=(); [ -z "$AFTER" ] || dep=(--dependency=afterok:"$AFTER")
export ANNOTATOR EPS EXTRA=${EXTRA:-} NANO5_DATA=$PE_DATA PE_DATASET     # sbatch passes the environment on
echo "dataset $PE_DATASET, data root $PE_DATA, eps $EPS"
total=0
for key in "${order[@]}"; do
  IFS=';' read -ra runs <<< "${GROUPS_[$key]}"
  n=${#runs[@]}
  [ "$n" -le 8 ] || { echo "job $key would need $n GPUs (> 8 per node): use a finer GROUP_BY"; exit 1; }
  total=$((total + n))
  name="pe-$key"; [ "$PE_DATASET" = biorxiv ] || name="$(job_prefix)pe$EPS-$key"     # e.g. or-pe4.0-t5-lam0
  echo "job $name: $n run(s) on $n GPU(s): ${GROUPS_[$key]}"
  RUNS="${GROUPS_[$key]}" sbatch --job-name="$name" --gres=gpu:"$n" --cpus-per-task=$((CPUS_PER_RUN * n)) \
    --mem=$((MEM_PER_RUN * n))G --time="$TIME" --partition="$PARTITION" "${dep[@]}" quota_sweep/nano5_pe.sbatch
done
echo "${#order[@]} job(s), $total run(s), limit $TIME per job${AFTER:+, after job $AFTER}; at most 2 of your jobs run at a time;" \
  "expect ~$(( (total * 3 + 3) / 4 )) GPU-hours (~45 min per run)"
