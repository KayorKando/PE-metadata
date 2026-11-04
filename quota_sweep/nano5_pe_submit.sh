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
# Runs land in $PE_DATA/quota_sweep/eps$EPS/<annotator>[-ft]/lam*/, lambda = 0 in maple[-ft]/. Finished runs
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
PE_DATA=${NANO5_DATA:-/work/$USER/PE-metadata}
[ -f quota_sweep/nano5_pe.sbatch ] || { echo "run from the repo root"; exit 1; }
case $GROUP_BY in run|lam|all) ;; *) echo "GROUP_BY must be run, lam or all"; exit 1;; esac
for f in round0/eps$EPS/pool.csv quota_sweep/test_ref_qwen.csv MAPLE/biorxiv/biorxiv_train_metadata.csv; do
  [ -s "$PE_DATA/$f" ] || { echo "missing $PE_DATA/$f (see quota_sweep/nano5_pe_setup.sh)"; exit 1; }
done
[ "$ANNOTATOR" != classifier ] || [ -s "$PE_DATA/reannotate/eps$EPS/best_annotator_nano5.json" ] || \
  { echo "missing best_annotator_nano5.json: rerun quota_sweep/nano5_pe_setup.sh after copying the spec"; exit 1; }
[[ " $VOTE_EMBS " != *" ft "* ]] || [ -n "$AFTER" ] || [ -s "$PE_DATA/vote_embedding/eps$EPS/normsoftmax/heads.pt" ] || \
  { echo "no ft embedder yet: sbatch vote_embedding/nano5_check.sbatch first, then pass its id as AFTER=<jobid>"; exit 1; }
source quota_sweep/nano5_lib.sh
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
export ANNOTATOR EPS EXTRA=${EXTRA:-} NANO5_DATA=$PE_DATA     # sbatch passes the environment on
total=0
for key in "${order[@]}"; do
  IFS=';' read -ra runs <<< "${GROUPS_[$key]}"
  n=${#runs[@]}
  [ "$n" -le 8 ] || { echo "job $key would need $n GPUs (> 8 per node): use a finer GROUP_BY"; exit 1; }
  total=$((total + n))
  echo "job pe-$key: $n run(s) on $n GPU(s): ${GROUPS_[$key]}"
  RUNS="${GROUPS_[$key]}" sbatch --job-name="pe-$key" --gres=gpu:"$n" --cpus-per-task=$((CPUS_PER_RUN * n)) \
    --mem=$((MEM_PER_RUN * n))G --time="$TIME" --partition="$PARTITION" "${dep[@]}" quota_sweep/nano5_pe.sbatch
done
echo "${#order[@]} job(s), $total run(s), limit $TIME per job${AFTER:+, after job $AFTER}; at most 2 of your jobs run at a time;" \
  "expect ~$(( (total * 3 + 3) / 4 )) GPU-hours (~45 min per run)"
