#!/bin/bash
# Submit the three extrapolated sizes, each with resources matched to its actual cost,
# and chain the aggregation so the table appears on its own when the last size lands.
#
#   cd /gscratch/<group>/<uwnetid>/Thermal-State-Project
#   ACCOUNT=<your-account> PARTITION=compute bash hpc/submit.sh
#
# The per-size numbers below come from timing one RK4 step at each size (2.1 s at
# L = 12, 49 s at L = 14 measured; ~880 s at L = 16 extrapolated) times the number of
# steps needed to reach 1 - F = 1e-4.  They are estimates -- the jobs checkpoint every
# 30 min, so one that runs over its limit is requeued rather than lost.
set -euo pipefail

ACCOUNT="${ACCOUNT:?set ACCOUNT=... (see: hyakalloc)}"
PARTITION="${PARTITION:-compute}"
REPO="${REPO:-$PWD}"
BACKEND="${BACKEND:-scipy}"
SIZES="${SIZES:-12 14}"           # add 16 once you have picked a backend for it

mkdir -p logs "$REPO/hpc/results"

declare -A IDX=( [12]=0 [14]=1 [16]=2 )
declare -A TIME=( [12]="8:00:00" [14]="72:00:00" [16]="72:00:00" )
declare -A MEM=(  [12]="32G"     [14]="64G"      [16]="180G" )
declare -A CPUS=( [12]="8"       [14]="16"       [16]="40" )

ids=()
for L in $SIZES; do
  extra=()
  if [ "$L" = "16" ] && [ "$BACKEND" = "gpu" ]; then
    extra=(--gpus=1)
  fi
  jid=$(sbatch --parsable \
      --account="$ACCOUNT" --partition="$PARTITION" \
      --array="${IDX[$L]}" \
      --time="${TIME[$L]}" --mem="${MEM[$L]}" --cpus-per-task="${CPUS[$L]}" \
      --job-name="relax-L$L" \
      "${extra[@]}" \
      --export=ALL,REPO="$REPO",BACKEND="$BACKEND" \
      "$REPO/hpc/relax.slurm")
  echo "L = $L  ->  job $jid   (${TIME[$L]}, ${MEM[$L]}, ${CPUS[$L]} cpus, $BACKEND)"
  ids+=("$jid")
done

dep=$(IFS=:; echo "${ids[*]}")
agg=$(sbatch --parsable \
    --account="$ACCOUNT" --partition="$PARTITION" \
    --job-name=relax-aggregate \
    --dependency="afterany:$dep" \
    --time=00:20:00 --mem=8G --cpus-per-task=2 \
    --output=logs/aggregate_%j.out \
    --wrap="source $REPO/hpc/env.sh && python $REPO/hpc/aggregate_relax.py --results $REPO/hpc/results")
echo "aggregate -> job $agg  (runs when the sizes finish; afterany, so partial results still get a table)"
echo
echo "watch with:  squeue -u \$USER"
echo "logs in:     $REPO/logs/"
