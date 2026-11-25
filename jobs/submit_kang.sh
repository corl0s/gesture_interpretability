#!/bin/bash
# Part A: exact Kang et al. spatial-ID replication (K1), chunked over the 90 object pairs, then the
# CPU analysis (universal IDs, axes, grid R^2 with bootstrap CI + permutation null).
# Usage: bash jobs/submit_kang.sh [model ...]     (default: qwen25vl-7b qwen3vl-8b)
# Env: OUT (results_kang), OBJ (Kang's Objaverse renders), CHUNK (pairs per job, default 10),
#      QSUB_EXTRA (e.g. "-l gpu_memory=80G" for 32B)
set -e
OUT=${OUT:-results_kang}
OBJ=${OBJ:-/projectnb/bu-mlb/vishnuav/deixis/data/kang/objects}
CHUNK=${CHUNK:-10}
Q=jobs/scc_experiment.qsub
mkdir -p logs/env
[ -f "$OBJ/backpack.png" ] || { echo "Kang objects missing in $OBJ"; exit 1; }
sub() {  # name, hold, model, xq, step, args...
  local name=$1 hold=$2 model=$3 xq=$4 step=$5; shift 5
  local h=""; [ -n "$hold" ] && h="-hold_jid $hold"
  local envf="$PWD/logs/env/$name.env"
  printf 'STEP=%q\nMODEL=%q\nOUT=%q\nARGS=%q\n' "$step" "$model" "$OUT" "$*" > "$envf"
  qsub -terse -N "$name" $h ${QSUB_EXTRA:-} $xq -v ENVFILE="$envf",STEP="$step",MODEL="$model",OUT="$OUT" $Q
}
for m in ${@:-qwen25vl-7b qwen3vl-8b}; do
  t=k${m#qwen}; t=${t//./}; t=${t//-/}
  names=()
  for ((s=0; s<90; s+=CHUNK)); do
    e=$((s+CHUNK)); [ $e -gt 90 ] && e=90
    n=${t}_k1_$s
    sub $n "" $m "" k1_kang_ids --kang_obj_dir $OBJ --pair_start $s --pair_end $e
    names+=($n)
  done
  hold=$(IFS=,; echo "${names[*]}")
  # analysis on CPU: no GPU request (override the qsub's gpu flags)
  envf="$PWD/logs/env/${t}_k1an.env"
  printf 'STEP=%q\nMODEL=%q\nOUT=%q\nARGS=%q\n' k1_analyse "$m" "$OUT" "--ids_dir $OUT/$m/kang_ids" > "$envf"
  echo "analysis env: $envf (submit as a CPU job holding on: $hold)"
done
