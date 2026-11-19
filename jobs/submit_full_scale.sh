#!/bin/bash
# Full-scale runs on SCC (14 GPU jobs). Dependencies enforced with -hold_jid.
# Usage: bash jobs/submit_full_scale.sh [model] [out_dir]
# Optional env:
#   QSUB_EXTRA  extra qsub flags, override the .qsub defaults (e.g. "-l gpu_memory=80G -l h_rt=24:00:00" for 32B)
#   COMMON      extra args for every step (e.g. "--seed 1" for a replication with new layouts)
#   E4_EXTRA    extra args for E4 only (e.g. "--layer_stride 2" for 32B)
#   PREFIX      job-name prefix so several models can be in the queue at once (default: model tag)
set -e
MODEL=${1:-qwen3vl-8b}
OUT=${2:-results_full}
Q=jobs/scc_experiment.qsub
PREFIX=${PREFIX:-${MODEL#qwen3vl-}}
mkdir -p logs/env
sub() {  # name, hold, step, args...
  # ARGS travel in a per-job env file (qsub -v splits on commas / can mangle spaces).
  local name=${PREFIX}_$1 hold=$2 step=$3; shift 3
  local h=""; [ -n "$hold" ] && h="-hold_jid ${PREFIX}_$hold"
  local envf="$PWD/logs/env/$name.env"
  printf 'STEP=%q\nMODEL=%q\nOUT=%q\nARGS=%q\n' "$step" "$MODEL" "$OUT" "$* ${COMMON:-}" > "$envf"
  qsub -terse -N "$name" $h ${QSUB_EXTRA:-} -v ENVFILE="$envf",STEP="$step",MODEL="$MODEL",OUT="$OUT" $Q
}
# E0 behaviour: 500 minimal pairs per cue (+ no-cue prior)
sub e0 "" e0_baseline --n 500 --n_generate 50
# E1 object spatial IDs (Kang Fig. 5 replication): all 56 ordered pairs of 8 shapes
sub e1obj "" e1_extract_ids --condition object --n_pairs 56
# E2 deictic IDs: four cue conditions (with / without objects)
for c in hand arrow hand_empty arrow_empty; do
  sub e1$c "" e1_extract_ids --condition $c --n_pairs 56
done
# E3 cue swapping (Kang Fig. 4 analogue), every layer
sub e3arrow "" e3_swap --cue arrow --n 200
sub e3hand  "" e3_swap --cue hand  --n 300
# E4 steering with object-derived IDs (Kang Fig. 6 analogue), alpha sweep
for a in 2 5 10; do
  sub e4hand$a  e1obj e4_steer --cue hand  --n 150 --alpha $a --id_source $OUT/$MODEL/ids/object.pt ${E4_EXTRA:-}
  sub e4arrow$a e1obj e4_steer --cue arrow --n 150 --alpha $a --id_source $OUT/$MODEL/ids/object.pt ${E4_EXTRA:-}
done
echo "submitted; E2 analysis (CPU) after the e1* jobs finish:"
echo "  python experiments/e2_compare_ids.py --ids_dir $OUT/$MODEL/ids --n_perm 200"
