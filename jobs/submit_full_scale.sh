#!/bin/bash
# Full-scale runs on SCC. Dependencies enforced with -hold_jid.
# Usage: bash jobs/submit_full_scale.sh [qwen3vl-8b] [results_full]
set -e
MODEL=${1:-qwen3vl-8b}
OUT=${2:-results_full}
Q=jobs/scc_experiment.qsub
sub() {  # name, hold, step, args...
  local name=$1 hold=$2 step=$3; shift 3
  local h=""; [ -n "$hold" ] && h="-hold_jid $hold"
  qsub -terse -N "$name" $h -v STEP="$step",MODEL="$MODEL",OUT="$OUT",ARGS="$*" $Q
}
# E0 behaviour: 500 minimal pairs per cue (+ no-cue prior)
sub e0 "" e0_baseline --n 500 --n_generate 50
# E1 object spatial IDs (Kang Fig. 5 replication): all 56 ordered pairs of 8 shapes
sub e1obj "" e1_extract_ids --condition object --n_pairs 56
# E2 deictic IDs: four conditions, plus an origin-jitter robustness run for the hand
for c in hand arrow hand_empty arrow_empty; do
  sub e1$c "" e1_extract_ids --condition $c --n_pairs 56
done
# E3 cue swapping (Kang Fig. 4 analogue), every layer
sub e3arrow "" e3_swap --cue arrow --n 200
sub e3hand  "" e3_swap --cue hand  --n 300
# E4 steering with object-derived IDs (Kang Fig. 6 analogue), alpha sweep
for a in 2 5 10; do
  sub e4hand$a  e1obj e4_steer --cue hand  --n 150 --alpha $a --id_source $OUT/$MODEL/ids/object.pt
  sub e4arrow$a e1obj e4_steer --cue arrow --n 150 --alpha $a --id_source $OUT/$MODEL/ids/object.pt
done
echo "submitted; E2 analysis (CPU) after the e1* jobs finish:"
echo "  python experiments/e2_compare_ids.py --ids_dir $OUT/$MODEL/ids"
