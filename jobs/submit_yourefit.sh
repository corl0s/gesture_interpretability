#!/bin/bash
# YouRefIt (real pointing) behaviour runs: R0 (grounding, 5 conditions) + R1 (two-choice same-scene pairs).
# Prereqs (run once on a login node, from the repo root, venv active):
#   export TORCH_HOME=$PWD/torch_home
#   python -c "from torchvision.models.detection import keypointrcnn_resnet50_fpn as k, KeypointRCNN_ResNet50_FPN_Weights as W; k(weights=W.COCO_V1)"
#   python experiments/r_pairs.py --out results_scaled          # pair lists (CPU, minutes)
# Usage: bash jobs/submit_yourefit.sh [models...]   (default: qwen3vl-2b qwen3vl-4b qwen3vl-8b qwen3vl-32b)
set -e
OUT=${OUT:-results_scaled}
Q=jobs/scc_experiment.qsub
PEOPLE=$OUT/yourefit/people_test.json
mkdir -p logs/env
sub() {  # name, hold, model, qsub-extra, step, args...
  local name=$1 hold=$2 model=$3 xq=$4 step=$5; shift 5
  local h=""; [ -n "$hold" ] && h="-hold_jid $hold"
  local envf="$PWD/logs/env/$name.env"
  printf 'STEP=%q\nMODEL=%q\nOUT=%q\nARGS=%q\n' "$step" "$model" "$OUT" "$*" > "$envf"
  qsub -terse -N "$name" $h $xq -v ENVFILE="$envf",STEP="$step",MODEL="$model",OUT="$OUT" $Q
}
# person boxes + arm keypoints for the test split (needed for the person-masked controls)
sub yr_people "" qwen3vl-2b "-l h_rt=02:00:00 -l gpu_memory=16G" r_people --splits test
MODELS=${@:-qwen3vl-2b qwen3vl-4b qwen3vl-8b qwen3vl-32b}
for m in $MODELS; do
  t=y${m#qwen3vl-}
  xq=""; [ "$m" = "qwen3vl-32b" ] && xq="-l gpu_memory=80G"
  sub ${t}_r0 yr_people $m "$xq" r0_yourefit --split test --people $PEOPLE
  sub ${t}_r1 yr_people $m "$xq" r1_yourefit_pairs --split test --n 1000 --people $PEOPLE
done
echo "submitted YouRefIt R0/R1 for: $MODELS"
