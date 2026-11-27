#!/bin/bash
# Part B1: gesture-ID extraction on Kang-style scenes with real hand cut-outs.
# Usage:
#   MODE=pilot bash jobs/submit_gesture.sh [model ...]   behaviour only, scenes 0-40, one job per model
#   MODE=full  bash jobs/submit_gesture.sh [model ...]   activations, NSCENES scenes in chunks of CHUNK
#   MODE=b2    bash jobs/submit_gesture.sh [model ...]   B2 positions (foil, object/cue/tip image tokens),
#                                                        every 2nd layer; NSCENES (100) in chunks of CHUNK (25)
#   MODE=swap  bash jobs/submit_gesture.sh [model ...]   mirror swap, one job per cue in CUES
#                                                        (photo,cartoon,arrow), NPAIRS (150) pairs each
#   MODE=text  bash jobs/submit_gesture.sh [model ...]   swap of text pieces (names, prefix, last) + steering
#                                                        with the name-tag direction, scenes 100-300 (the
#                                                        directions come from scenes 0-99); needs
#                                                        $OUT/<model>/gesture_b2/name_tag_dirs.pt from b2_analyse
#                                                        (hold with QSUB_EXTRA="-hold_jid <b2_analyse job>")
#   MODE=oracle bash jobs/submit_gesture.sh [model ...]  inject the correct name tag (oracle), the reverse
#                                                        and equal-norm noise at L20/22/24, scenes 100-200
#   MODE=tokens bash jobs/submit_gesture.sh [model ...]  per-token swap map of the text at L16-26, NPAIRS_TOK
#                                                        (100) pairs per cue, scenes 100-300
# Env: OUT (results_gesture), OBJ (Kang renders), HANDS (cut-out dir with hands.json),
#      EXCLUDE_HANDS (comma list), NSCENES (300), CHUNK (30), QSUB_EXTRA.
# The CPU analysis (b1_analyse.py --ids_dir $OUT/<model>/gesture_ids [--spatial_ids ...]) is printed,
# not submitted, like submit_kang.sh.
set -e
MODE=${MODE:-pilot}
OUT=${OUT:-results_gesture}
OBJ=${OBJ:-/projectnb/bu-mlb/vishnuav/deixis/data/kang/objects}
HANDS=${HANDS:-/projectnb/bu-mlb/vishnuav/deixis/data/hands_yrf_preview}
NSCENES=${NSCENES:-300}
CHUNK=${CHUNK:-30}
Q=jobs/scc_experiment.qsub
mkdir -p logs/env
[ -f "$HANDS/hands.json" ] || { echo "no hands.json in $HANDS"; exit 1; }
[ -f "$OBJ/backpack.png" ] || { echo "Kang objects missing in $OBJ"; exit 1; }
COMMON="--kang_obj_dir $OBJ --hand_dir $HANDS"
[ -n "$EXCLUDE_HANDS" ] && COMMON="$COMMON --exclude_hands $EXCLUDE_HANDS"
sub() {  # name, model, step, args...
  local name=$1 model=$2 step=$3; shift 3
  local envf="$PWD/logs/env/$name.env"
  printf 'STEP=%q\nMODEL=%q\nOUT=%q\nARGS=%q\n' "$step" "$model" "$OUT" "$*" > "$envf"
  qsub -terse -N "$name" ${QSUB_EXTRA:-} -v ENVFILE="$envf",STEP="$step",MODEL="$model",OUT="$OUT" $Q
}
for m in ${@:-qwen3vl-8b}; do
  t=g${m#qwen}; t=${t//./}; t=${t//-/}
  if [ "$MODE" = pilot ]; then
    sub ${t}_pilot $m b1_gesture_ids $COMMON --scene_start 0 --scene_end 40 --behaviour_only
  elif [ "$MODE" = b2 ]; then
    N=${NSCENES_B2:-100}; C=${CHUNK_B2:-25}; names=()
    for ((s=0; s<N; s+=C)); do
      e=$((s+C)); [ $e -gt $N ] && e=$N
      sub ${t}_b2_$s $m b1_gesture_ids $COMMON --scene_start $s --scene_end $e --b2 --layer_stride 2
      names+=(${t}_b2_$s)
    done
    echo "analysis: python experiments/b2_analyse.py --ids_dir $OUT/$m/gesture_b2  (hold on: $(IFS=,; echo "${names[*]}"))"
  elif [ "$MODE" = text ]; then
    for c in $(echo ${CUES:-photo,cartoon,arrow} | tr , ' '); do
      sub ${t}_tx_$c $m b2_swap $COMMON --cue $c --n_pairs ${NPAIRS:-150} --layer_stride 2 --scene_start 100         --subsets names,name_A,name_B,prefix,last,text_all --steer_dirs $OUT/$m/gesture_b2/name_tag_dirs.pt --tag _text
    done
  elif [ "$MODE" = oracle ]; then
    for c in $(echo ${CUES:-photo,cartoon,arrow} | tr , ' '); do
      sub ${t}_or_$c $m b3_oracle $COMMON --cue $c --scene_start 100 --scene_end ${ORACLE_END:-200} --steer_dirs $OUT/$m/gesture_b2/name_tag_dirs.pt --layers ${ORACLE_LAYERS:-20,22,24}
    done
  elif [ "$MODE" = tokens ]; then
    for c in $(echo ${CUES:-photo,cartoon,arrow} | tr , ' '); do
      sub ${t}_tk_$c $m b2_swap $COMMON --cue $c --n_pairs ${NPAIRS_TOK:-100} --layer_stride 2 --scene_start 100 --subsets last --token_layers 16,18,20,22,24,26 --tag _tok
    done
  elif [ "$MODE" = swap ]; then
    for c in $(echo ${CUES:-photo,cartoon,arrow} | tr , ' '); do
      sub ${t}_sw_$c $m b2_swap $COMMON --cue $c --n_pairs ${NPAIRS:-150} --layer_stride 2
    done
  else
    names=()
    for ((s=0; s<NSCENES; s+=CHUNK)); do
      e=$((s+CHUNK)); [ $e -gt $NSCENES ] && e=$NSCENES
      sub ${t}_b1_$s $m b1_gesture_ids $COMMON --scene_start $s --scene_end $e
      names+=(${t}_b1_$s)
    done
    echo "analysis: python experiments/b1_analyse.py --ids_dir $OUT/$m/gesture_ids [--spatial_ids results_kang/$m/kang_ids/universal.pt]  (hold on: $(IFS=,; echo "${names[*]}"))"
  fi
done
