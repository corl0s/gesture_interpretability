#!/usr/bin/env bash
# Small-scale versions of E0-E4 for a 6 GB laptop GPU (Qwen3-VL-2B). ~30-40 min total.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
OUT=${OUT:-results_local}
M=${MODEL:-qwen3vl-2b}
export PYTHONIOENCODING=utf-8
$PY experiments/e0_baseline.py --model $M --out $OUT --n 60 --n_generate 10
$PY experiments/e1_extract_ids.py --model $M --out $OUT --condition object --n_pairs 10
for c in arrow hand arrow_empty hand_empty; do
  $PY experiments/e1_extract_ids.py --model $M --out $OUT --condition $c --n_pairs 10
done
$PY experiments/e2_compare_ids.py --ids_dir $OUT/$M/ids
$PY experiments/e2_compare_ids.py --ids_dir $OUT/$M/ids --conditions hand,arrow --correct_only || true
$PY experiments/e3_swap.py --model $M --out $OUT --cue arrow --n 20 --layer_stride 2
$PY experiments/e3_swap.py --model $M --out $OUT --cue hand  --n 30 --layer_stride 2
$PY experiments/e4_steer.py --model $M --out $OUT --cue arrow --n 15 --layer_stride 2 --id_source $OUT/$M/ids/object.pt
$PY experiments/e4_steer.py --model $M --out $OUT --cue hand  --n 15 --layer_stride 2 --id_source $OUT/$M/ids/object.pt
echo ALL_DONE
