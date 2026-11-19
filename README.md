# Deictic grounding in VLMs: a Kang-et-al.-style mechanistic study

**Question.** When a VLM is asked what a hand (or gaze) points at, does the cue create a
*referent* inside the model, and in which representation? We test this the way Kang, Chen,
Gkioxari & Perona test spatial reasoning in *Linear Mechanisms for Spatiotemporal Reasoning
in Vision Language Models* (arXiv 2601.12626; the repo README says ICLR 2026, the paper text
itself does not). Their reference code is cloned in `external/linear-mech-vlms/` (read-only).

Kang et al. show that VLMs bind an object's location to its **object-word token** as a
linear **spatial ID** (Δ_L(i,j), eqs. 2–5). The IDs lie on an approximate grid spanned by two
axes (h_L, v_L). Swapping or steering these IDs changes left/right answers. Our hypothesis
follows from that:

> **H1 (shared code).** If a model resolves a pointing cue, the token that stands for the
> referent (`at` / last token of "Which object is the hand pointing at?") carries the *same*
> spatial ID as the pointed cell. Gesture failures are then failures to *write* that ID,
> or to *read* it.

## How Kang's experiments map to ours

| Kang et al. | Here | Evidence type | Script |
|---|---|---|---|
| Fig. 5 spatial-ID derivation (object b swept over a 4×4 grid, query "Is b left or right of a?", mean-centred object-word activations) | **E1** exact replication on our stimuli, Qwen3-VL | correlational | `e1_extract_ids.py --condition object` |
| — | **E2** *deictic IDs*: cue points at each of the 16 cells; activations at `hand`/`pointing`/`at`/last token, same centring. Also `*_empty` (cue, no objects) to separate *direction* from *referent* | correlational | `e1_extract_ids.py --condition hand\|arrow\|hand_empty\|arrow_empty`, then `e2_compare_ids.py` |
| Fig. 4 mirror swapping (swap token subsets between image and mirrored image; belief shift) | **E3** cue swapping on true minimal pairs (same scene, cue→A vs cue→B), by layer × subset (all image / cue patches / object patches / text / cue word / last) | causal (patching) | `e3_swap.py` |
| Fig. 6 arbitrary steering (add α‖x‖·(Δ(target)−Δ(other)), α=5) | **E4** steer a gesture answer from A to B with **object-derived** IDs (a transfer test of H1), with a random-direction control | causal (steering) | `e4_steer.py` |
| Fig. 8 ground-truth deviation diagnosis (encoder vs integration failure) | **E5** (todo): on failed trials, is the target-cell ID present at the referent slot and then overridden (prior/saliency, explanation 3), or never written (encoder or relay, explanations 1–2)? | correlational + causal | — |
| Table 2 / probes | none in Kang (they fit regressions, they do not run probes). Add layer-wise probes for cue direction at ViT / merger / LLM (todo) | correlational | — |

`e0_baseline.py` gives behaviour: two-way accuracy on minimal pairs, a no-cue prior, and free-form answers.

## Stimuli (`deixis/stimuli.py`)
- Kang's geometry: a 4×4 grid of 112 px cells, two objects, object b swept over all cells. Below the grid is a 128 px strip from which a cue points (448×576 image, i.e. 14×18 = 252 merged Qwen3-VL tokens).
- Cue **direction** is the only signal. Cue length is fixed (150 px), so the fingertip is never next to the target. The arrow uses exactly the same origin, angle and tip as the hand (the drawn-arrow control).
- Minimal pairs require ≥25° angular separation between target and distractor, seen from the cue origin.
- Objects: 8 procedural shapes, each name one token in Qwen3. Drop RGBA PNGs named `<object>.png` into `data/objects/` to use Kang's Objaverse renders instead.
- **Limitation:** the hand is a 2D cartoon. Swap in EgoPoint-style rendered or real hands before drawing claims about natural pointing.

## Instrumentation (`deixis/model.py`)
Raw PyTorch pre-hooks on `model.model.language_model.layers[L]`. `resid_pre[L]` is the input to layer L, and `L = n_layers` is the input to the final norm. Pre-hooks matter for Qwen3-VL, which adds **DeepStack** visual features to image positions after each of the first 3 layers. A pre-hook sees the stream after injection. Patching image tokens at L ≤ 3 is still partly undone by the receiving run's own DeepStack features (the shaded band in the plots). Answers are scored by the log-prob of the answer's first token, summed over case and space variants.

## Run
```bash
# environment (Python 3.11, CUDA torch)
uv venv .venv --python 3.11
uv pip install -p .venv torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -p .venv -r requirements.txt
PY=.venv/Scripts/python   # Windows; .venv/bin/python elsewhere

$PY experiments/preview_stimuli.py                       # look at the stimuli first
$PY experiments/e0_baseline.py --n 100
$PY experiments/e1_extract_ids.py --condition object --n_pairs 20
for c in hand arrow hand_empty arrow_empty; do $PY experiments/e1_extract_ids.py --condition $c --n_pairs 20; done
$PY experiments/e2_compare_ids.py --ids_dir results/qwen3vl-2b/ids
$PY experiments/e3_swap.py --cue arrow --n 40     # arrow first: likely the model's easiest cue
$PY experiments/e3_swap.py --cue hand  --n 40
$PY experiments/e4_steer.py --cue hand --id_source results/qwen3vl-2b/ids/object.pt
```
Models: `--model qwen3vl-2b` (fits the 6 GB laptop GPU in bf16), `qwen3vl-8b` (needs a ≥24 GB GPU, e.g. on the cluster).

## Order of work / decision points
1. E0: if hand accuracy is near chance while arrow accuracy is high, the *directional extraction gap* (explanation 5) is already visible. E3/E4 then need the arrow condition, or easy hand cases, as the "clean" run.
2. E1 must reproduce Kang's grid on Qwen3-VL before E2 means anything (sanity check).
3. E2 asks whether a code exists at the referent slot. E3 asks where the cue information flows. E4 asks whether the referent slot causally uses the object spatial-ID code.
