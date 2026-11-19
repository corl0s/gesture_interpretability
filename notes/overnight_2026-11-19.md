# Overnight results, 2026-11-19 (scaled sweep on BU SCC)

Source: reports from the SCC session (batchjob-gesture_interp). Raw files are on SCC under
`/projectnb/bu-mlb/vishnuav/deixis/results_scaled/<model>/` (8B 500-pair run: `results_full/`).
Stimuli: 10 shapes, 4x4 grid, cartoon hand or arrow. E1 is at Kang scale (90 pairs x 16 cells x 4 sizes).
Evidence type: E0 behavioural; E2 correlational; E3/E4 causal.

## E0: behaviour (2000 minimal pairs per cue)
| model | hand acc / both / follows | arrow acc / both / follows |
|---|---|---|
| 2B | .575 / .164 / .822 | .588 / .181 / .831 |
| 4B | .478 / .057 / .478 | .477 / .046 / .468 |
| 8B | .611 / .262 / .830 | .567 / .167 / .717 |
4B ignores the cue: its answers track the no-cue prior. The generations are sane shape names (sometimes "plus" for the cross, which isn't scored yet). A loader-vs-HF check (chk4b) is pending.
8B hand > arrow on both-correct: .262 vs .167, z ~ 7.

## E2: does pointing use the object location map? (8B, Kang scale)
- Object grid R^2 is .957 at L1 and .989 at L27.
- At the referent words, the hand matches the object map (nearest-ID above the 200-permutation null) at "at" in 29/36 layers and at "last" in 27/36. The empty-grid controls stay at 0–6 layers, so these slots encode the referent object, not just a direction.
- The cue words ("hand"/"arrow", "pointing") are above null even on an empty grid, so they encode the pointing direction.
- At the referent words the hand beats the arrow (27–29 layers vs 8–9).

2B (Kang scale) differs:
- The object grid is very clean (R^2 .992 at L12), and hand/"at" is above null on 17/28 layers (peak .5 at L12).
- But the empty-grid controls are NOT near null. arrow_empty/"at" is above null on 20 layers (R^2) and 11 (NN); hand_empty/"at" on 17 (R^2).
- So in 2B the referent word carries location even without objects, i.e. a direction or position code rather than an object-bound referent.
- This suggests scale-dependent processing: 2B's referent slot carries a direction/position code, while 8B's binds to the object. This is a hypothesis to test with 32B. (The E3 cue-patch "support" for it was a belief_shift artifact and has been withdrawn.)

## E3: activation swapping, image -> text hand-off (correct-both pairs)
| model | pairs (hand / arrow) | image/text crossover (ld_shift) | last picks up | relative depth of crossover |
|---|---|---|---|---|
| 2B (28 L) | 171 / 193 | L14 | L17–19 | ~0.52 |
| 8B (36 L) | 276 / 170 | L19–20 | L23–24 | ~0.54 |
- The same three stages, image -> question text -> answer, at the same relative depth in 2B and 8B, for both cues.
- Cue-patch and object-patch shares (ld_shift, early-layer plateau; these supersede the belief_shift numbers):
  | run | cue patches | object patches |
  |---|---|---|
  | 8B hand (276) | .39–.40 | .22–.23 |
  | 8B arrow (170) | .30–.31 | .19 |
  | 2B hand (171) | .31 | .17–.21 |
  | 2B arrow (193) | .32 | .08–.10 |
  Both fade to ~0 at the hand-off, in step with image_all. In 8B the hand's patches carry more than the arrow's (~.10, about 3 SE); in 2B they carry the same.
- RETRACTED: "2B reads the cue more locally (arrow .83–.92)". That was a belief_shift artifact. In ld_shift, 2B and 8B cue-patch shares are similar (.30–.40).
- RETRACTED: "the arrow's patches carry ~0 at 8B". That was belief_shift; in ld it's .17 (38 pairs) and .30 (170 pairs).
- belief_shift only tracks logP(A), so partial patches that move B can read as ~0, negative or >1. Use ld_shift.

### E3 follows mode: right vs half-wrong pairs (2B hand, 475 pairs; 108 both-right, 367 one-wrong)
"Neither" is empty by construction: the filter LD_x − LD_y > 0.5 excludes both-wrong pairs.
- **Same relay layers.** Both groups hand off image -> text at L14->15, and "last" rises over L16–19. Wrong answers are NOT caused by a missing or late relay. This also rules out the "arrives too late" idea.
- **The cue effect is smaller in absolute terms** in half-wrong pairs: median LD_x−LD_y is 3.1–3.5 nats vs 5.45.
- **The cue's own patches carry less** early on (L2–11: .16–.27 vs .33–.38), and the object patches a bit more (.20–.22 vs .16–.20).
- Revised reading: failures start with a weaker cue signal at the source (early cue-patch encoding), plus more pull from the scene/objects. That signal then passes through an intact relay. So the "relay is the bottleneck" claim needs refining: the relay route is the same, but what enters it is weaker. To check in 8B (whose follows runs are queued).

2B arrow follows (462 pairs; 114 both-right, 348 one-wrong):
- Same relay layers (L14->15) for right and half-wrong pairs, which replicates the hand.
- The absolute cue effect is again smaller in half-wrong pairs (median 3.75–4.1 vs 6.25 nats).
- BUT the cue-patch share is NOT lower in half-wrong arrow pairs (.30–.34 in both groups). So the "weaker cue encoding at the source" pattern is hand-only so far.
- Right pairs start sharing into text earlier (from L7), while half-wrong pairs switch abruptly at L15.
- Robust across cues: the relay timing is the same whether the answer is right or wrong; the absolute cue effect is smaller when wrong.
- Not robust: where the weakness comes from.

METRIC CORRECTION: the E3 img_cue / img_obj numbers quoted above for correct mode (8B hand .22–.26, arrow .15–.18; 2B arrow .83–.92) are belief_shift. In 2B follows mode with ld_shift, arrow img_cue is .18–.34. The two metrics diverge for small subsets, so the cue-patch comparisons must be redone in ld_shift (requested from SCC). The hand-off layers agree in both metrics.

## E4: steering with object location IDs at the answer position (8B)
Max ID−random flip gap (rows are cues, columns are steering strengths):
| | α=2 | α=5 |
|---|---|---|
| hand (254 / 255 samples) | .776 @ L17 | .526 @ L16 |
| arrow (230 / 226 samples) | .787 @ L17 | .637 @ L16 |
At α=2 both cues have a window of L15–23 (ID flips >= .66–.69, peak ~.91) while random stays <= .18 through L33. This overlaps the E3 hand-off. Larger α is generic damage and moves the window earlier. Steering at "at" or at the cue patches is weak (<= .27).

2B E4 (232–236 samples): max ID−random gap at "last" (rows are cues, columns are α):
| | α=2 | α=5 |
|---|---|---|
| hand | .595 @ L12 | .349 @ L11 |
| arrow | .550 @ L15 | .275 @ L11 |
- At α=2 the window is L11–18 (ID flips >= .59), i.e. ~0.39–0.64 of depth. 8B is L15–23 (~0.42–0.64), so the same relative position, overlapping 2B's E3 hand-off (L14–15) and its best object-grid layer (L12).
- 2B is more fragile: random flips are .15–.30 even at α=2, so the gaps are smaller than 8B's.
- In 2B, steering at "at" works earlier (L5–8: .29–.32 vs random ~.09) than in 8B (L16–19: .21–.26).

## Status / pending
- Pending: 4B triage (chk4b), the 4B suite (held), the 2B E2/E4/follows runs, all of 32B, 8B seed 1, and the follows-mode E3 (right vs wrong relay strength).
- Incidents: follows-mode E3 had a ZeroDivisionError on exact bf16 ties, fixed in ab5b174 (NaN + checkpointing). Two 8B follows runs were lost and resubmitted. SGE job names can't start with a digit (fixed in 80f4170).
