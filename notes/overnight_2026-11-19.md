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

## E3: activation swapping, image -> text hand-off (correct-both pairs)
| model | pairs (hand / arrow) | image/text crossover (ld_shift) | last picks up | relative depth of crossover |
|---|---|---|---|---|
| 2B (28 L) | 171 / 193 | L14 | L17–19 | ~0.52 |
| 8B (36 L) | 276 / 170 | L19–20 | L23–24 | ~0.54 |
- The same three stages, image -> question text -> answer, at the same relative depth in 2B and 8B, for both cues.
- How much the cue's own patches carry early: 8B hand .22–.26, arrow .15–.18. 2B hand .38–.43, arrow .83–.92 at L0–2. The smaller model reads the cue more locally.
- Correction: the earlier claim that the arrow's patches carry ~0 (8B, 38 pairs) did not hold at scale.

## E4: steering with object location IDs at the answer position (8B)
Max ID−random flip gap (rows are cues, columns are steering strengths):
| | α=2 | α=5 |
|---|---|---|
| hand (254 / 255 samples) | .776 @ L17 | .526 @ L16 |
| arrow (230 / 226 samples) | .787 @ L17 | .637 @ L16 |
At α=2 both cues have a window of L15–23 (ID flips >= .66–.69, peak ~.91) while random stays <= .18 through L33. This overlaps the E3 hand-off. Larger α is generic damage and moves the window earlier. Steering at "at" or at the cue patches is weak (<= .27).

## Status / pending
- Pending: 4B triage (chk4b), the 4B suite (held), the 2B E2/E4/follows runs, all of 32B, 8B seed 1, and the follows-mode E3 (right vs wrong relay strength).
- Incidents: follows-mode E3 had a ZeroDivisionError on exact bf16 ties, fixed in ab5b174 (NaN + checkpointing). Two 8B follows runs were lost and resubmitted. SGE job names can't start with a digit (fixed in 80f4170).
