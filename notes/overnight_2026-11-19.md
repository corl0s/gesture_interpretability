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
| 32B | .683 / .377 / .945 | .666 / .334 / .968 |

32B follows the cue on 95–97% of pairs but is right only 67–68% of the time. The gap between seeing the cue and acting on it persists even in the strongest model.
4B VERDICT (chk4b, CPU fp32): our loader matches HF exactly (714/714 params identical, same generations). So 4B is genuinely cue-blind in short answers, YET when asked to describe the image it correctly says the hand "points diagonally upward and to the left toward a blue circle". It perceives the direction but doesn't use it to answer. This is the sharpest case of perceive-but-don't-use.
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

32B E2 (Kang scale; 64 layers): object grid best R^2 .988 at L56.
Hand/"at" nearest-ID is above null on 58/64 layers (peak .688 = 11/16 cells at L9). hand_empty/"at" is at 11/64, hand_empty/"last" at 19/64.
Scale trend for hand/"at" (NN layers above null, with objects vs empty):
| model | with objects | empty grid |
|---|---|---|
| 2B | 61% | 21% |
| 8B | 81% | 17% |
| 32B | 91% | 17% |
- Referent binding at "at" grows with scale. NOTE: the earlier "2B = direction code" reading rested on arrow_empty and R^2; on hand NN, 2B also separates, just more weakly.
- Arrow: a strong direction code in 32B (empty-grid R^2 above null on 59–60/64 layers), with moderate referent NN (29/64).

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

8B follows mode (hand 492 pairs: 159 both / 333 half-wrong; arrow 383: 93 / 290):
- The relay layers are the same in all four groups (image -> text crossing at L19–21, last rising at L23–24). This replicates 2B.
- The absolute cue effect is much smaller in half-wrong pairs: median LD gap hand 11.9 vs 6.2–6.5, arrow 12.6 vs 4.2–4.9 nats.
- The cue-patch share is NOT consistent across runs. In half-wrong pairs it is: 2B hand LOWER, 2B arrow SAME, 8B hand HIGHER (.52 vs .42), 8B arrow LOWER (.20–.28 vs .30–.33).
- 8B arrow half-wrong pairs: the hand-off is later and incomplete (text_all slightly negative at L14–17; .27–.43 vs .47–.66 at L19–21; a residual image share stays late). This is the only condition with a relay-timing deficit.
SUMMARY ACROSS 4 RUNS: robust = same relay route plus a smaller cue effect when wrong. Not robust = where the weakness originates. The 8B arrow shows a partial relay delay.

METRIC CORRECTION: the E3 img_cue / img_obj numbers quoted above for correct mode (8B hand .22–.26, arrow .15–.18; 2B arrow .83–.92) are belief_shift. In 2B follows mode with ld_shift, arrow img_cue is .18–.34. The two metrics diverge for small subsets, so the cue-patch comparisons must be redone in ld_shift (requested from SCC). The hand-off layers agree in both metrics.

## E4: steering with object location IDs at the answer position (8B)
Max ID−random flip gap (rows are cues, columns are steering strengths):
| | α=2 | α=5 |
|---|---|---|
| hand (254 / 255 samples) | .776 @ L17 | .526 @ L16 |
| arrow (230 / 226 samples) | .787 @ L17 | .637 @ L16 |
At α=2 both cues have a window of L15–23 (ID flips >= .66–.69, peak ~.91) while random stays <= .18 through L33. This overlaps the E3 hand-off. Larger α is generic damage and moves the window earlier. Steering at "at" or at the cue patches is weak (<= .27).

8B SEED 1 (new layouts) reproduces E4 to within ~0.03: hand α2 .784 @ L17 (seed 0 .776), arrow α2 .805 @ L17 (.787), hand α5 .557 @ L16 (.526). The windows are layout-independent.

2B E4 (232–236 samples): max ID−random gap at "last" (rows are cues, columns are α):
| | α=2 | α=5 |
|---|---|---|
| hand | .595 @ L12 | .349 @ L11 |
| arrow | .550 @ L15 | .275 @ L11 |
- At α=2 the window is L11–18 (ID flips >= .59), i.e. ~0.39–0.64 of depth. 8B is L15–23 (~0.42–0.64), so the same relative position, overlapping 2B's E3 hand-off (L14–15) and its best object-grid layer (L12).
- 2B is more fragile: random flips are .15–.30 even at α=2, so the gaps are smaller than 8B's.
- In 2B, steering at "at" works earlier (L5–8: .29–.32 vs random ~.09) than in 8B (L16–19: .21–.26).

## 32B E3 arrow (342 pairs) — CORRECTS the "same relative depth" claim
image->text crossing at ~L47.5 of 64 = 0.74 of depth (2B 0.52, 8B 0.54); last rises L50–52 (~0.8).
So the relay position is NOT a constant fraction of depth: 32B keeps the cue in image tokens much longer.
Cue-patch share (ld) .40 flat over L2–33 (8B arrow .30), object patches ~.10 (8B .19). 32B hand E3 (375 pairs): crossing L46–47 (~0.73 depth), same as arrow; hand patches alone carry .70 (L2–35) vs arrow .40 (8B hand .40); object patches .21–.24. At 32B the hand's information is strongly localised in the hand patches and differs from the arrow.
8B R0 (YouRefIt, 1251 frames): lang .778 | gesture .265 | both .771 | gesture_mask .075 | lang_mask .747. Adding the gesture never helps (2B: .612 vs .640). Gesture-prompt predictions land closer to the person than the targets are (proximity bias; recompute with the fixed arm).

## YouRefIt (real photos)
8B R1 (1000 same-scene pairs, 2-choice): acc .726, both .49, follows .824; person-masked: acc .595, follows .692
 -> non-gesture cues (where the person stands, mask leaks, mask position) explain part; gesture-specific share = .69 -> .82.
2B R0 (1251 frames, centre-point grounding): lang .640 | gesture-only .222 | both .612 | gesture_mask .070 | lang_mask .621
 -> the gesture carries real but weak information (3x the masked floor); adding it to the sentence slightly HURTS.

## Status / pending
- Pending: 4B triage (chk4b), the 4B suite (held), the 2B E2/E4/follows runs, all of 32B, 8B seed 1, and the follows-mode E3 (right vs wrong relay strength).
- Incidents: follows-mode E3 had a ZeroDivisionError on exact bf16 ties, fixed in ab5b174 (NaN + checkpointing). Two 8B follows runs were lost and resubmitted. SGE job names can't start with a digit (fixed in 80f4170).

## K1: exact Kang spatial-ID replication (Fig 5), 2026-11-25 (SCC, patch applied uncommitted)
90 pairs x 960 images, black canvas, query "Is {b} to the left or right of {a}?", read at b's last sub-token.
Grid R^2 (affine fit of 2-D axis projection to true cells) [95% bootstrap CI over pairs], shuffled-cell null95:
- qwen25vl-7b: best L9 @224 R2 .990 [.988,.991] null95 .61, var_expl .83; L12 best at sizes 74/124 (.985/.986).
- qwen3vl-8b: best L14 at all sizes, R2 .994-.995 [~.993,.995], null95 .46-.61, var_expl .81-.89.
- R2 is ~.97 already at L1 and stays >= .89 to the last layer: the grid is readable at almost every depth,
  so geometry alone does not pick a "binding layer" -> causal tests (Fig 2 steering, Fig 4 mirror swap) are needed.
- behaviour acc field INVALID for qwen25vl-7b (first-token scoring; it answers "The ... is to the right ...");
  qwen3vl-8b .963 unverified. Separate forced-prefix behaviour pass to be written.
- Env: transformers 4.57.6, torch 2.8.0+cu128, fast image processor (default).

## B1 gesture pilot (qwen3vl-8b, 40 scenes, 2312 images, behaviour only), 2026-11-25
Kang-style scenes, 5 objects, real YouRefIt hand cut-outs (9 hands), LEN 340, 10 origins, forced prefix.
acc            photo  cartoon arrow  empty
all            .543   .554    .689   .460
tip-nearest!=T .473   .470    .601   .468  (n=404/cue)
tip-nearest=T  .707   .747    .891   .443  (n=174)
- Hands (real = cartoon) are at chance when the fingertip-nearest object is not the target and below chance
  at large fingertip gaps (.29-.37 beyond 500 px): the model reads a hand as "what is near the fingertip".
  The arrow keeps some direction signal (.60). Realism of the hand is not the bottleneck at 8B.
- Hands add answer-order bias (first-named picked .64 photo / .62 cartoon vs .48 arrow).
- No hand is an outlier (.46-.62). Hand #7 excluded for the full run anyway (thin finger at LEN 340).
- Analysis now also tests pointed cell vs fingertip-nearest cell in the activations (planted-data check OK);
  grid R2 alone cannot separate the two (0.95 for a planted nearest-object code).

## B1 full run (qwen3vl-8b, 300 scenes, 17,260 images, acts at cue/tgt/ans), 2026-11-25
SCC: results_gesture/qwen3vl-8b/gesture_ids/{gesture_stats.json, gesture_ids.pt}; tables in logs/b1_report_qwen3vl8b.txt.
Behaviour (n=4315/cue; 3036 with tip-nearest != target):
              acc   order-bal  pick-first  acc tip-nearest!=T
  photo      .523   .521       .633        .441
  cartoon    .521   .519       .618        .430
  arrow      .647   .648       .469        .546
Activations (decode chance .301; held-out var = within-group variance explained, 2-fold over scenes):
- tgt (target name token): photo = cartoon = arrow to 3 decimals; empty (no objects) at chance; cross-origin >=
  within (L20: .84/.86); spatial-ID nearest .875. -> this is the target object's own spatial ID, not gesture info.
  Expected confound; the tgt position cannot answer Q1 as designed.
- cue ("hand"/"arrow" token), drawn cues: cartoon var cell .15-.20 / angle .30-.39 / cell x origin .55-.67;
  arrow .19-.26 / .44-.58 / .62-.75; decode within .60-.74 vs cross-origin .36-.51.
  -> encodes the drawn direction (orientation from that origin), not the pointed cell.
- cue, real hand (photo, empty): held-out var <= 0 at every layer; decode within .44-.50, cross-origin .34-.41.
  -> a weak signal (cosine decoding above chance) but no linear direction code at the cue token.
- ans: modest cell information at L20-25 for all three cues (arrow var ~.07-.08, decode .77/.71).
- Cartoon encodes direction at the cue token as well as arrow does (cross-origin aside) yet behaves like the photo
  hand (.52), so encoding the direction is not sufficient: the step from direction to object fails.
Caveats: null95 computed only at L1,5,...,33 and is high (.66-.87): grid R2 is not informative here.
spatial_nearest_acc at cue (photo .50 @L27, arrow .56) has no null yet; direction codes correlate with position.

## B2 + mirror swap + B1 nulls (qwen3vl-8b, d3ec43c), 2026-11-26
SCC logs: swap_summary_qwen3vl8b.txt, b2_analyse_qwen3vl8b.txt, b1_report_qwen3vl8b_run2.txt.
Mirror swap (x points at A, y at B, same text; 150 pairs/cue that follow the cue; median ld_shift):
  photo/cartoon/arrow alike: image_all 1.0 to L12, .9 L16, .3 L20, 0 L24; text_all 0 to L12, .1 L16, .7 L20,
  1.0 L24+; img_cue (hand/arrow pixels) .29-.32 L8 -> 0 by L20; cue word 0 and object image tokens 0 at
  EVERY layer; noise and other-hand controls ~0. Followers: photo 150/231, cartoon 150/241, arrow 150/190
  candidates (anti-followers not logged).
  -> pointing moves image -> text at L16-24; the cue-word direction code (B1) is not on the causal path.
B2 tags (held-out by scene parity):
  object image tokens: acc .50-.56 vs chance .30, but empty (no objects) .44 and pointed > fingertip-nearest
  only .51 -> a proximity/hand-nearby effect from the vision encoder, not a pointed tag; swap 0.
  NAME tokens (same object as target vs foil): photo .69 @L26, cartoon .69 @L24, arrow .85 @L22, chance .5,
  empty .49-.52. Emerges L16-26 = the swap's image->text window. behav_r .21 (hands) / .47 (arrow).
  -> candidate gesture binding on the object NAME. For hands it is more accurate than the answers (.69 vs .52).
  Real-hand pixels carry little direction (img_cue var_angle .05-.18 vs cartoon .37-.58, arrow .56-.63).
B1 nulls at every layer: spatial link at cue word well above null (cos .12-.33 vs null95 .01-.09), also for
  empty (no objects) -> the hand word holds a location in spatial-ID code (hand/fingertip place?).
Bug: cross-origin decoding splits by origin side but not by scene, so name-token identity leaks the cell at
  tgt/ans (empty tgt cross .51-.56 vs within .33). Fix: split scenes too.

## Round 2: text swap, name-tag steering, fingertip cell (qwen3vl-8b, dcab47f), 2026-11-26
SCC logs: text_swap_summary_qwen3vl8b.txt, b2_analyse_qwen3vl8b_run2.txt, b1_report_qwen3vl8b_run3.txt.
Candidates (all pairs A/B in one scene+origin): LD_x - LD_y > .5 / < -.5 / median:
  photo .615 / .164 / 1.25 nats; cartoon .680 / .151 / 1.75; arrow .773 / .119 / 4.94.
  -> the real hand DOES move the answer the right way in most pairs, but weakly (1.25 vs 4.94 nats for arrow);
  item biases (first-named, fingertip proximity) then decide the sign, so accuracy stays ~.52.
Text swap, median ld_shift (L20 / L22 / L24 / L28):
  names     photo .13/.11/.06/0   cartoon .09/.08/0/0     arrow .11/.13/.08/.03
  prefix    photo .27/.61/.89/.97 cartoon .28/.67/.89/.97 arrow .26/.64/.88/.95
  last      ~.04/.13/.43/.77 for all cues; text_all .64-.72 / .96-.99 / ~1 / 1
  noise_names and hand_names 0.
  -> the pointing reaches the answer mainly through the answer-prefix tokens ("The hand is pointing at the",
  L20-24), then the last token (L24-28). The names carry ~10%.
Name-tag steering (+s*w at B's name, -s*w at A's, scenes 100-300, w fitted on 0-99), ld_shift L20 / L22:
  s1  photo .13/.13  cartoon .10/.09  arrow .15/.22
  s3  photo .41/.46  cartoon .29/.33  arrow .45/.63;  equal-norm noise 0 everywhere
  -> causal and dose-dependent; s1 ~ the names swap (consistent). A validated but minor route.
Name tag by fingertip (n tgt-nearest 206 / foil-nearest 22 / neither 308), at best layer:
  photo .71/.59/.68, cartoon .69/.64/.69, arrow .93/.77/.80 -> follows pointing, small proximity boost
  (foil-nearest n=22 is small).
Cross-origin with scene split: empty tgt leak gone (.30-.33). Cue token: var_tip_cell > var_target for
  drawn cues (e.g. arrow L25 .27 vs .18), both ~0 for photo -> the cue word holds hand geometry (and is not
  causal: swap 0).

## Replication: qwen25vl-7b B2 tags (Kang's reference model), 2026-11-26
Name tag (chance .5; empty .47-.52): photo .647 @L18, cartoon .679 @L18, arrow .838 @L20 (8B: .69/.69/.85).
  Fingertip split (tgt-nearest / foil-nearest n=22 / neither): photo .66/.73/.64, cartoon .72/.77/.64,
  arrow .89/.86/.80 -> follows pointing, not proximity, as in 8B.
Object image tokens: .54-.59 vs .30 but pointed > fingertip-nearest only .52-.56 and empty .37 -> proximity.
tgt = target's own location (cross-origin .76-.77 @L20); cue word = arm angle (arrow vA .42-.49); real-hand
  pixels carry little direction (img_cue vA .09-.13 vs cartoon .40-.47, arrow .46-.58).
-> The B2 picture replicates across models; the tag peaks ~2 layers earlier (L18-20 of 28 vs L22-26 of 36).

## Replication: qwen25vl-7b swaps + steering; scale sweep 2B, 2026-11-27
Mirror swap (photo/cartoon): same as 8B, ~4 layers earlier: image_all 1 to L12, .57-.67 L16, .11 L18;
  text_all .38-.43 L16, .88-.90 L18, 1 L20+; img_cue .43 L8 -> 0 by L18; cue word and object tokens 0;
  controls ~0.
Text swap: prefix .10-.20 L16, .32-.46 L18, .75-.78 L20, .89-.93 L22; last .33 L20, .88-.91 L22;
  names .14-.17 at L18 only.
Name-tag steering, only at the tag peak L18: s1 .09-.20, s3 .22 photo / .22 cartoon / .57 arrow; noise 0.
Followers (LD_x - LD_y > .5 / < -.5 / median): photo .41/.08/.375, cartoon .40/.05/.375, arrow .62/.04/1.125
  -> 7B follows the hand far more weakly than 8B (8B photo median 1.25).
Scale sweep (behaviour, 40 scenes, n=578/cue; acc / acc when tip-nearest != target / pick-first):
  qwen3vl-2b photo .469/.373/.542, cartoon .502/.410/.557, arrow .512/.402/.543 -> chance for every cue
  (8B: .52/.44, .52/.43, .65/.55). 4B, 32B pending.

## Scale sweep complete + 7B arrow swap, 2026-11-27
Behaviour, pilot scenes 0-40 (n=578/cue; acc / acc when tip-nearest != target / pick-first):
            photo            cartoon          arrow
  2B   .469/.373/.542   .502/.410/.557   .512/.402/.543
  4B   .533/.447/.678   .522/.437/.619   .611/.509/.528
  8B   .543/.473/.64    .554/.470/.62    .689/.601/.48   (pilot, older code)
  32B  .578/.489/.657   .522/.412/.640   .697/.593/.542
-> arrow improves with scale (.51 -> .70); both hands stay at .47-.58 with a first-named bias of .62-.68,
   and below chance when the fingertip is nearest another object, even at 32B. Scale does not fix hands.
7B arrow mirror swap: same as the others (image_all 1 -> .17 at L18; text_all .81 L18, ~1 L20+; cue word and
  object tokens ~0; noise ~0). Followers .685 / wrong way .018 / median 1.25.
Submitted on 638bcc7: tag validation 8985354-55, oracle 8B 8985356-58, tokens 8B 8985359-61,
  oracle 7B (L16/18/20) 8985362-64, tokens 7B 8985365-67.
