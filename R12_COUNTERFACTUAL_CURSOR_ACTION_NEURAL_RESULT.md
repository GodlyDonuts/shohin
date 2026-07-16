# R12 Counterfactual Cursor-Action Neural Result

**Decision:** `neural_selector_no_go`

**Claim boundary:** This is a negative result for the frozen final-block,
head-zero Q-sidecar realization and its orbit-interchange training protocol. It
does not prove that a learned cursor controller is impossible. It does reject
promotion of this adapter, this placement, and this confirmation result.

## 1. Frozen evidence chain

- Implementation commit bound by the canary:
  `4dfcec195477c23d9e88276d58030d373bc2db6c`
- Canary SHA-256:
  `baf985855c396f63dffba1e09733a7372bd8b29c852cb5b9f482b4d59de714a1`
- Independent canary-audit SHA-256:
  `5deb9dc396e3c8d99f32b9f0e14482d288cff9d82145582665569c911a802e5d`
- Raw 260k base SHA-256:
  `91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d`
- Six-arm training job: Newton `689932`, completed on `evc30` in 3m23s.
- Training-manifest SHA-256:
  `8c499c215ade7be26ac75ea4e50cc5b335edd80f3bd5eb2225eb0166eb5bc13e`
- Score-blind evaluation array: Newton `689936`, tasks `0--5`, all completed
  without restart on `evc30`.
- Independent score SHA-256:
  `88a5e0e86cd4228fe3dd82282efae910fb36634e097f7e4de599d7c008315cc0`
- The six adapters, six raw inference files, six receipts, and training
  manifest were mirrored byte-for-byte, committed, and pushed as `51d2cd4`
  before the independent scorer was invoked.

Every arm trained for four epochs and 1,152 updates. Each evaluation contains
4,800 confirmation cells, three cursor conditions, and 450 full-model forward
batches. The scorer independently re-hashed the base, canary, audit, training
manifest, adapters, evaluator code, raw artifacts, and receipts, then ran the
frozen 20,000-replicate paired cluster bootstrap.

## 2. Preregistered selector result

| Arm | Trainable scalars | Full-vocabulary cell accuracy | Restricted cell accuracy | Exact five-action groups | Directed cursor switch |
|---|---:|---:|---:|---:|---:|
| Orbit interchange | 192 | 0/4,800 | 960/4,800 (20.0%) | 0/960 | 0/19,200 |
| Ordinary loss | 192 | 0/4,800 | 960/4,800 (20.0%) | 0/960 | 0/19,200 |
| Relation sham | 192 | 0/4,800 | 960/4,800 (20.0%) | 0/960 | 0/19,200 |
| Source only | 192 | 0/4,800 | 960/4,800 (20.0%) | 0/960 | 0/19,200 |
| Cursor table | 512 | 0/4,800 | 960/4,800 (20.0%) | 0/960 | 0/19,200 |
| Text-cursor LoRA | 640 | 0/4,800 | 973/4,800 (20.27%) | 0/960 | 88/19,200 (0.46%) |

The treatment has zero advantage over ordinary loss and relation sham under
both full-vocabulary and restricted scoring. Both observed exact-group
differences and both simultaneous one-sided lower bounds are exactly zero. All
preregistered selector checks fail. Atomic execution and one-call DONE/EOS
remain pending because a selector failure cannot advance to those gates.

## 3. Mechanistic diagnosis

This is an actuation and binding collapse, not a serialization failure.

1. **The sidecar changes logits but not decisions.** For orbit interchange,
   canonical-versus-five-cycle cursor intervention changes the five restricted
   logits by mean L-infinity `0.1035` and maximum `0.5000`, yet changes zero of
   4,800 restricted predictions. Canonical-versus-clamped changes them by mean
   `0.0823`, also with zero prediction switches.
2. **The full-vocabulary margin is much larger.** The full-vocabulary winner is
   above the best action token by mean `2.7450`, median `2.5366`, and 95th
   percentile `4.8438` logits. The cursor effect is therefore roughly 25 times
   smaller than the typical margin it must cross.
3. **More cursor-table capacity does not repair placement.** The favorable
   512-scalar table has the same 20% restricted and 0% full-vocabulary result.
   The failure is not specific to the centered three-bit factorization.
4. **The final-block single-head Q path has weak gradient leverage.** Treatment
   mean full-vocabulary CE falls only from `6.72330` in epoch 1 to `6.70808` in
   epoch 4. Mean adapter gradient norm falls from `0.00201` to `0.00142` while
   the projection norm grows to about `62`. Because the delta is inserted
   before QK normalization, growing its norm mainly saturates query direction;
   it does not create a direct write channel into action logits.
5. **A stronger text perturbation still does not bind source and cursor.** The
   640-scalar text LoRA changes canonical-versus-cycle restricted logits by
   mean L-infinity `1.0407` and flips 870 restricted predictions, but remains at
   20.27% accuracy with zero exact groups. Logit leverage alone is insufficient
   without the correct source-by-cursor interaction.
6. **DONE is never learned by the matched sidecars.** Restricted accuracy at
   cursor four is `0/960` for the four 192-scalar arms and cursor table. This
   independently blocks autonomous termination.

The strongest supported statement is:

> Orbit-interchange supervision did not install a usable causal selector
> through one final-block query head. The frozen base exposes some
> cursor-sensitive logit movement, but this interface cannot bind operation
> order to cursor strongly enough to overcome vocabulary competition.

## 4. Next admissible gate

Do not tune this score, reuse this confirmation as development data, move the
same sidecar to another layer, or increase its size post hoc. The next bounded
experiment must separate **representation availability** from **vocabulary
actuation** on the already designated development split:

1. Fit a source-by-cursor tensor-product readout from frozen pre-final hidden
   states to the five action classes, with source-only and cursor-only collapse
   controls. This is a diagnostic probe, not a reasoning mechanism.
2. If joint development accuracy is high, test a separately accounted
   SELECT-only action-logit valve that writes the five readout scores into the
   corresponding existing vocabulary logits. This determines whether the
   remaining failure is the write interface rather than the representation.
3. If the development probe fails, stop. The frozen representation does not
   expose the required joint variable to a small readout and another decoder
   fit is unjustified.
4. If both development gates pass, freeze one implementation and generate a
   fresh operand/renderer confirmation split before any new score-bearing run.
   The exposed v1 confirmation may not be used to choose ranks, layers,
   thresholds, or losses.

The tensor-product probe and logit valve are known classifier/gating machinery;
they carry no primitive-novelty claim. Their purpose is to locate the missing
causal interface precisely enough to decide whether the independent R12
training-protocol track is still viable.
