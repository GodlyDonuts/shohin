# R12 Matched Recurrent Controls Result

**Status:** frozen negative control result, 2026-07-15. These measurements do
not establish reasoning, context scaling, or a new mechanism. They measure two
ordinary supervised recurrent-state representations from the same immutable
raw-200k checkpoint under the same 900 held-out episodes.

## Immutable evidence

| Arm | Fit job | Evaluation job | Checkpoint SHA-256 | Result SHA-256 |
|---|---:|---:|---|---|
| DRS complete-basis state | `689524` | `689525` | `5a0328d0128aa06a9a4cbaa77a40eeceab8d6f55a266efceef3f4437932c4b97` | `eb0b15413e7dcf42f27d275a5a922c3f293dbead6c5507ca7910e802d80d9484` |
| STRR static tape plus short register | `689526` | `689527` | `21a32f39de6874b9c8ccd52dff97c189445ab135b6c33b70e45379ca531c76cd` | `9a8bd97cc5f450b626aed204c47ebb6260e3f1af89e39c8eb959175f9b2adf5f` |

Both fits used 311,127 examples, one epoch, 1,115 updates, and the same
`best_step200000.pt` parent. DRS used 36,516,108 source tokens; STRR used
36,532,447. Both evaluations used 300 cases in each of `recombine_w4`,
`recombine_w6`, and unseen `width_ood_w8`. DRS evaluation wrote its complete,
hash-stable 900-case JSON and final summary before hanging during CUDA teardown;
job `689525` was then canceled solely to release the idle H100. The result was
mirrored locally and matched the remote SHA-256 before cancellation.

## Exact scores

### DRS complete-basis state

| Regime | First transition | Correct transitions / attempted | Exact final | Closed-loop state | Paired intervention |
|---|---:|---:|---:|---:|---:|
| recombine width 4 | 191/300 | 452/697 | 49/300 | 55/300 | 37/300 |
| recombine width 6 | 189/300 | 477/761 | 14/300 | 16/300 | 10/300 |
| unseen width 8 | 153/300 | 330/630 | 0/300 | 0/300 | 0/300 |
| **Total** | **533/900 (59.22%)** | **1,259/2,088 (60.30%)** | **63/900 (7.00%)** | **71/900 (7.89%)** | **47/900 (5.22%)** |

The first responses were all distinct (`900/900` unique; mode count one), so
this arm did not collapse to one repeated output string. It nevertheless
failed every exact unseen-width chain.

### STRR static tape plus short register

| Regime | First transition | Correct transitions / attempted | Exact final | Closed-loop state | Paired intervention |
|---|---:|---:|---:|---:|---:|
| recombine width 4 | 117/300 | 218/503 | 14/300 | 15/300 | 6/300 |
| recombine width 6 | 135/300 | 251/550 | 1/300 | 1/300 | 1/300 |
| unseen width 8 | 113/300 | 184/484 | 0/300 | 0/300 | 0/300 |
| **Total** | **365/900 (40.56%)** | **653/1,537 (42.49%)** | **15/900 (1.67%)** | **16/900 (1.78%)** | **7/900 (0.78%)** |

STRR emitted only 59 unique first responses and its mode occurred 50 times.
Keeping the operand tape outside the recurrent register did not reduce the
dominant transition or depth error.

## Decision

Both arms are **closed as reasoning mechanisms**.

1. Local transition imitation is real but insufficient. DRS reaches 60.30%
   transition accuracy and STRR 42.49%, while exact full-chain accuracy falls
   to 7.00% and 1.67% respectively.
2. Neither arm shows length generalization. Both score 0/300 exact finals,
   0/300 state-closed loops, and 0/300 paired interventions at unseen width 8.
3. Exposing the source again each turn is not the missing mechanism. STRR is
   worse than the complete-state DRS arm despite its immutable supplied tape.
4. Enumerating a complete local transition basis improves the learned local
   chart but does not produce an update rule that extrapolates to a longer
   machine state.
5. No further DRS/STRR SFT is authorized merely by changing the amount of the
   same local-transition data. A future candidate must directly test exact
   consume-and-transport behavior and must beat these controls at unseen depth
   without source replay, external execution, or hidden answer supervision.

This result narrows the current Shohin failure to **semantic state update plus
transport under composition**, not simply an absence of arithmetic examples,
canonical formatting, a recurrent register, or immutable source access.
