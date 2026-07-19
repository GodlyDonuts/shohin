# R12 S5 Learned Generator-Factored Executor Result

**Decision:** `confirm_s5_learned_generator_factored_execution`

S5 replaces the promoted S4 v5 host action table with a 4,934-parameter neural
unit generator. It learns only six one-step cells, receives no amount-two or
recurrent-program supervision, and exactly reproduces the host transition law
when recurrently composed behind the frozen whole-source parser. This is a
confirmed model-owned transition component, not unrestricted native reasoning.

## Custody

- Mechanism/prereg commit `fd3b3cf` preceded every fit and score.
- S5 v1 seed `107732609041319044` retired before board creation after the old
  confirmation guard failed closed; no model or result existed.
- S5.1 seed `7741142465189679834` retired before model access because 4/2,048
  rows failed the public roster-multiset gate.
- S5.2 development seed `1639560669058669827` passed all corpus gates. Data
  SHA-256 is `5d58f97f6763ac4b6550b4b2aeb959993537c185994b0ed49ac4b102c568582f`.
- Development job `693183` completed once on H100 `evc24` in 2m25s, exit
  `0:0`; assessment SHA-256 is
  `421a27fbcf6d4eb5e2084f2d01048963dbe0cafb7664bdbcf4c76cc9035c6f44`.
- Confirmation source/board commit `d6112f3` preceded the sole confirmation
  access. Seed `2190224777450473319` passed all corpus gates. Data SHA-256 is
  `7786919b6d284c359e434783638dcaed96d1c654c6e9174566c4e76767d73fc0`.
- Confirmation job `693185` completed once on `evc24` in 36s, exit `0:0`.
  Evaluation SHA-256 is
  `6aae5e9981c7f7e4a3832a75955754d37f30ed1a3ffc289e13e8334f8703017a`;
  assessment SHA-256 is
  `165b1a9ae40c8b3f52d133983c21fb22bfa2f2507b30a318f86106c96e6abc4e`.
- Prior sealed confirmation rows were never used by S5.1/S5.2 development or
  by the new confirmation builder. No post-score fit, repair, threshold change,
  second confirmation board, or rescore occurred.

## Architecture and Training

The frozen S4 v5 parser emits a variable-length categorical program of
`(direction, identity, amount)` events plus a query. The S5 executor contains:

1. an exact categorical three-identity assignment register;
2. one tied MLP receiving only current location (three-way) and direction
   (two-way);
3. hard-forward choice among six position-permutation matrices; and
4. one neural replay for amount one or two tied replays for amount two.

Training contains exactly six balanced unit cells: three locations times two
directions. Both treatment and fixed-deranged controls start from identical
weights and train for 500 updates. Both fit their assigned six labels 6/6.
There are zero source-token, identity-name, amount-two, recurrent-program,
development, confirmation, or answer-label training examples.

The treatment checkpoint SHA-256 is
`fbf7004e8094fc2c6100f108169f2283e2ad0dd3efd0408b87dff6c6583ff384`;
the matched deranged checkpoint is
`50b7284c0cc95f96a33ac871e63f1550ef5039080ac2f92ca7d22331d44a6457`.
The complete system has 133,694,869 parameters, below the 150M cap.

## Scores

| Arm | Development program/state/answer | Confirmation program/state/answer |
|---|---:|---:|
| Host exact upper bound | 97.510% / 98.193% / 98.682% | 96.924% / 97.607% / 98.096% |
| **Learned S5 generator** | **97.510% / 98.193% / 98.682%** | **96.924% / 97.607% / 98.096%** |
| Fixed deranged law | 97.510% / 19.336% / 35.107% | 96.924% / 22.217% / 36.523% |
| Direction rotated | 97.510% / 1.709% / 38.574% | 96.924% / 1.807% / 40.430% |
| State reset each event | 97.510% / 43.311% / 57.275% | 96.924% / 40.234% / 57.471% |

Parser parity with the promoted v5 decoder is 2,048/2,048 on each board.
Treatment closure is 36/36 unit transitions and **36/36 amount-two transitions
never present in training**. The deranged control is 0/36 on true unit actions
and 18/36 on amount-two closure. Every depth-three-through-eight state gate,
every amount-two-row gate, parameter/access gate, and causal-drop gate passes
on development and confirmation.

## Established Claim

For the confirmed bounded three-entity known-operation domain, a neural kernel
trained only on six primitive transition cells learns a source-deleted local
group action and composes it recurrently through depth eight. Its end-to-end
state and answer outputs are bit-for-bit score-equivalent to the old exact host
action table. The law, parsed direction, and persistent state are causally
necessary: matched law derangement, direction rotation, and state reset all
collapse exact state.

This closes the claim that host-authored action semantics are necessary for
the promoted bounded system. It also demonstrates a useful small-model design:
learn a minimal generator basis and reuse it, rather than train a continuous
state updater on every long trajectory.

## Boundary and Next Frontier

S5 is not full standalone native reasoning. The operation vocabulary remains
the twelve known left/right language atoms, the v5 decoder uses deterministic
hard-island/monotone-region assembly, the runtime invokes a fixed maximum of two
microsteps from the parsed amount, and termination follows the structurally
detected event list. No unseen operation meaning, open-ended planning,
self-generated subgoal, learned halt, free-form answer serialization, or public
benchmark gain is established.

Promote S5 as the strongest bounded reasoning baseline. The next lawful test
must attack **law induction for unseen operation semantics** or **model-owned
active-step/halt control** while freezing this parser/register/generator stack
and retaining matched derangement/reset controls.
