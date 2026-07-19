# R12 S9 Occurrence-Quotient Development Result

**Decision:** rejected for confirmation after a large development-only gain

**Development access:** 1

**Confirmation access:** 0; the sealed confirmation board remains unopened

## Custody

- neural source commit: `9fd8aea`
- bounded-memory repair commit: `ba9e4c6`
- frozen board commit: `d3cacd7`
- board seed: `7563652620455132721`
- training seed: `1782702123750965299`
- scoreless infrastructure jobs: `693705` (`evc44`, no GPU visible) and
  `693706` (empty output-directory guard after the prior preflight failure)
- sole valid score-bearing job: Slurm `693707` on `evc45`, completed cleanly
  in 18m59s
- board report SHA-256:
  `fb81b75f5963ad4bcd513d9e4a14e2fa36ad02dabd1085b9f4387c270755cd93`
- development SHA-256:
  `193df4513e9b7186aefbe3890931be85f4a7f0b154ab39c0409614603a686ff0`
- sealed confirmation SHA-256:
  `2f0967bc35ee4b01f1adb59e6f0278c18394f3a0f84a5727b21e8df21e256419`
- checkpoint SHA-256:
  `02a0ae680aa817d5f20c7fab0a75baeae4e4e262231e3051925a980f492ff8fc`
- evaluation SHA-256:
  `874f762648d2eeca7868cea6f9b3a51c6eb9186e2ad22aedfc1db5ba07ae3a94`
- assessment SHA-256:
  `85565f07f880730d35672cefa597c9d2c2498278c94c6db53e6cafd456e70a09`

The local mirror contains the checkpoint, evaluation, and assessment with the
same hashes. The large checkpoint remains outside Git.

## Frozen contract

The 125,081,664-parameter Shohin base is frozen through layer 19. The treatment
adds a 9,498,382-parameter bounded-span relational compiler and the confirmed
218-parameter S7 cyclic generator, for 134,580,264 total parameters. Every
contiguous source span up to four tokens is scored. The treatment receives a
mean message from other candidate spans with the same exact trimmed source
surface. The equal-parameter no-class arm zeros only that message. The shuffled
arm retains the architecture but trains on independently permuted relation
labels.

All arms receive 750 updates over the same 48,000 graph-only training sources.
They receive no final state, answer, recurrent trace, development law, or
confirmation law. The treatment and no-class arms both reach 100% sampled
candidate and positive-label accuracy at the end of fitting; the shuffled arm
does not. End-to-end development therefore remains the decisive comparison.

## Development scores

| Arm | Exact graph | Exact state | Answer |
|---|---:|---:|---:|
| Gold graph | 2,048/2,048 = 100.000% | 2,048/2,048 = 100.000% | 2,048/2,048 = 100.000% |
| **S9 treatment** | **1,941/2,048 = 94.775%** | **1,943/2,048 = 94.873%** | **1,943/2,048 = 94.873%** |
| Equal-budget no-class message | 950/2,048 = 46.387% | 951/2,048 = 46.436% | 951/2,048 = 46.436% |
| Shuffled relations | 0/2,048 = 0.000% | not a scored runtime arm | not a scored runtime arm |
| Closed S8.1 treatment | 514/2,048 = 25.098% | 514/2,048 = 25.098% | 514/2,048 = 25.098% |

S9 improves exact graph compilation by **69.678 percentage points** over S8.1
and **48.389 points** over the matched no-class arm. State accuracy remains
above 85.67% at every evaluated depth from three through eight:

| Depth | Exact state |
|---:|---:|
| 3 | 293/342 = 85.673% |
| 4 | 313/340 = 92.059% |
| 5 | 335/342 = 97.953% |
| 6 | 332/342 = 97.076% |
| 7 | 334/341 = 97.947% |
| 8 | 336/341 = 98.534% |

The treatment emits 1,943 valid graphs. Of these, 1,941 are exact semantic
graphs. The two non-exact graphs nevertheless execute to the expected state and
answer on their particular programs. This is not counted as exact graph
compilation.

## Attribution and causal controls

The explicit occurrence-class message is causally useful on this board. It
raises exact graph accuracy from 46.387% to 94.775% with the same parameter
count, training examples, update count, frozen base, and global memory encoder.
The shuffled-label arm emits zero exact graphs.

Every graph-storage reindexing is invariant on all 1,943 valid treatment rows.
The unchanged S8/S7 execution controls sharply reduce exact state:

| Intervention | Exact state |
|---|---:|
| Reversed links | 140/2,048 = 6.836% |
| Deranged cards | 28/2,048 = 1.367% |
| One witness | 95/2,048 = 4.639% |
| State reset | 54/2,048 = 2.637% |
| Early nil | 77/2,048 = 3.760% |

These controls rule out storage order, ignored links, ignored card witnesses,
source-state replay, and ignored nil halt as explanations for the high treatment
score.

## Why confirmation is forbidden

Twenty of twenty-two frozen gates pass. Two fail:

1. **Exact class membership:** 1,941/2,048 = 94.775%, below the frozen 95%
   floor by five examples.
2. **Operation-name recoding:** all 1,925 mutually valid original/recoded pairs
   are bit-identical, but 18 originally valid parses become invalid after the
   source operation names are rotated and fully retokenized. The frozen gate
   requires eligibility for every valid original graph.

The aggregate span F1 is 99.997% on graph-producing rows, with no false-positive
spans and six false negatives in that scored subset. It must not be described as
an unconditional all-row boundary score because rows that fail graph assembly
do not contribute span sets to that aggregate. The all-row class-exact score is
the binding robustness measure and is the one that fails.

No threshold is relaxed, no same-board repair is scored, and the sealed
confirmation file remains unopened.

## Claim boundary and next phase

S9 supplies strong development evidence for a bounded model-owned chain:
natural-language span/relation extraction, repeated-identity binding, graph
order and nil halt, recurrent cyclic state updates, and query consumption. It
does not yet establish a confirmed mechanism, non-identical coreference,
unbounded reasoning, self-generated problem decomposition, or broad benchmark
reasoning.

The admissible successor is a fresh-board S9.1 robustness experiment, not a
wider repeat. It should keep the parameter count and proven S7/S8 runtime fixed
while testing two preregistered changes:

1. operation-name orbit augmentation or a consistency objective so source
   recoding is an explicit learned equivariance rather than an incidental OOD
   test;
2. a model-logit-only constrained relation assignment that guarantees the
   declared graph grammar without receiving gold spans, names, event order, or
   halt.

Both changes require new source commits, seeds, development, and sealed
confirmation bytes. The same no-class and shuffled controls and all current
thresholds remain required.
