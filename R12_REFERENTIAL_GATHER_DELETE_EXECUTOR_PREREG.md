# R12 Referential Gather-Delete Permutation Executor Preregistration

**Status:** closed negative after jobs `693111--693114`; exact result is frozen
in `R12_REFERENTIAL_GATHER_DELETE_EXECUTOR_RESULT.md`

**Claim class:** isolated source-deleted execution component. A pass does not
establish natural-language reasoning, autonomous rollout, halting, broad
generalization, or architectural novelty.

## Question

The conventional complete compiler is now independently qualified at more than
99.9% exactness on a fresh known-atom board. The next unresolved interface is
not parsing:

> Can a separately parameterized model-owned state updater learn atomic list
> transitions, reuse the same weights to compose two operations after the
> source is deleted, and expose the final identity to an independent query
> consumer?

This experiment deliberately trains the treatment on one operation at a time.
The two-operation answer and state are never treatment training targets. Full
two-step execution appears only in development evaluation.

## Frozen upstream identities

| Object | SHA-256 |
|---|---|
| raw Shohin 300k | `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6` |
| qualified ordinary compiler file | `747a559b827c6d114943c091b9dea5b4b90cef7af13aa5003b8435c092d24991` |
| factorized train, 96,000 rows | `e6feb311c37f34a88ce7bda59ebb4f968c9ce3b4052cb5c0f6c2ef2e3fca44a8` |
| compositional development, 2,048 rows | `e69fb70bddfb827a428c297352a72e45612ff3528a9fa107dec38c04189e1922` |
| factorized report | `d481114232e438294bd1ea7f5b739f6068c2bf10fe02c1ee3c216c2e56aa3be3` |
| tokenizer | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |

The old factorized confirmation and the one-shot qualification board are not
training or evaluation inputs. Confirmation access remains zero.

## Architecture and hard source boundary

The raw base and complete compiler are frozen. The compiler emits ten pointer
distributions, two operation-kind distributions, and 384-dimensional contextual
token states. A zero-parameter differentiable gather produces exactly:

- three initial-entity vectors;
- for each of two operations, one operation-kind context, entity vector,
  literal vector, and two-class kind distribution;
- one query-position vector.

No token IDs, source mask, pointer logits, or full source memory are arguments
to the treatment executor. Its `forward(packet)` API is the deletion boundary.
Host code may invoke the frozen compiler and perform the declared weighted
tensor gather. It may not decode entity strings, operation polarity, amounts,
query positions, current state, or answers, and it may not apply a list move.

The mutable state is a differentiable `3 x 3` assignment matrix in the Birkhoff
polytope. Rows are current positions and columns are initial entity identities.
It starts as identity. A neural cell compares the operation entity with current
entity states, consumes literal and kind representations, and predicts a `3 x
3` destination-to-source transition. Six alternating row/column log
normalizations make it doubly stochastic. Matrix composition updates the state.
The same cell instance is called twice. A separate query head predicts one of
three positions; multiplying that distribution by the final assignment returns
an identity distribution.

The fixed matrix multiplication and Sinkhorn normalization enforce only the
state type. They do not encode left/right, amount, entity binding, destination,
or query semantics. All those choices are learned. This is a typed
neuro-symbolic inductive bias closely related to learned permutation networks
and recurrent program executors, not a new computational class.

## Training contract

The treatment and untied comparator each receive one epoch, seed `2026071901`,
batch size 64, AdamW learning rate `0.001`, 50-update warmup, clip 1.0, and
1,517 optimizer updates. Each source batch supplies two independent atomic
examples:

1. operation 0 applied from identity state;
2. operation 1 applied independently from identity state.

Thus each epoch contains 192,000 atomic transition targets. Neither cell sees a
two-step transition target, final two-step assignment, or full-program answer
during training. Atomic losses supervise transition rows, moved-entity
location, amount, query position, and the one-step answer identity. Base and
compiler trainable parameters are exactly zero.

## Frozen arms and interventions

1. **Primary tied/predicted:** frozen compiler packet; one shared update cell.
2. **Favorable untied/predicted:** two independent update cells, one per source
   operation slot; more parameters and the same atomic supervision.
3. **Gold-packet tied ceiling:** gold source spans and operation kinds in both
   fit and evaluation; still no host state update or answer.
4. **Source-retained upper bound:** two-layer cross-attention decoder receives
   the entire frozen compiler memory and is favorably trained on full
   two-operation answer identities. It is not a source-deleted positive.
5. **No-fit operation shuffle:** the trained treatment receives another
   equal-length row's two operation packets while initial/query packets stay
   fixed.
6. **No-fit query shuffle:** the trained treatment receives another
   equal-length row's query packet while initial/operation packets stay fixed.
7. **No-fit gold packet:** the trained predicted-packet treatment is rescored
   with gold spans/kinds to measure the compiler-interface ceiling.

All arms use width 192. Instantiated parameter counts are:

| Arm | Executor/control | Base + compiler + arm |
|---|---:|---:|
| tied | 1,416,783 | 135,106,333 |
| untied | 2,384,665 | 136,074,215 |
| source-retained | 1,262,787 | 134,952,337 |

Every arm remains below the strict 150,000,000-parameter ceiling. The gold arm
has the tied count. Unused parameter padding is forbidden.

## Development metrics

The evaluator scores exact destination-to-source transition rows at both
steps, final three-identity assignment, query position, answer identity,
operation-entity matching, amount, every surface separately, and all-four
quartets. The scorer may derive gold permutations from structured rows only
after inference; those labels never enter a positive forward pass.

## Frozen advancement gates

The primary mechanism advances to one fresh confirmation design only if all are
true:

1. source-retained full-answer accuracy is at least 95%, proving the board and
   frozen representation permit a favorable solution;
2. gold-packet tied training reaches at least 98% two-step answer accuracy,
   98% final-assignment exact, 95% both-transitions exact, and 99% query
   accuracy;
3. predicted-packet tied treatment reaches at least 90% answers, 90% exact
   final assignments, 85% both-transitions exact, and 99% query accuracy;
4. every canonical/paraphrase/order-twin/binding-twin surface is at least 85%
   answer-accurate and at least 400/512 quartets have all four answers correct;
5. treatment answer and final-assignment accuracy are each no more than five
   percentage points below its no-fit gold-packet rescore;
6. operation-shuffled answer and final-assignment accuracy are each at most
   45%, and treatment exceeds each by at least 40 percentage points;
7. query-shuffled answer accuracy is at most 45%, and treatment exceeds it by
   at least 40 percentage points;
8. at least 99% of rows receive each requested packet intervention;
9. tied answer and final-assignment accuracy are each within two percentage
   points of the favorable untied arm while using fewer parameters;
10. every artifact binds base, compiler, data, report, tokenizer, initialized
    state, final state, arm, source boundary, and zero confirmation access.

If the tied arm solves the board but loses the untied comparison, retain it only
as a conventional source-deleted executor baseline; do not attribute an
advantage to tying. If shuffles remain high, reject causal packet use. If gold
passes and predicted fails, return to Stage A packet quality. If both fail, the
typed updater/consumer is inadequate.

## Promotion boundary

A development pass permits a new, commit-before-seed confirmation corpus with
longer three-to-eight-operation programs and fresh language/name/factor
combinations. It does not permit opening any old confirmation. Only a later
depth-extrapolation and packet-causality pass could justify connecting the
executor to a model-owned serializer or natural-language answer path.

No outcome from this development board alone may be called general reasoning
or a state-of-the-art result.
