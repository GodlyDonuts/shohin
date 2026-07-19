# R12 Referential Gather-Delete Executor v1.1 Preregistration

**Status:** frozen before fit

**Claim class:** isolated source-deleted execution component. Passing this
board is not natural-language reasoning, autonomous rollout, halting, broad
generalization, or architectural novelty.

## Falsifiable question

RGDE v1 failed because categorical pointer softmax collapsed a multi-token
referent to an unstable subtoken. The committed no-fit probe recovers entity
identity at 4,090/4,096 = 99.854% using a set-valued lexical span. The v1.1
question is therefore narrow:

> When source text is deleted, can a tied neural permutation updater compose
> two operations learned only as independent atomic updates when identity is
> carried by complete lexical spans and control semantics remain contextual?

## Immutable upstream evidence

| Object | SHA-256 |
|---|---|
| raw Shohin 300k | `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6` |
| qualified ordinary compiler file | `747a559b827c6d114943c091b9dea5b4b90cef7af13aa5003b8435c092d24991` |
| factorized train, 96,000 rows | `e6feb311c37f34a88ce7bda59ebb4f968c9ce3b4052cb5c0f6c2ef2e3fca44a8` |
| compositional development, 2,048 rows | `e69fb70bddfb827a428c297352a72e45612ff3528a9fa107dec38c04189e1922` |
| factorized report | `d481114232e438294bd1ea7f5b739f6068c2bf10fe02c1ee3c216c2e56aa3be3` |
| tokenizer | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| no-fit identity result | `dcc16fa3101e403a5cd2452171511fe9f4497c5c879ca1fa65da0e31ba615f60` |

The frozen implementation identities are compiler
`debf439a61dfb33efe1c863c2d0df3ec2e049f7c4178ef41e6b500d3a8975d23`,
packet/executor
`d7496f1969eebc494186a1984f45617310ba4fe90c17e0472b461b13c12a0ee4`,
trainer `e0ab2110119bf50a3d918850f54d4077bfc9f030f7f894c55e2af0502148e999`,
evaluator `8b2883c1a8195b5dc27f6e6b9c25ba441e387ed90c3f557bb29902945cbf1d92`,
test `fa24d5a4d4b545bd00699b8f406ffd77c1f9a33216635b607fdd65c123c2d0eb`,
and Slurm job
`0d06071e4ee3bb9e43808ffa932bb2926bec17f6cf2de5c66fbf920d53c6b6bb`.

Old factorized confirmation and the one-shot compiler qualification board are
not fit or evaluation inputs. Confirmation access is zero.

## Dual-channel packet and deletion boundary

The frozen ordinary compiler produces role logits, operation-kind logits,
384-dimensional contextual token states, and the base model's frozen
576-dimensional vocabulary embeddings. The gather has zero parameters.

For every predicted role, it computes `sigmoid(role_logit) * valid_mask` and
normalizes the result to sum to one. It gathers:

- three initial-entity vectors from frozen lexical embeddings;
- each operation entity and literal from frozen lexical embeddings;
- each operation-kind context and the query position from contextual states;
- each operation's two-class compiler-owned kind distribution.

The treatment executor receives no token IDs, valid mask, role logits, full
lexical memory, full contextual memory, strings, source positions, decoded
entities, host state, or host answer. The bounded packet is the only executor
argument. Both source memories are discarded before its forward call.

The mutable state remains a differentiable `3 x 3` destination-to-source
assignment matrix. One 192-wide neural cell is called twice. Sinkhorn
normalization enforces only the matrix type; operation direction, amount,
entity match, destination, and query semantics are learned. A separately
parameterized query consumer reads the final assignment.

## Frozen training arms

All arms use one epoch, seed `2026071902`, batch size 64, AdamW `0.001`, 50
warmup updates, clip 1.0, and the same frozen base/compiler.

1. **Tied predicted treatment:** 192,000 independent atomic examples. Op0 and
   op1 each start from identity. No composed transition, final state, or full
   two-step answer is a training target.
2. **Untied predicted comparator:** identical atomic examples, but separate
   cells for source slots 0 and 1. It is more parameterized and favorable.
3. **Tied gold-packet ceiling:** identical atomic objective with gold source
   roles and gold operation kinds. State update and answer remain neural.
4. **Tied composed-supervision ceiling:** identical predicted packet and tied
   architecture, but the complete two-update transition/final-answer objective
   is supplied during training. This tests architecture learnability; it is
   not a promotion arm.
5. **No-fit operation shuffle:** rotate both operation packets across rows.
6. **No-fit query shuffle:** rotate the query packet across rows.
7. **No-fit gold rescore:** rescore the tied predicted treatment with gold
   roles/kinds.

No unused parameter padding or seed sweep is allowed.

## Parameter ledger

| Arm | Executor | Total system |
|---|---:|---:|
| tied predicted / tied gold / tied composed | 1,491,279 | 135,180,829 |
| untied predicted | 2,459,161 | 136,148,711 |

The ledger includes 125,081,664 frozen base parameters and 8,607,886 frozen
compiler parameters. Every arm remains below 150,000,000 parameters.

## Advancement gates

The atomic tied mechanism advances to a fresh, commit-before-seed depth board
only if every gate passes:

1. the no-fit lexical carrier remains at least 99% entity identity;
2. the composed-supervision ceiling reaches at least 99% answer accuracy, 98%
   exact final assignment, and 98% both-transition exactness;
3. tied gold atomic reaches at least 98% answers/final assignment, 95% both
   transitions, and 99% query accuracy;
4. tied predicted atomic reaches at least 95% answers/final assignment, 90%
   both transitions, and 99% query accuracy;
5. every surface reaches 90% answers and at least 450/512 quartets have all
   four answers correct;
6. treatment is within two points of its gold rescore on answers and final
   assignment;
7. operation shuffle is at most 40% answers/final assignment and loses at
   least 50 points to treatment;
8. query shuffle is at most 45% answers and loses at least 45 points;
9. tied answers/final assignment are within two points of untied while using
   fewer parameters;
10. all runs complete `0:0`, bind every upstream/source/state hash, record zero
    confirmation access, and apply at least 99% of requested interventions.

If composed supervision fails, reject the updater architecture. If composed
passes but gold atomic fails, reject atomic transfer. If gold passes but
predicted fails, reject the packet. If shuffles remain high, reject causal use.
Passing development authorizes only a new three-to-eight-operation source-
deleted confirmation board. It never opens old confirmation by itself.
