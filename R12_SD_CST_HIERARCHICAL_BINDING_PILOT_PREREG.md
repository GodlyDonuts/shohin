# R12 SD-CST Frozen-Parent Hierarchical Binding Pilot

**Status:** frozen training-only optimization/architecture falsifier; scored
development and sealed confirmation are forbidden

## Diagnosis

Content-addressable binding-bus pilot `693974` is rejected overall, but it
contains a sharp positive localization. On 8,000 held-out rows from the consumed
training split, declaration and initial-occurrence pointers reach 100%, and the
six-way arbitrary initial binding rises from the byte compiler's 16.725% to
8,000/8,000 exact after one epoch. Shared position-free byte-bigram equality is
therefore sufficient for declaration-to-initial binding.

The from-scratch joint fit regresses the parent compiler: final line pointer is
88.8125%, raw kind 5.7375%, amount 3.3375%, event-occurrence pointer 0%, identity
0.325%, and whole tape 0%. Four address losses dominate the joint objective,
while each event-name query redundantly has to relearn which randomized physical
line contains its semantic ordinal. Report SHA-256 is
`6a7d0ed94bc13194ceb5bac207f1726d49b75adea865b942c5ba7417e7b85b95`.

This pilot tests the smallest causal repair: preserve the exact parent byte
compiler and make event binding hierarchical over its model-owned semantic line
addresses.

## Frozen parent

Parent checkpoint:
`sd_cst_byte_pilot_4554941412679220339/compiler.pt`

SHA-256:
`e5f87a1d5b22d24250a6aac6fb7c70b4a77dbdf01bd5f5c509020a3584dfa6f9`

The parent reached 8,000/8,000 line pointer, raw kind with exactly one raw STOP,
amount, and late query on this deterministic training holdout. Every inherited
parameter is loaded from that exact checkpoint and frozen. A fail-fast prefit
gate requires the inherited line/kind/amount/query cells to remain exact before
optimization begins.

## Hierarchical binding mechanism

For each semantic event slot, the frozen parent line pointer chooses one source
byte. The architecture expands that model-owned address to its containing
newline-delimited source line using only public byte syntax. A trainable event-
name pointer is masked to that selected line. It then selects the entity span
and compares its shared byte-bigram fingerprint with the three model-selected
declaration fingerprints.

The line is not supplied by the row, compiler target, storage order, entity
role, executor, state, or answer. A wrong model line address produces the wrong
mask and cannot be repaired. The source encoder, semantic line pointers,
slot-mixing layers, kind/amount/query heads, and their projections remain
frozen. Trainable parameters are only:

- three declaration queries;
- three initial-occurrence queries;
- eight event-occurrence queries;
- the shared 65,537-by-96 bigram embedding;
- the shared 96-by-96 projection; and
- one similarity scale.

Exactly 6,306,145 parameters are trainable.

Complete system size remains 145,614,843 parameters. Global authority is below
200M, but this matched pilot retains the stricter 150M gate.

## Frozen data and optimization

- consumed v1.1 `train.jsonl` only, receipt/train hashes verified;
- deterministic `sha256(row_id)` 40,000/8,000 fit/held-out partition;
- exact parent checkpoint hash verified before load;
- four epochs, batch 64, AdamW lr `3e-4`, betas `(0.9, 0.95)`, weight decay
  `0.01`, 100-update warmup, cosine decay, gradient clip `1.0`;
- losses only for initial state, active event identity, declaration span,
  initial-occurrence span, and active event-occurrence span; and
- a fresh seed drawn only after exact source commit and push.

No kind, amount, query, line-pointer, state-transition, answer, trajectory,
reward, repair, development, or confirmation loss is applied. This isolates the
new binding parameters from the frozen solved extractor.

## Immutable gates

All held-out gates must pass:

1. prefit inherited line/kind/amount/query are each exactly 8,000/8,000;
2. declaration pointers at least 90%;
3. initial-occurrence pointers at least 90%;
4. active event-occurrence pointers at least 90%;
5. initial state at least 80%;
6. raw kind at least 90%;
7. active identity at least 80%;
8. amount at least 90%;
9. query at least 98%;
10. raw whole tape at least 60%;
11. exactly one raw STOP on every row;
12. all inherited parent parameters remain byte-identical to their loaded
    tensors after training;
13. complete system below 150M; and
14. development/confirmation access `0/0`.

A pass advances only to causal controls and fresh-board integration. It is not
a native-reasoning score. Failure rejects or revises hierarchical binding
without opening a scored split.
