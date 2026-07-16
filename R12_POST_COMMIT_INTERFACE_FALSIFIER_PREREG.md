# R12 Post-Commit Interface Falsifier Preregistration

**Status:** COMPLETED PASS on 2026-07-16. The immutable result is
`artifacts/r12/post_commit_interface_falsifier_v1.json`; the result boundary is
frozen in `R12_POST_COMMIT_INTERFACE_FALSIFIER_RESULT.md`. No Shohin adapter,
SFT, H100 job, workspace claim, or reasoning claim is authorized by this
document or by the exact scorer pass.

## 1. Question

Every representation tested so far was selected against a finite interface
known before it was frozen. Such a representation may be only a bundle of
answers for those consumers. The missing experimental axis is:

> After a fixed-size packet is committed and every source channel is deleted,
> can it support a jointly new state update, consumer, and output recoding that
> are generated only after commitment?

The first experiment tests whether the evaluator can distinguish a complete
state packet from an equal-size fitted-consumer motor packet. It does not test
whether a neural network can learn either packet.

## 2. Finite-protocol limitation

No finite protocol can exclude every finite answer table without a resource or
uniformity bound. Given a finite test tree, a machine may store source ID and
tested update prefix, return the table entry for every tested consumer, and
enter a failure sink outside the tree. Output recoding does not defeat this
construction when the recoding is supplied to the reader.

Accordingly, this falsifier makes only a bounded claim. It separates two
declared four-field-element packet classes over an exhaustive finite source
space. Passing it establishes that the harness can reject the declared motor
control, not that every motor implementation is impossible.

## 3. Exact system

Let the field be `F_17` and state be

```text
x in F_17^4.
```

An update is an invertible affine map

```text
x' = A x + b,
```

and a consumer is a nonzero row vector `c` returning

```text
y = c^T x in F_17.
```

The public fitted-consumer subspace is

```text
W = span(e_0, e_1).
```

Public updates have block form

```text
A = [[B, 0],
     [C, D]],
```

with invertible `B,D`, so every public consumer in `W` remains a function only
of `(x_0,x_1)` after any public update sequence.

## 4. Matched packets

Both packets contain exactly four `F_17` values and receive the same affine
matrix-vector operation count.

```text
state packet: z_state(x) = (x_0, x_1, x_2, x_3)
motor packet: z_motor(x) = (x_0, x_1, 0, 0)
```

The motor packet is sufficient for every public consumer under every public
update. Its last two fields are source-independent padding, not hidden access.
The packet schema contains no source ID, source text, pointer, cache, RNG state,
callable, or retrieval key.

## 5. Exact separation theorem

For a committed affine sequence with total linear part `A_*` and final
consumer `c`, define the effective source functional

```text
r = A_*^T c.
```

If `r` is outside `W`, then conditioned on `(x_0,x_1)` and uniform
`x in F_17^4`, `r^T x` is uniform on `F_17`. Therefore every deterministic
reader of `z_motor` has exact accuracy at most

```text
1 / 17.
```

The complete state packet has exact accuracy `1`. Any bijective output
permutation `pi:F_17->F_17` preserves both bounds.

For every decisive challenge, the audit must also exhibit two sources with the
same motor packet and distinct correct outputs. That collision certificate is
an exact witness independent of aggregate accuracy.

## 6. Two-phase custody

Phase one enumerates all `17^4 = 83,521` sources, constructs both packets,
serializes only packet fields, and freezes packet hashes. The source objects are
then discarded.

Phase two deterministically generates from a separately declared challenge
seed:

1. public block-preserving updates and public consumers;
2. unseen invertible affine updates whose effective consumer leaves `W`;
3. unseen consumers outside `W`;
4. update depths `1,2,4,8,9`;
5. one fresh derangement of all 17 output symbols.

The scorer receives only packet, update sequence, consumer, and output
permutation. It must reject any serialized source field or inconsistent packet
width before scoring.

## 7. Frozen gates

The exact CPU falsifier passes only if all conditions hold:

1. state and motor packets each contain exactly four field elements;
2. both arms are 100% exact on every public control cell;
3. the state arm is 100% exact on every decisive post-commit cell at every
   depth, before and after output recoding;
4. the motor arm is exactly `1/17` accurate on each exhaustive decisive cell;
5. every decisive cell has a replayable packet-collision witness;
6. a source-pointer decoy fails structural admission;
7. a depth-8 horizon decoy passes depths through 8 and is rejected at depth 9;
8. repeated generation with the same seeds is byte-identical;
9. changing only the challenge seed leaves phase-one packet hashes unchanged.

One failure closes the harness until the preregistration is revised before a
fresh result. Thresholds may not be relaxed after output exists.

## 8. Implementation boundary

The only authorized new files are:

```text
pipeline/post_commit_interface_falsifier.py
pipeline/test_post_commit_interface_falsifier.py
artifacts/r12/post_commit_interface_falsifier_v1.json
```

Existing DAQC commitment/deletion patterns and exact affine helpers may be
reused, but the result must bind its own code, parameters, seeds, packet hashes,
and report hash. The implementation may not import a solver into a later neural
reader or reinterpret this symbolic positive control as learned reasoning.

## 9. Successor boundary

A CPU pass authorizes only a tiny synthetic neural preregistration with equal
packet bits, parameters, examples, optimizer updates, and compute across the
state-seeking treatment and favorable motor controls. It does not authorize a
Shohin fit. The neural experiment must generalize across unseen scale and
post-commit generated interfaces; otherwise it remains a finite table result.

The theoretical object is residual equivalence / bisimulation, and a complete
answer bundle closed under every generator is behaviorally a state. The
potential project contribution is therefore a resource-bounded training and
evaluation protocol, not a new state ontology.
