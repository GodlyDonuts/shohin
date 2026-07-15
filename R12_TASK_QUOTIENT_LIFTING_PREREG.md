# R12 Task-Quotient Lifting CPU Preregistration

**Status:** **FROZEN 2026-07-15 before any committed board artifact, model
fit, score, or GPU execution.** This package authorizes only deterministic CPU
generation and independent admission audit of the finite falsifier specified
below. It does not authorize a Shohin fit or a capability claim.

**Claim status:** no novelty claim, no reasoning claim, no context-compression
claim, and no learned-sufficient-state claim. The finite analytic reference is
a known linear sufficient statistic supplied as a positive control.

## 1. Task-conditioned theorem object

Let `T` be a task contract, `X` a context, `Q` a future query drawn from the
declared support of `T`, and `Y=f_T(X,Q)`. Define

```text
x ==_T x'  iff  f_T(x,q)=f_T(x',q)
                   for every q in support(T).
```

The exact task quotient is `R_T(X)=[X]`. Any fixed-length exact state `S` that
answers every declared query without reopening source information must refine
this quotient. Therefore

```text
b >= ceil(log2 |range(R_T)|).
```

For a distributional prefix code, expected state length is at least
`H(R_T|T)`. Under expected task loss `ell`, the approximate information limit
is the task-conditioned rate-distortion function

```text
R_T(D) = inf I(X;S|T)
         subject to E[ell(Y,g(S,Q,T))] <= D.
```

These are lower bounds and specification objects. They do not imply that a
125M model can discover or execute the quotient.

## 2. Reversible archive and retrieval bounds

The proposed accounting object is a deterministic reversible factorization
`Phi_T(X)=(S,A)` with `H(X|S,A,T)=0`. When both outputs are deterministic
functions of `X`, exact reversibility gives

```text
H(A|S,T) = H(X|S,T)
H(S,A|T) = H(X|T).
```

Thus task quotienting can reduce active state, but cannot losslessly compress
arbitrary context below source entropy. Every discarded distinction remains in
the archive.

For a prefix-free retrieval transcript `Z`, **all** payload, address, call
count, order, and timing channels are charged. If an answer over alphabet `Y`
has error `epsilon`, Fano's inequality and data processing require

```text
E[bits(Z)] >= H(Y|S,Q,T)
              - h2(epsilon) - epsilon log2(|Y|-1).
```

For retrieval steps `Z_1,...,Z_k`, define ambiguity debt

```text
D_i = H(Y|S,Z_1,...,Z_i,Q,T).
```

Then

```text
D_(i-1)-D_i = I(Y;Z_i|S,Z_<i,Q,T)
             <= H(Z_i|S,Z_<i,Q,T).
```

The equality is distributional: realized posterior entropy may rise on a
surprising packet, but expected debt cannot fall by more information than the
retrieval transcript carries. Model parameters are a shared program and are
never counted as episode-specific context bits.

## 3. Exact novelty and equivalence boundary

The quotient is functional compression / graph coloring. Minimal relevant
state is information bottleneck and sufficient-statistic learning. Sequential
future state is predictive-state or approximate-information-state territory.
Layered refinement is successive refinement. Choosing reads by expected
information gain is active feature acquisition. Reversible context memory and
hierarchical compression also have direct precedents.

Consequently, neither the quotient, the archive, uncertainty, nor their
combination is claimed as a new primitive. A future learned implementation
would have to be compared against information bottleneck, functional
compression, predictive-state representations, active acquisition, ordinary
retrieval, raw replay, recurrent summaries, and reversible-memory controls.

The present `GF(17)` reference is exactly a finite-dimensional linear
sufficient statistic. It is not evidence of neural discovery, reasoning,
semantic understanding, length generalization, or a resource separation.

## 4. Frozen Q-LIFT finite-field task

The world is `z in GF(17)^6`. Each case samples an invertible public basis
`C in GF(17)^(6x6)`. The task projection `P` is the first two rows of `C`, and
the exact active state is

```text
s = P z in GF(17)^2.
```

There are exactly `17^2=289` task quotient states, so the exact fixed-state
lower bound is

```text
ceil(log2 289) = 9 bits.
```

In transformed coordinates `y=Cz`, every context event has form

```text
y' = [B 0; K D] y + [c; d].
```

Therefore `s'=B s+c` is closed and independent of kernel coordinates. The
generator converts the event back to world coordinates and records both forms.
The auditor independently checks `P A = B P` and `P b = c` for every event.

There are exactly 32 base cases: eight each at context lengths
`4, 8, 16, 32`. Every case has:

- one in-family query after one additional quotient-preserving affine event;
- one out-of-family linear query whose coefficient vector is outside
  `rowspan(P)`;
- the exact final world, exact quotient state, and fixed 9-bit state code;
- an exact reversible archive of the initial vector and every event.

In-family evaluation is root-only and charges zero retrieval bits. The finite
reference answers the out-of-family query only after reconstructing the world
from the archive. This package makes no selective-retrieval efficiency claim.

## 5. Archive and transcript accounting

Each source record is canonical ASCII JSON, base64-encoded in a packet with a
consecutive integer address, byte length, and SHA-256 digest. Reconstruction
must recover the complete structured context exactly and re-encoding must be
byte-identical.

For `N` packets:

```text
payload_bits = 8 * sum(packet_payload_bytes)
address_bits_per_read = max(1, ceil(log2 N))
full_retrieval_bits = payload_bits + N * address_bits_per_read.
```

The retrieval-only reference reads every packet in canonical order. It receives
no discount for deterministic addresses, ordering, or packet count. There is
no hidden source mount, cache, pointer, verifier, solver, or uncounted replay.

## 6. Frozen controls

The CPU package contains the following controls, each with eight paired
witnesses unless stated otherwise:

- **Analytic quotient:** exact `Pz`, 9 active bits, no in-family reads.
- **Sham projection:** rows 3-4 of the same basis, same dimension and readout.
- **Capacity-matched prefix copy:** the first two raw field coordinates have
  exactly 289 possibilities and therefore the same 9-bit fixed capacity as the
  quotient. Paired contexts share this prefix but have different task answers.
- **Retrieval-only:** no sufficient active state; reconstruct the whole source
  and pay every archive bit.
- **Merge:** different kernel histories with the same quotient must have the
  same declared behavior.
- **Split:** a one-coordinate quotient perturbation must have a separating
  declared query.
- **Archive swap:** root-only answers follow retained state while an explicitly
  reopened out-of-family answer follows the substituted archive.
- **State swap:** with archive fixed, root-only answers follow substituted
  state.
- **INDEX:** exhaustive `n=8` witnesses pair every 7-bit prefix with two source
  strings separated only at the eighth bit. The exact all-coordinate state
  lower bound is eight bits.

The copy and sham controls falsify only those exact baselines. They do not prove
that every possible copying or retrieval scheme fails.

## 7. Frozen seeds and digests

```text
case   = 2026071521
merge  = 2026071522
split  = 2026071523
copy   = 2026071524
swap   = 2026071525
INDEX  = 2026071526
```

The canonical content object consists only of `cases` and `controls`:

```text
content SHA-256 = b08ab33faabe15aa09fad0b6abfa1cc94e423c3bd6447de55f547d1312d02165
board SHA-256   = 06ea09988dd2b1f84d5cc2ee5baa6e0a8bc1ea0102c3ba325d371af1929dc376
```

The generator and auditor must reject any other digest. Output creation uses
`O_EXCL`, refuses symlink replacement where supported, fsyncs the descriptor,
and removes all write bits. Existing outputs are never overwritten.

The auditor does not import the generator. It checks the frozen digest and
independently recomputes field dimensions, ranks, event closure, world and
state trajectories, in/out answers, archive reconstruction, every bit count,
all paired controls, all INDEX collisions, and all reported metrics.

## 8. Absolute CPU admission gates

The exact frozen reference metrics are:

```text
analytic quotient:       32/32, zero retrieval bits
capacity-matched copy:    0/32 on the zero-fill baseline
sham projection:          2/32
retrieval-only:           32/32, 863144 charged transcript bits
merge witnesses:          8/8
split witnesses:          8/8
copy collision witnesses: 8/8
archive/state swaps:       8/8
INDEX collisions:        128/128
```

Admission requires all of the following:

1. exact board and content digests;
2. canonical, immutable, regular-file inputs and exclusive immutable outputs;
3. all 512 frozen event/future-event closure checks;
4. exact quotient/world agreement for every case;
5. exact context reconstruction and complete address/payload accounting;
6. every merge, split, copy, swap, and INDEX witness;
7. exact independent recomputation of the metrics above;
8. no accelerator framework, subprocess execution, model checkpoint, fitting,
   optimization, or GPU path.

One mismatch rejects the board. Passing admits only this CPU falsifier.

## 9. Explicit no-go claim

TQ-Lift cannot provide bounded lossless compression of arbitrary contexts.
Exact arbitrary late INDEX over `n` independent bits requires `n` retained bits
when memory is sealed; with average bit error `epsilon`, at least
`n(1-h2(epsilon))` bits are required. Longer computation and fixed model
weights cannot recreate discarded episode-specific information.

With reversible memory, total state plus archive still carries the source
entropy. The only admissible scaling claim would be a reduction in **active**
state or expected charged retrieval for a declared task distribution whose
quotient has low entropy. This finite package neither establishes that
condition for natural language nor shows that Shohin can learn it.

## 10. Decision

This version is a frozen mathematical accounting and adversarial-control
package. It may be generated and audited on CPU. It does not authorize model
training, GPU use, board tuning, seed search, threshold search, artifact
shopping, or any statement that TQ-Lift is novel or that Shohin has acquired a
reasoning or context-scaling capability.
