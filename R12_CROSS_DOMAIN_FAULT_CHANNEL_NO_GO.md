# R12 Cross-Domain Fault-Channel No-Go

**Status:** three candidate mechanisms rejected before neural implementation.
No data generation, fit, accelerator work, or Shohin capability claim is
authorized.

**CPU protocol:** `R12-CROSS-DOMAIN-FAULT-CHANNEL-NO-GO-v1`

## 1. Motivation

Shohin's measured failure is not a generic absence of useful internal signal.
The protected raw-300k model shows:

- natural-language compilation: `0/6`;
- oracle-compiled frozen DRS transitions: `28/34`;
- terminal serialization: `2/6`.

Other frozen boards show the same asymmetry: local transitions can be strong,
while autonomous operation selection, state transport, halting, correction, and
state consumption fail. Cross-domain inspiration is useful only if it attacks
that measured error channel and survives a resource-matched collapse test.

The search considered four source domains:

1. biological error-correcting population codes;
2. paired forward/inverse motor models;
3. conservative and reversible dynamics; and
4. hippocampal replay and compositional state construction.

These are real scientific precedents, not novelty claims. Relevant primary
sources include [fault-tolerant neural networks from biological error-correction
codes](https://arxiv.org/abs/2202.12887), [tandem forward and inverse internal
models in cerebellar motor learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC6048491/),
and [compositional memory construction through hippocampal
replay](https://www.nature.com/articles/s41593-025-01908-3).

## 2. Fault-neighborhood lemma

Let `r` be one causal state, `E(r)` its retained encoding, and `F` the admitted
fault family. Define its observed fault neighborhood

```text
N(r) = { f(E(r)) : f in F }.
```

### Lemma

Exact autonomous recovery is possible only if

```text
N(r) intersect N(s) = empty
```

for every pair of distinct causal states `r != s` that require different
future behavior.

### Proof

If an observed configuration `y` lies in both neighborhoods, then one admissible
history requires recovery to `r` and another requires recovery to `s`. A
deterministic decoder receiving only `y` cannot return both. A randomized
decoder cannot be exact on both. Therefore exact recovery requires disjoint
neighborhoods or additional information that distinguishes the origin.

Disjoint neighborhoods are coding redundancy. Extra origin information is
retained source, provenance, a checkpoint, or another state channel. A selector
that supplies missing semantic information is an oracle. Reversible transport
can permute overlapping neighborhoods but cannot make them disjoint. A
contracting projection is a decoder; if it merges future-distinguishable clean
states, it violates causal separation.

The lemma does not reject error-correcting representations. It prevents them
from being credited with semantic information they do not contain.

## 3. Candidate A: Triadic Efference Commit

### Proposed object

A controller, executor, and forward model independently propose a next state.
The machine commits only when two of three agree. Output text is downstream of
the committed state and cannot modify it.

```text
z_t = (x_t, u_t^C, u_t^E, u_t^F, syndrome_t)
u_t^i = F_i(x_t, event_t)
x_(t+1) = majority(u_t^C, u_t^E, u_t^F)
```

### Finite result

For the complete one-bit transition `x' = x xor a`, the CPU board checks all
four state/action pairs and every one-of-three lane fault: 12/12 are recovered.
The decoder is byte-for-byte ordinary three-copy repetition-majority decoding.

Two disagreeing lanes cannot identify the faulty member. Observation `(0,1)`
is consistent both with truth zero and a fault in lane one, and with truth one
and a fault in lane zero.

When all lanes share the wrong semantic action, majority accepts the wrong
codeword in 4/4 cases. This matches Shohin's common-mode operation-selection
failure rather than repairing it.

### Verdict

**REJECT as a new mechanism.** A favorable three-copy recurrent control receives
the same parameters, retained bits, examples, and compute and has identical
behavior. Redundancy may remain an engineering control after semantic
compilation works, but it is not the missing compiler.

## 4. Candidate B: Shadowed Variational Transport

### Proposed object

Each event induces a reversible, invariant-preserving state action. A shadow
trajectory and conserved quantities are intended to prevent compositional drift.

```text
z_t = (q_t, p_t, constraints_t, invariants_t)
z_(t+1) = Phi_event(z_t)
```

where every `Phi_event` is invertible. A separate observer reads the endpoint.

### Finite result

The CPU board uses the determinant-one cat map

```text
A = [[2, 1],
     [1, 1]] mod 5.
```

It enumerates all 25 states, all 24 nonzero perturbations, and ten recurrent
steps: 6,000 perturbed/clean comparisons. No nonzero error ever contracts to
zero because every power of `A` is invertible.

Shadowing guarantees at most that a perturbed trajectory is a valid nearby
trajectory. It does not identify the trajectory belonging to the committed
history. A projection that chooses that history is a noninvertible decoder or
uses extra provenance. Reversible realization of an irreversible task must
retain discarded information in an ancilla or archive.

### Verdict

**REJECT as an error-correction mechanism.** A matched recurrent controller can
apply the same reversible map with identical state, precision, depth, and
compute. Conservation can preserve information but cannot supply missing
semantic selection or remove ambiguity.

## 5. Candidate C: Consolidated Relation-Syndrome Atlas

### Proposed object

Short event blocks enter a fast trace. Replay applies known algebraic relations,
checks a syndrome, commits a canonical block action into slow causal state, and
retires the raw trace. Late queries read only the slow state.

### Finite result

The CPU board uses the symmetric group `S3` with adjacent transpositions `s`
and `t`. It verifies the involution and braid relations

```text
s^2 = identity
t^2 = identity
sts = tst.
```

The complete relation atlas has six states and twelve state-generator pairs. It
is exactly the ordinary tied six-state recurrence in canonical coordinates.

The board then removes one of the twelve pairs and constructs a patched updater
that is exact on all eleven admitted pairs. It passes every observed transition
but fails a word that reaches the omitted pair; at least one late query separates
the patched endpoint from the exact endpoint.

Retaining the raw trace is retrieval. Updating the atlas within an episode is
fast weights. Host canonicalization is external symbolic execution. A complete
fixed atlas is the recurrent transducer itself.

### Verdict

**REJECT as a distinct reasoning primitive or finite identification protocol.**
Replay may allocate training examples usefully, but completeness or a uniform
generalization theorem is still required. Relation consistency on an
incomplete finite board does not identify the missing transition.

## 6. Resource and claim boundary

The mandatory resource vector remains

```text
(parameters, retained bits, precision, source bytes,
 training examples, oracle calls, training FLOPs, inference FLOPs,
 sequential depth, external memory, external execution).
```

Each candidate has a favorable conventional realization preserving that vector:

| Candidate | Favorable matched control | Surviving advantage |
|---|---|---|
| Triadic commit | Three-copy repetition-coded recurrence | None |
| Variational transport | Recurrence applying the same invertible map | None |
| Relation-syndrome atlas | Tied relation-aware finite transducer | None |

The three mechanisms therefore receive `0/3` survival at the exact-collapse
gate. This result does not prove that every possible biological, physical, or
mathematical inspiration fails. It proves only these three named reductions.

## 7. Consequence for Shohin

The next high-value measurement is error-channel attribution, not generic
redundancy:

1. freeze a source and correct typed program;
2. separately intervene on opcode, operand boundaries, local transition, carry,
   halt, and serializer state;
3. measure whether errors are independent across components or common-mode;
4. permit coding redundancy only for empirically independent corruption; and
5. direct new parameters and examples toward semantic compilation when lanes
   agree on the same wrong program.

This supports the current compiler/executor/serializer decomposition but grants
no VAMT neural authority. The full-program VAMT CPU board is reviewed
separately. A compiler failure cannot be rescued by calling an exact executor
"reasoning," and an exact executor cannot be blamed for a wrong compiled
program.

## 8. Reproducibility and authorization

The executable evidence is:

- `pipeline/cross_domain_fault_channel_falsifier.py`
- `pipeline/test_cross_domain_fault_channel_falsifier.py`
- generated report `scratchpad/cross_domain_fault_channel_no_go_v1.json`

The report must be deterministic, refuse overwrite, label all three candidates
rejected, and keep `neural_preregistration_authorized = false`.

Current authority:

```text
CPU no-go mechanics:         allowed
Neural preregistration:      NO-GO
Neural implementation:      NO-GO
Data generation or fitting: NO-GO
H100 work:                   NO-GO
Novelty/reasoning claim:     NO-GO
```
