# R12 Contractive Packet Recurrence CPU Preregistration

**Status:** **FROZEN 2026-07-15 before any Shohin fit, GPU execution,
production-data generation, or model score.** This contract authorizes only the
deterministic standard-library CPU theorem falsifier in
`pipeline/contractive_packet_recurrence_falsifier.py`.

**Decision:** **NO-GO as a new reasoning primitive.** A source-independent
projection can eliminate bounded off-manifold packet noise, but it cannot
strictly contract a wrong valid semantic packet. On the frozen finite board,
the favorable mechanism is exactly a five-lane repetition code around an
ordinary 889-state residual finite-state machine. Correcting a wrong valid
packet requires either a duplicate transition or source replay, and both
channels are separately charged.

**Claim boundary:** no novelty claim, no Shohin capability claim, no learned
reasoning claim, no context-compression claim, no GPU path, no fit, no network
call, and no production data. The finite result is an exact specification and
no-go boundary, not evidence about neural trainability.

## 1. Mechanism and channel separation

Let:

- `S` be a finite semantic-state space;
- `A` be an update alphabet;
- `F_a : S -> S` be the correct semantic transition for `a in A`;
- `E : S -> X` be an injective packet encoder into raw packet space `X`;
- `C = E(S)` be the valid code manifold;
- `d_X` be packet distance;
- `Pi : X -> C` be an idempotent retraction, so `Pi(c)=c` for every `c in C`;
- `Tau_a : X -> X` be the realized packet update before projection.

The recurrence is

```text
p_0 = Pi(Compiler(source, initial_state))
p_(t+1) = Pi(Tau_(a_t)(p_t)).
```

Every empirical result must account for four distinct channels:

1. **Compiler channel:** source symbols read, compiler calls and state updates,
   output bits, and compiler semantic errors.
2. **Transition channel:** semantic updates, physical-lane updates, and any
   duplicate transition used as a verifier.
3. **Projection channel:** calls, packet lanes read and written, correction
   radius, projection failures, and wrong-valid fixed points.
4. **Source channel:** externally retained source, source reads after sealing,
   residual source embedded inside the packet, and source replay.

Moving the source into a packet is not source-information deletion. It is only
deletion of an external source channel. The resource ledger records both.

## 2. Exact contraction theorem

Let `C` have minimum distance `Delta >= 2r+1`. Assume `Pi` is an exact
radius-`r` decoder:

```text
d_X(z, E(s)) <= r  implies  Pi(z) = E(s).
```

Assume a correct initial semantic state `s_0` and the following channel gates:

```text
d_X(Compiler(source,s_0), E(s_0)) <= r                 (compiler gate)
d_X(Tau_a(E(s)), E(F_a(s))) <= r for every s,a         (update gate)
Pi is called once after the compiler and every update  (projection gate)
no uncounted source, verifier, donor, or cache channel  (closure gate).
```

### Theorem 1: bounded-basin reset

Under those four gates, the projected recurrence is exact at every finite
depth:

```text
p_t = E(s_t),  where s_(t+1)=F_(a_t)(s_t).
```

The post-projection packet error is zero after every step. Thus bounded
off-manifold errors reset instead of multiplying.

**Proof.** The compiler gate and exact decoder give `p_0=E(s_0)`. If
`p_t=E(s_t)`, the update gate places `Tau_(a_t)(p_t)` inside the radius-`r`
ball around `E(F_(a_t)(s_t))`. Exact decoding gives
`p_(t+1)=E(F_(a_t)(s_t))`. Induction proves the claim. QED.

For a lane-wise transition with packet-error expansion factor `L`, inherited
error `e_t`, and fresh corruption `u_t`, a sufficient step condition is

```text
L*e_t + u_t <= r.
```

Calling an exact projector after every successful step makes `e_t=0` before
the next update. The frozen repetition code has `L=1`, `r=2`, and therefore
resets every zero-, one-, or two-lane corruption exactly.

### Theorem 2: general metric contraction

If a projected semantic error measure obeys

```text
e_(t+1) <= kappa*e_t + beta_t,  with 0 <= kappa < 1,
```

then

```text
e_L <= kappa^L e_0 + sum_(i=0)^(L-1) kappa^(L-1-i) beta_i.
```

**Proof.** Substitute the one-step inequality recursively and collect the
geometric coefficients. QED.

This theorem identifies the exact mathematical condition under which error
magnitude contracts. It does not establish that a code-manifold projection
satisfies the inequality globally. The next theorem shows why it cannot.

## 3. Global-contraction and exact-depth no-go

### Theorem 3: idempotent valid-codeword obstruction

If `C` contains two distinct valid codewords and `Pi` is a retraction onto
`C`, then `Pi` is not a strict contraction toward every target codeword on all
of `X`.

**Proof.** Choose distinct `c,c' in C` and take `c` as the target. Retraction
gives `Pi(c')=c'`, hence

```text
d_X(Pi(c'),c) / d_X(c',c) = 1.
```

No global constant `kappa<1` can satisfy strict contraction. QED.

The obstruction is semantic, not syntactic. A wrong but well-formed packet is
a valid fixed point. Distance, redundancy, checksums, and majority voting
cannot identify which valid codeword was intended without additional
information.

### Corollary 3.1: semantic errors still multiply

Let a transition independently emit the correct codeword with probability
`q_t` and a wrong valid codeword otherwise. Let all off-manifold errors remain
inside the correct decoding basin. With no semantic verifier,

```text
P(exact through depth L | compiler correct) = product_(t=0)^(L-1) q_t.
```

For constant `q<1`, this is `q^L`. Projection changes neither exponent nor
event because it fixes every wrong valid codeword. Exact recurrence avoids
depth collapse only if basin escape has zero probability under a hard bound,
or if an additional channel supplies enough information to identify and repair
semantic escapes.

The CPU witness records exact fractions `(99/100)^L` for
`L in {1,2,4,8,16,32,64}`. No floating-point approximation is used.

## 4. Classical-collapse theorem

### Theorem 4: finite source-deleted recurrence is an FSM

For finite `C` and finite update alphabet `A`, define

```text
G_a(c) = Pi(Tau_a(c)),  c in C.
```

`(C,A,G)` is an ordinary deterministic finite-state transducer with exactly
the same projected trajectories as contractive packet recurrence. Relabeling
the same implementation as an FSM preserves every resource coordinate
exactly. Tabulating `G` uses at most `|C||A|` transition entries; whether that
table is favorable must be charged, but there is no increase in computational
power.

If `E` is a bijection between `S` and `C`, the decoded transition is simply

```text
F'_a = E^-1 o G_a o E.
```

**Proof.** `Pi` maps every realized update back into `C`, so `G_a` is a total
map from the finite state set `C` to itself. Applying the definition at every
step gives the same projected trajectory by induction. QED.

The mechanism then falls into one of four exact classical cases:

1. **Source-independent nearest-code projection:** ordinary
   error-correcting coding around an FSM.
2. **Projection recomputes the expected transition from trusted prior state:**
   duplicate or verified execution.
3. **Projection reopens the source:** source replay or external execution.
4. **Compiler maps a source to a behavior quotient before late execution:** an
   ordinary finite-state compiler/executor.

On the frozen board, no noncollapsed fifth interface survives. The coded
mechanism and residual FSM agree on all 889 semantic states. The residual FSM
uses 10 active bits and depth 6, while coded recurrence uses 50 active bits and
depth 12. The FSM therefore strictly dominates this finite coded
implementation in active bits and sequential depth while preserving its
behavior.

## 5. Frozen noncommutative board

The board is the faithful action of the dihedral group `D14` on `Z_7`:

```text
T(x) = x+1 mod 7
N(x) = -x mod 7.
```

Order matters. With left-to-right execution,

```text
TN(0)=6, while NT(0)=1.
```

The board exhausts every binary source word over `{T,N}` at lengths zero
through six and every initial value in `Z_7`:

```text
source words                         127
initial values per source              7
cases                                889
local transition cells             4,494
independent local-transition checks 8,988
maximum recurrence depth                6
```

The semantic residual state is `(current_value, residual_source)`. There are
`7*(1+2+...+2^6)=889` states, requiring 10 fixed bits. This state explicitly
contains the residual source. The compiler seals it into five identical
10-bit lanes:

```text
E(s) = (s,s,s,s,s).
```

The code has lane-Hamming distance five and corrects two lanes. Projection is
the unique strict valid majority. A local update independently applies the
head operation and deletes it in every valid lane, then projection runs.

This is deliberately favorable: exact symbolic lanes, exact transitions, an
exact projector, no learned decoder, no approximate arithmetic, and no
resource-starved control.

## 6. Favorable controls

Every control must score `889/889` final cases:

- **Serial:** retain and execute the source left to right.
- **Balanced tree:** compile exact affine actions by balanced composition and
  apply once.
- **Action FSM:** consume the source with a 14-state compiler and apply its
  four-bit action code once.
- **Residual FSM:** update the exact 889-state residual machine without coding.
- **Coded recurrence:** perform five lane updates and project after every step.

The residual and coded controls must also pass every one of 4,494 local
transition cells, producing 8,988 independent local checks. A miss in any
favorable control rejects the board.

## 7. Frozen interventions

### 7.1 Off-manifold corruption

For each of 889 semantic states, the audit exhausts every lane subset at
weights zero through three. It uses both an invalid lane symbol and a coherent
wrong-valid donor lane:

```text
weight 0 subsets       889
weight 1 subsets     4,445
weight 2 subsets     8,890
weight 3 subsets     8,890
```

All 14,224 subsets at weights zero through two must project to the original
state for each corruption mode. At weight three, all 8,890 invalid-symbol
packets must reject and all 8,890 coherent donor packets must project to the
wrong donor. This confirms the exact local basin without hiding its boundary.

### 7.2 Wrong-valid semantic packets

The audit exhausts all

```text
889 * 888 = 789,432
```

ordered distinct valid state pairs. For compiler, transition, and projection
channel labels, a donor packet must remain unchanged under projection, retain
full Hamming distance five from the target, and make the recurrence follow the
donor. This is the finite witness for Theorem 3.

### 7.3 Source deletion and rescue

The sealed runtime object has exactly one field, `lanes`, and no external
source, pointer, cache, retrieval key, or verifier handle. All 889 cases execute
without post-seal external reads. However, decoding the initial semantic lane
recovers the complete residual source in all 889 cases. The source has moved
inside the packet; it has not been compressed away.

At every one of 4,494 transitions, the audit injects a wrong valid next state:

- projection preserves all 4,494 semantic errors;
- a duplicate trusted transition repairs all 4,494, charging 4,494 duplicate
  transition updates;
- source replay repairs all 4,494, charging 14,322 source symbols read after
  sealing.

No rescue is attributed to projection.

## 8. Complete resource ledger

Every algorithm has exact `compiler_channel`, `transition_channel`,
`projection_channel`, `source_channel`, `state_and_fixed_resources`, and
`external_resources` records. All model parameters, training examples,
training FLOPs, oracle calls, network calls, subprocess calls, and accelerator
calls are zero.

| Mechanism | Compiler source reads | Transition semantic/lane updates | Projection calls and lane reads/writes | Post-seal source | Active bits | Max depth |
|---|---:|---:|---:|---:|---:|---:|
| Serial | 0 | 4,494 / 4,494 | 0 | 4,494 runtime reads; max 6 retained | 9 | 6 |
| Balanced tree | 4,494 | 889 / 889 | 0 | 0 | 7 | 4 |
| Action FSM | 4,494 | 889 / 889 | 0 | 0 | 7 | 7 |
| Residual FSM | 4,494 | 4,494 / 4,494 | 0 | max 6 symbols embedded | 10 | 6 |
| Coded recurrence | 4,494 | 4,494 / 22,470 | 4,494; 22,470 / 22,470 | max 6 symbols embedded | 50 | 12 |
| Duplicate-verified coded | 4,494 | coded plus 4,494 duplicate updates | coded | max 6 embedded | 50 | 12 |
| Source-replay rescue | 4,494 | coded | coded | 14,322 replay reads; max 6 retained | 50 | 12 |

Additional exact fixed costs:

```text
balanced-tree merges             3,612
action-FSM transition entries      126
residual-FSM transition entries    889
coded projector correction radius    2 lanes
```

The action FSM is a stronger favorable compiler control: it retains a four-bit
behavior quotient instead of a ten-bit residual source state or fifty-bit
coded packet.

## 9. Frozen bytes, immutability, and admission

```text
protocol                         CPR-D14-R5-v1
canonical board bytes           174,208
source commitment SHA-256       cf12740f920062d993b89457e5de880eeae3fd536e204fa9e1c5282ad34335e4
canonical board SHA-256         ac61dc756b70c338aabb9245e1d48017048a959b02d5001e8e0aba847f7d38bd
canonical audit-report SHA-256  d119fc88af77c9dde163d644749654a70b455260613be6b287fb29cffb524187
```

Generation uses canonical ASCII JSON, `O_EXCL`, `O_NOFOLLOW` where available,
descriptor and parent-directory `fsync`, and final mode `0444`. Generation
refuses overwrite. File audit rejects symlinks, non-regular files, any write
bit, duplicate keys, non-finite values, non-ASCII input, noncanonical bytes,
digest drift, schema drift, case drift, ledger drift, or a failed exhaustive
gate.

The auditor recomputes every source, trajectory, answer, commitment, resource
entry, control, corruption subset, donor swap, rescue, theorem witness, and
classical collapse. There is no seed search, threshold search, board repair,
or score-conditioned generation.

## 10. Narrow neural hypothesis and smallest falsifier

The finite board leaves one narrow engineering hypothesis, not a new reasoning
mechanism:

> A learned redundant packet may improve exact-depth reliability only if most
> raw neural update errors are lane-local, off-manifold, and remain inside the
> correct semantic codeword's decoding basin. It cannot repair coherent
> semantic transition errors without an additional verified-information
> channel.

The smallest future Shohin falsifier would be an isolated, preregistered,
equal-call comparison among a plain residual packet, a capacity-matched
redundant sham without projection, and projected redundant packets. Before any
fit, it must freeze:

1. a source-deleted noncommutative board and exact channel ledger;
2. independent lane decoders and the semantic codebook;
3. one- and two-lane corruptions that must recover;
4. coherent all-lane valid donor swaps that must follow the donor;
5. matched calls, tokens, active bits, source access, and verifier access;
6. depth-held-out exact recurrence as the primary endpoint;
7. a gate requiring raw errors to have a correct strict lane majority often
   enough to explain any projected gain.

If projection helps only because an external parser, verifier, source replay,
or duplicate transition supplies the expected semantic state, the result is
classified under Theorem 4 and not as learned self-correction. No such Shohin
experiment is authorized by this document.

## 11. Final decision

Contractive packet recurrence has an exact and useful **local** theorem:
bounded off-manifold noise can be reset after each update. It has an equally
exact **global** obstruction: a source-independent projection cannot identify
or contract a wrong valid semantic state. Exact-depth collapse therefore
persists for semantic transition errors.

On the frozen finite board, the mechanism collapses exactly and
resource-dominatingly to ordinary repetition coding around a residual FSM;
the only semantic rescues collapse to verified execution or source replay.
The result is a rigorous NO-GO for novelty and a precise diagnostic for whether
future neural packet redundancy is worth a bounded canary.
