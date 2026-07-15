# R12 Deferred-Argument Quotient Compilation (DAQC) Preregistration

Status: **frozen CPU theory/falsifier package**

Protocol: `DAQC-D34-v1`

Scope: `pipeline/daqc_cpu_falsifier.py` and its exhaustive CPU tests

Neural fitting: **prohibited in this package**

GPU use: **prohibited in this package**

## 1. Claim Discipline

This preregistration tests whether a precise deferred-argument compilation
protocol is internally coherent on one finite noncommutative action system. It
does not test Shohin, fit any model, or establish a new computational class.

The allowed conclusion after every gate passes is only:

> For the frozen `D34` board, a source word can be replaced before a late input
> arrives by its exact six-bit action-equivalence class, and a fixed exact
> decoder can reproduce the source program on every input in `Z17`.

Passing does **not** establish neural learnability, tiny-model reasoning,
general context scaling, latent thought, transfer outside this action system,
state-of-the-art performance, or novelty over finite-state compilation and
partial evaluation. A failed mandatory gate falsifies the v1 implementation.
A passed gate only permits a separately preregistered, isolated neural test.

## 2. Formal Problem

Let `X = Z17`. Source programs are words `w` over generators `{T, N}`, applied
from left to right, where

```text
T(x) = x + 1 mod 17
N(x) = -x mod 17.
```

Write `U_w : X -> X` for execution of `w`. The late argument `x` is unavailable
when the source is compiled. The compiler emits immutable code `C(w)`, the
source is deleted, and only then is `x` delivered to a fixed executor `E`:

```text
source phase:      w -> C(w) -> commit -> delete w
late-input phase:  (C(w), x) -> E(C(w), x).
```

Exactness means `E(C(w), x) = U_w(x)` for every admissible `w` and every
`x in X`. The board uses all late inputs, not a sampled subset.

## 3. Action-Equivalence Theorem

Define

```text
w ~ v  iff  U_w(x) = U_v(x) for every x in X.
```

This is an equivalence relation and a congruence under source concatenation.
Let `M = {U_w : w in {T,N}*}` be the induced transformation monoid.

**Theorem 1 (exact quotient sufficiency and separation necessity).**

1. The quotient code `[w]` is sufficient: the decoder `E([w], x) = U_w(x)` is
   well-defined and exact.
2. For any deterministic exact code/executor pair, two inequivalent programs
   cannot share one code unless the executor has another source-dependent
   channel. Therefore an exact source-deleted interface needs at least `|M|`
   distinguishable messages, or `ceil(log2 |M|)` retained bits.

**Proof.** If `w ~ v`, both induce the same function, so evaluating their class
is well-defined. Conversely, if `w !~ v`, some `x*` separates them. If
`C(w) = C(v)`, deterministic `E` must return the same result on that code and
`x*`, contradicting exactness for one source. The message-count bound follows
by pigeonhole. QED.

The theorem does not require every exact compiler to assign identical bit
patterns to equivalent sources; it says inequivalent behaviors must remain
distinguishable. DAQC v1 chooses one canonical code per equivalence class.

## 4. Deferred-Context Theorem and No-Go Boundary

**Theorem 2 (finite quotient context bound).** If a source family induces
exactly `m` distinguishable late-input behaviors, exact source-deleted
execution requires at least `ceil(log2 m)` retained bits and can be implemented
with that many bits plus a fixed decoder when the classes are enumerated.
Source byte length is irrelevant to this bound only to the extent that many
long sources collide in the same semantic class.

For this board, `m = 34`, so five bits are impossible and six bits suffice.
Identity relations permit arbitrarily longer source words without increasing
the six-bit action code. This is semantic quotient compression, not compression
of arbitrary strings.

**No-go boundary.** For a free binary semigroup in which all length-`L` words
have different late-input behavior, there are `2^L` classes. Any exact
source-deleted code therefore requires at least `L` bits. More generally the
bound is `ceil(L log2 a)` for an alphabet of size `a` when all `a^L` words are
semantically distinct. DAQC cannot provide sublinear retained context without
real action-equivalence collisions. The CPU falsifier checks the binary result
exactly for every `L = 1,...,128` in tests and `L = 1,...,64` in the audit.

## 5. Noncommutative Translation-Reflection Witness

The generated transformations are the dihedral group of order 34. Every action
has the affine normal form

```text
A_(s,b)(x) = s*x + b mod 17,
s in {+1,-1}, b in Z17.
```

If `A_(s1,b1)` is followed by `A_(s2,b2)`, then

```text
(s1,b1) then (s2,b2) = (s2*s1, s2*b1 + b2 mod 17).
```

The frozen relations are

```text
T^17 = e
N^2 = e
N T N = T^-1 = T^16
N T N T = e.
```

Order matters. With left-to-right execution,

```text
T N (x) = -x - 1 mod 17
N T (x) = -x + 1 mod 17.
```

At the frozen witness `x = 0`, `TN(0) = 16` while `NT(0) = 1`. This prevents
the board from collapsing to a commutative count of generator occurrences.

## 6. Frozen Score-Blind Board

### 6.1 Constants

```text
protocol_id                 DAQC-D34-v1
modulus                     17
actions                     34
variants per action         4
source cases                136
late inputs per case        17 (all of Z17)
evaluation cells            2,312
SOURCE_SEED                 2026071501
LATE_INPUT_SEED             2026071502
CONTROL_SEED                2026071503
source commitment SHA-256   10d012c974431249fe9517983e0a9e6760d64bc08998c1bd5e3f1082e0130142
canonical board SHA-256     5a9e533757384ecdfe541f9b13ce0997b736fb5e4acc398e490d2815093839c7
canonical board bytes       53,920
source token bytes          2,895
source length range         1..67 operations
```

The seeds, schema, case order, canonical JSON encoding, and hashes are frozen.
Changing any of them requires a new protocol identifier. No score, model
output, benchmark result, or human preference participates in board creation.

### 6.2 Source construction

Actions are ordered as all translations `(+,b)` for `b = 0,...,16`, followed
by all reflections `(-,b)`. Each action receives four unique but exactly
equivalent source words:

1. `canonical`: deterministic nonempty normal-form representative.
2. `involution_padding`: inserts one or more `NN` identity blocks.
3. `conjugacy_padding`: inserts one or more `NTNT` identity blocks.
4. `cycle_padding`: inserts one or more `T^17` identity blocks.

Insertion positions are derived from `SOURCE_SEED` with SHA-256 ranking. The
generator checks every resulting word before accepting it. All 136 words and
case identifiers must be unique. Four programs within an action class must
have identical 17-output signatures; signatures across the 34 classes must be
distinct.

### 6.3 Commitment, source deletion, and late inputs

Board generation has two ordered phases:

1. Construct source cases, compile their action codes, and hash a canonical
   payload containing only source fields. This produces the frozen source
   commitment.
2. After that hash exists, use `LATE_INPUT_SEED` to permute all 17 inputs for
   each case and attach exact expected outputs. Hash the complete board.

The source commitment deliberately excludes `late_inputs` and
`expected_outputs`; the full board hash covers them. Thus changing late inputs
leaves the source commitment unchanged but fails the board hash and structural
audit.

For post-commitment direct execution, `SealedCode` contains exactly:

```text
code_index
```

It contains no case identifier, provenance, commitment, word, source tokens,
source bytes, program, output table, or retrieval key. Its `code_index` is in
`[0,33]` and has a six-bit logical encoding. The CPU auditor separately holds a
non-executable `CodeAuditEnvelope` with recipient case identifier, the global
source commitment, and donor provenance. The late-input executor receives only
the nested `SealedCode`, never that envelope. Source-retaining controls are kept
in separate code paths and are labeled as such.

The board is written with exclusive creation (`O_EXCL`) and read-only mode
`0444`; generation refuses to overwrite an existing path. The auditor requires
canonical JSON bytes and both frozen hashes.

## 7. Favorable Exact Controls

Every control is intentionally given exact algebra and the full source or
oracle information its definition requires. DAQC gets no credit from weak
baselines. Each control must score exactly `2,312 / 2,312` cells.

1. **Exact FST control.** A 34-state transition table consumes every `T` or `N`
   and emits the exact action code. It retains source only while compiling.
2. **Sequential control.** The direct interpreter executes every source token
   on every late input in order.
3. **Balanced-tree control.** Exact affine actions are composed in a balanced
   binary tree, then applied to the late input. This reduces composition depth
   but not total merge count.
4. **Direct control.** The preregistered oracle action label is applied directly
   to the late input. This is the finite-board ceiling.

If any exact control misses one cell, the package fails; there is no tolerance,
seed retry, majority vote, or score-based board repair.

## 8. Code Interchange Falsifier

The code must carry causal action information rather than merely correlate
with a case identifier. For every ordered pair of distinct actions `(a,b)`, the
six-bit code from donor `b` is inserted into a sealed recipient object from
`a`, while recipient identity and source commitment remain fixed.

The swapped object must execute donor behavior on all 17 inputs and must differ
from recipient behavior on at least one input. The exhaustive audit contains

```text
34 * 33 = 1,122 ordered distinct action pairs
1,122 * 17 = 19,074 interchange cells.
```

No sampled swaps are permitted. A failure is evidence that information leaked
through another field or that the code is not semantically interchangeable.

## 9. Reliability Theorem

Let `epsilon in (0,1]` be independent per-atomic-step success, `c_L` compiler
success for a length-`L` source, `rho` fixed-runtime-call success, and `K` the
number of post-commitment runtime calls.

```text
P_serial(L)   = epsilon^L
P_tree(L)     = p^(L-1)       if every one of L-1 merges must be correct
P_compiled(L) = c_L * rho^K.
```

**Theorem 3 (conditional reliability advantage).** If `K = O(1)`, `rho > 0`,
`epsilon < 1`, failures satisfy the stated independent fatal-error model, and
`-log(c_L) = o(L)`, then

```text
log(P_compiled / P_serial)
  = log(c_L) + K log(rho) - L log(epsilon) -> +infinity.
```

Thus compilation can asymptotically beat a fallible serial executor only when
compiler reliability does not decay exponentially as fast as the serial
chain. The result is conditional, not a claim that a neural compiler meets the
condition. Correlated errors, self-correction, nonfatal errors, verifier calls,
or source-dependent `K` require a different model.

A balanced tree has logarithmic dependency depth but still has `L-1` fallible
merges. Under the fatal independent-merge model it has success `p^(L-1)`; depth
alone does not remove reliability compounding.

The fixed CPU arithmetic witness uses exact rational values
`epsilon = p = rho = 99/100`, `c_64 = 999/1000`, `L = 64`, and `K = 1`. It
checks the formulas and that `c_64*rho > epsilon^64`. This is an algebra test,
not empirical evidence about Shohin.

## 10. Exact Resource Vector

The mandatory comparison vector is

```text
(parameters, retained bits, precision, source bytes, training examples,
 oracle calls, training FLOPs, inference FLOPs, sequential depth,
 external memory, external execution).
```

For the frozen CPU package it is:

| Field | Frozen value |
|---|---|
| parameters | `0` learned parameters |
| retained bits | `6` per sealed code; `816` logical bits for 136 codes; `0` retained source bits after sealing |
| precision | exact integers modulo 17; input/output need 5 logical bits; action code needs 6; no floating point |
| source bytes | `2,895` ASCII generator bytes across the board |
| training examples | `0` |
| oracle calls | `0` |
| training FLOPs | `0` |
| inference FLOPs | `0`; controls use exact integer/symbolic operations, counted separately rather than mislabeled as FLOPs |
| sequential depth | direct post-commit `1`; sequential/FST compile `1..67`; balanced-tree compile at most `ceil(log2 67) = 7`; post-compile execute `1` |
| external memory | canonical board `53,920` bytes; sealed-code logical payload `816` bits; non-executable audit envelopes `13,668` ASCII bytes |
| external execution | `true`: Python standard-library CPU symbolic execution |

Additional exact work counts for the four board controls are:

```text
sequential generator updates: 2,895 * 17 = 49,215
FST transitions:              2,895, then 2,312 direct applications
tree action merges:           2,895 - 136 = 2,759, then 2,312 applications
direct applications:          2,312
interchange applications:     19,074 logical donor cells
```

These are CPU falsifier costs. They are not projected neural inference costs.

## 11. Prior-Art and Equivalence Boundary

DAQC v1 must not be renamed prior art presented as a new primitive.

| Known method | Exact relationship to this package |
|---|---|
| Partial evaluation / multi-stage programming | The source is known before a late argument and specialized into residual code. DAQC is an instance of this protocol. |
| Finite-state transducer / syntactic monoid | The exact compiler is a 34-state FST over the finite transformation group. The quotient is a standard action/syntactic quotient. |
| Group normal form | The six-bit `(sign, offset)` representation is a dihedral normal form. |
| Parallel prefix / balanced product tree | The exact tree control computes the same product with logarithmic dependency depth and linear work. |
| Memoization / output tables | DAQC does not retain a 17-entry output table; it retains a six-bit action plus a fixed decoder. On this finite domain, an FST still exactly subsumes it. |
| Program synthesis / neural compiler | A future learned source-to-code map would be a neural compiler; this CPU package supplies no learning result. |
| CoT | DAQC emits no natural-language trace and does not preserve a token chain after commitment. CoT can emulate the computation, so this is an operational distinction, not a computability separation. |
| RNN / recurrent latent state | DAQC v1 executes a fixed sealed code in one late-input call rather than repeatedly updating learned hidden state. An RNN can implement the 34-state machine, so no expressive-class novelty is claimed. |
| Global workspace / scratchpad | No mutable workspace survives except the immutable six-bit code. A workspace can store the same code, so the distinction is interface and resource accounting. |
| Retrieval | Source is deleted and no source-dependent record is retrieved. The provenance hash is not decoded. |

Consequently, success on `D34` cannot establish a mechanism "beyond" exact
FSTs, group representations, or partial evaluation. The value of this package
is a hard interface, causal code-swap test, complete late-input board, and
resource/no-go accounting suitable for falsifying a later neural claim.

## 12. Mandatory Gates

All gates are binary and exact:

1. **Frozen bytes:** source and board hashes equal the constants in Section 6.
2. **Completeness:** exactly 34 actions, four unique source variants per action,
   all 17 late inputs, and 2,312 cells.
3. **Commit-before-late-input:** source commitment reconstructs from source-only
   fields; late-input mutation preserves that commitment but fails board audit.
4. **Source deletion:** sealed execution object has exactly one field,
   `code_index`, and reproduces all 2,312 cells from code plus late input;
   identities and provenance remain outside the executable object.
5. **Relations:** all four presentation/identity relations hold on all 17 inputs
   and the `TN != NT` witness is present.
6. **Faithfulness:** all 34 action signatures are unique; each action's four
   source words collide only within its intended equivalence class.
7. **Favorable controls:** exact FST, sequential, tree, and direct controls each
   score 2,312/2,312.
8. **Code interchange:** all 19,074 donor cells follow donor behavior and every
   distinct ordered pair separates from recipient behavior somewhere.
9. **Capacity:** deterministic five-bit assignment produces explicit behavioral
   collision witnesses; six bits admit all 34 codes without collision.
10. **Linear-bit no-go:** the free binary-semigroup counting bound is exact at
    every preregistered length.
11. **Reliability arithmetic:** exact rational formulas and the fixed witness
    pass without floating-point approximation.
12. **Resource ledger:** every field in the exact vector is present; no model,
    training data, oracle, learned parameter, training FLOP, or GPU is hidden.
13. **Immutability/tamper:** board creation refuses overwrite, mode is `0444`,
    noncanonical bytes fail, and any covered field mutation fails audit.

No failed gate may be waived. No seed may be retried. No case may be removed.
No favorable control may be weakened. Any repair after looking at a score must
be assigned a new protocol identifier and reported as post-registration.

## 13. Execution

Run the exhaustive tests:

```bash
python3 -m unittest -v pipeline.test_daqc_cpu_falsifier
```

Run every frozen in-memory gate:

```bash
python3 pipeline/daqc_cpu_falsifier.py self-check
```

Create and audit an immutable board outside the repository:

```bash
python3 pipeline/daqc_cpu_falsifier.py generate /tmp/daqc_d34_v1.json
python3 pipeline/daqc_cpu_falsifier.py audit /tmp/daqc_d34_v1.json
```

This package ends at the CPU falsifier boundary. It must not be cited as proof
that Shohin reasons until a separate frozen neural protocol passes its own
held-out, source-deleted, code-interchange, direct-control, and transfer gates.
