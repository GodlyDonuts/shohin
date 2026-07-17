# R12 Packet-on-Lattice Carry Cell CPU Preregistration

**Protocol:** `R12-PLCC-CPU-v1`

**Status:** **FROZEN 2026-07-17 before any PLCC neural fit, GPU pilot, Shohin
score, or architecture integration.** This contract authorizes only the
deterministic standard-library CPU falsifier in
`pipeline/plcc_cpu_falsifier.py` and its focused tests.

**Decision:** **NARROW GO for the CPU mechanics falsifier only. NO-GO for a
novelty claim, SoTA claim, model integration, or neural promotion.** A passing
falsifier must record PLCC as exactly equivalent to a favorable explicit
`(cursor, carry)` recurrent transducer on the frozen board. Mechanical collapse
to that control is the expected hostile result, not a win.

**Claim boundary:** no novelty claim, no SoTA claim, no Shohin capability
claim, no neural learnability claim, no natural-language reasoning claim, no
GPU path, no fit, no cluster job, and no production-data interface. The CPU
result specifies oracle mechanics and a classical equivalence boundary only.

## 1. Frozen question

Can decimal carry or borrow be transported with exactly one mutable arithmetic
bit when the cursor is encoded by the physical support of a single hard packet,
without generated text, causal KV state, a result tape, learned addressing,
host execution, or terminal metadata entering the local transition?

The finite falsifier answers two narrower questions:

1. Does the proposed packet mechanic implement the exact local and two-column
   oracle under the stated channel restrictions?
2. Does it provide any mechanical advantage over an ordinary explicit
   recurrent `(cursor, carry)` state machine?

It does not answer whether a neural model can learn the cell or whether the
cell improves language-model reasoning.

## 2. Exact mechanism

Operands are represented as immutable decimal columns in least-significant
first order:

```text
B = (op, ((a_0,b_0),...,(a_(w-1),b_(w-1))))
```

where `op` is `ADD` or `SUB` and each digit lies in `{0,...,9}`.

Mutable runtime state is one packet:

```text
P_t = (location_t, polarity_t).
```

The contract is:

- exactly one lattice position is occupied;
- packet location is the cursor;
- packet polarity is exactly one carry or borrow bit;
- no other mutable field exists;
- the same tied local transition is used at every location;
- the local transition receives exactly `(op,a_p,b_p,c_p)`;
- position, width, terminality, prior digits, future digits, and history are
  absent from that local signature;
- the local digit is ephemeral;
- on a nonterminal slot, only the shifted packet with overwritten polarity is
  returned;
- on the terminal slot, only `(final_digit, terminal_carry)` is emitted.

The local transition is:

```text
ADD:
    u = a_p + b_p + c_p
    digit = u mod 10
    next_carry = 1[u >= 10]

SUB:
    u = a_p - b_p - c_p
    digit = u mod 10
    next_borrow = 1[u < 0]
```

After a nonterminal transition:

```text
location_(t+1) = location_t + 1
polarity_(t+1) = next_carry_or_borrow.
```

No generated token, causal KV entry, intermediate result symbol, result tape,
learned address head, verifier result, parsed state, or host arithmetic result
may enter the next cycle.

The CPU falsifier itself uses host integer arithmetic to define and verify the
oracle. The report field
`host_arithmetic_calls_during_neural_inference=0` is a frozen resource claim
about the proposed neural interface, not a claim that the CPU auditor avoids
integer arithmetic.

## 3. Necessary-state boundary

### Proposition 1: one arithmetic bit is locally sufficient

For a readable current operand column and operation, `(a_p,b_p,c_p)` uniquely
determines `(digit_p,c_(p+1))` for both frozen operations. Exhausting all

```text
2 operations * 10 a digits * 10 b digits * 2 incoming bits = 400 cells
```

is therefore a complete local oracle audit.

### Proposition 2: one total state bit is not sufficient at width two

At width two, the runtime states are:

```text
(location,polarity) in {(0,0),(0,1),(1,0),(1,1)}.
```

The CPU falsifier must exhibit a deterministic operand-board witness separating
every one of the six unordered pairs by their terminal endpoint. Four pairwise
distinguishable states require at least:

```text
ceil(log2(4)) = 2 logical bits.
```

PLCC assigns one bit to packet polarity and encodes the cursor in packet
support. It does not compress the complete width-two runtime state into one
bit.

### Proposition 3: exact recurrent equivalence

Define the coordinate map:

```text
Packet(location,polarity) <-> RecurrentState(cursor,carry).
```

The map is a bijection. Both systems read the same immutable source column,
apply the same 400-cell tied transition, advance one cursor position, retain
one arithmetic bit, use the same sequential depth, and expose the same terminal
endpoint. Therefore PLCC is a finite-state transducer and a coordinate change
of the explicit recurrent control on this finite board.

If the CPU implementation does not prove exact endpoint and resource-vector
equivalence, the run fails closed as a hidden channel or implementation error.
It must not reinterpret a mismatch as evidence of greater expressivity.

## 4. Frozen finite boards

### 4.1 Local table

The local board contains all 400 cells. Each cell is checked in every valid
position for widths one through four:

```text
widths                              1,2,3,4
position/width contexts                    10
terminal contexts per cell                  4
nonterminal contexts per cell               6
total contextual observations           4,000
```

The tied local function signature must be exactly:

```text
(op,a_p,b_p,c_p)
```

The canonical newline-delimited local-table commitment is:

```text
sha256 6c21e29e5341a3343cab76edeadb613d71a83bb00c2b7c1438d24c2160c0c7e2
```

### 4.2 Two-column board

The trajectory board exhausts:

```text
2 operations
* 10^4 assignments to (a_0,b_0,a_1,b_1)
* 2 initial carry/borrow bits
= 40,000 cases.
```

The endpoint under test is the second-column digit and terminal carry or
borrow. There is deliberately no full result tape. This board tests transport
and terminal consumption of the one-bit state, not full-number emission.

The canonical newline-delimited board commitment is:

```text
sha256 1911674b7ea403ac70a4de0f6cd04f1f2e99f62df84cd1d49ce38dcd11322181
```

## 5. Frozen interventions

### 5.1 Source-prefix deletion

For every one of the 40,000 two-column cases:

1. execute the first column;
2. delete the completed source prefix;
3. normalize the packet support to the first slot of the one-column suffix;
4. retain only packet polarity;
5. run the suffix to its endpoint.

The endpoint must be bit-identical to the undeleted baseline in all 40,000
cases. Any dependency on the completed prefix is a failure.

### 5.2 Different-carry donor swaps

For each operation, each recipient carry class, and every possible suffix
column, construct unrelated prefixes producing opposite carry classes. Swap
the donor polarity into the recipient packet while keeping recipient support
and source suffix fixed.

```text
2 operations * 2 recipient carries * 100 suffixes = 400 swaps.
```

All 400 endpoints must equal the oracle under donor carry. All 400 must differ
from the opposite recipient-carry endpoint. A failure rejects causal carry
transport.

### 5.3 Same-carry sham swaps

For each operation and carry class, select two distinct prefixes that produce
the same carry but different discarded local digits. Across all 100 suffix
columns, swapping their packets must leave the recipient endpoint unchanged:

```text
2 operations * 2 carry classes * 100 suffixes = 400 shams.
```

### 5.4 Cursor-location swaps

For all 20,000 two-column operand boards and both packet polarities, move the
same-polarity packet between locations zero and one. The selected local source
column must follow packet location exactly, while pre-scatter polarity remains
unchanged:

```text
40,000 swap pairs
80,000 selected-column observations.
```

This intervention distinguishes support-coded cursor state from the arithmetic
bit.

### 5.5 One occupancy and one bit

For every location and polarity at widths one through four:

- occupancy must be a binary one-hot vector with sum one;
- the packet type must be exact, not a subclass;
- dataclass fields and slots must be exactly `location` and `polarity`;
- no dynamic attribute dictionary may exist;
- packet-to-recurrent-to-packet conversion must round-trip exactly.

Any extra field, subclass, out-of-range support, dynamic payload, nonbinary
polarity, or non-one-hot support fails closed.

### 5.6 Zero intermediate emission

Across all 40,000 trajectories:

- the first cycle must return exactly a `Packet`;
- the terminal cycle must return exactly an `Endpoint`;
- intermediate emitted symbols must equal zero;
- generated tokens must equal zero;
- generated causal KV bytes must equal zero;
- retained result-tape slots must equal zero.

Auditor-only reflection of the ephemeral local output is non-causal and cannot
be passed into the next transition.

## 6. Favorable explicit recurrent control

The control directly stores:

```text
RecurrentState(cursor,carry).
```

It uses an independent exact decimal reference transition and receives the same
immutable source board. It is deliberately favorable and must score all
40,000 endpoints exactly.

At width two, both arms are charged:

```text
mutable arithmetic payload bits       1
cursor states                          2
cursor logical bits                    1
total runtime states                   4
total logical state bits               2
local transition cells               400
sequential depth                       2
result tape slots                      0
external execution calls              0
```

The resource vectors and all 40,000 endpoints must match exactly. The report
must set:

```text
mechanical_verdict = equivalent_to_explicit_recurrent_control
novel_reasoning_primitive_supported = false
neural_pilot_authorized_by_this_report = false
```

## 7. Promotion and rejection gates

The CPU mechanics contract passes only if every gate is true:

1. all 400 local cells match the independent decimal reference;
2. all 4,000 position/width/terminal observations are invariant;
3. the local signature has exactly four authorized arguments;
4. both frozen commitments match;
5. all 40,000 PLCC endpoints are exact;
6. all 40,000 source-prefix deletions are invariant;
7. all different-carry swaps follow the donor and diverge from the recipient;
8. all same-carry sham swaps are invariant;
9. all cursor swaps select the support-addressed source column;
10. every reflected packet has one occupancy and one payload bit;
11. intermediate token, KV, result-tape, verifier, learned-address, and external
    execution channels are zero;
12. all four width-two states are pairwise distinguishable;
13. the explicit recurrent control is exact;
14. PLCC and the recurrent control are endpoint-, state-, and resource-equivalent.

Failure of any gate yields `mechanical_verdict=mechanics_rejected` and a
nonzero process exit status. Passing all gates does not authorize a neural
pilot under this document; a separate preregistration with matched training,
seeds, FLOPs, recurrent controls, and held-out data would be required.

## 8. Machine-readable report

The report contains no wall-clock time, hostname, random identifier, model
score, or environment-dependent field. Canonical JSON uses sorted keys,
compact separators, ASCII, and one trailing newline. The report includes a
SHA-256 commitment over its content before the commitment field is attached.

Commands:

```bash
python3 -m pipeline.plcc_cpu_falsifier
python3 -m pipeline.plcc_cpu_falsifier --pretty
python3 -m pipeline.plcc_cpu_falsifier --output /tmp/plcc_report.json
```

`--output` publishes a read-only file once and refuses to overwrite an existing
target.

## 9. Required verification

Before this CPU artifact is accepted:

```bash
python3 -m pytest -q pipeline/test_plcc_cpu_falsifier.py
ruff check pipeline/plcc_cpu_falsifier.py pipeline/test_plcc_cpu_falsifier.py
python3 -m py_compile \
  pipeline/plcc_cpu_falsifier.py \
  pipeline/test_plcc_cpu_falsifier.py
git diff --check -- \
  R12_PACKET_ON_LATTICE_CARRY_CELL_PREREG.md \
  pipeline/plcc_cpu_falsifier.py \
  pipeline/test_plcc_cpu_falsifier.py
```

## 10. Interpretation boundary

A successful report means only that a one-bit arithmetic payload can be moved
by a support-coded cursor on this finite oracle board without an intermediate
text or result-tape channel. It simultaneously proves that the complete state
still has four distinguishable width-two configurations and that PLCC is
exactly an explicit recurrent finite-state transducer in different coordinates.

No result from this protocol may be described as a new computational class, a
new reasoning primitive, a Shohin capability gain, or evidence that a neural
network can learn the mechanism.
