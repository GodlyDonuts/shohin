# R12 VAMT v3 Bounded Program-Machine Theory

**Status:** CPU mechanics candidate only. Neural code, data generation, fitting,
accelerator use, capability claims, and novelty claims are not authorized.

**Protocol:** `R12-VAMT-FULL-MACHINE-FALSIFIER-v3`

## 1. Decision

VAMT v2 is rejected because its CPU artifact tested a host-selected one-operation
kernel rather than the declared program machine. V3 replaces that partial object
with one complete, fixed-cycle, bounded interpreter whose source addressing,
program counter, accumulator updates, rejection state, halting, and terminal
serialization are all executed in the candidate path.

This repair does **not** establish a new primitive. Under a fixed permutation of
the ten decimal categories, the proposed compiler plus tied executor is
isomorphic to a favorable Pointer Network plus tied Mealy/NPI controller with
the same cases, labels, state, cycles, parameters, and MAC accounting. A future
neural result could therefore support only an optimization, data-efficiency, or
vocabulary-alignment claim relative to that matched control.

The CPU artifact uses oracle-injected categorical tables and Python lookup. A
pass is external symbolic execution, not autonomous arithmetic or reasoning.

## 2. Frozen capability bound

The only admitted family is bounded unsigned decimal register execution:

- source length `T <= 256` tokens;
- exactly `L = 8` instruction slots;
- source operands contain at most `W = 16` decimal digits;
- the accumulator has `D = W + 1 = 17` decimal digits;
- opcodes are `LOAD`, `ADD`, `SUB`, and `HALT`;
- `SUB` that would produce a negative result rejects;
- arithmetic overflow beyond 17 digits rejects;
- every instruction slot receives 17 executor cycles;
- the executor always charges `8 * 17 = 136` cycles;
- the serializer always charges 17 cycles;
- no retry, verifier repair, parser repair, or result-conditioned extra compute
  is admitted.

The tokenizer artifact is
`artifacts/shohin-tok-32k.json`, SHA-256
`87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4`.
Each decimal digit is one frozen token:

| Digit | Token ID |
|---:|---:|
| 0 | 28 |
| 1 | 29 |
| 2 | 30 |
| 3 | 31 |
| 4 | 32 |
| 5 | 33 |
| 6 | 34 |
| 7 | 35 |
| 8 | 36 |
| 9 | 37 |

The source may contain other tokens, but every referenced operand span must
contain only these ten token IDs.

## 3. Complete machine state

One executor state is

```text
q = (
  pc:uint3,
  phase:uint5,
  source_cursor:257-category,
  carry_or_borrow:bit,
  accumulator:10-category[17],
  invalid:bit,
  halted:bit
)
```

`source_cursor = 256` is the unique pad cursor. The program is an immutable
eight-element sequence

```text
instruction = (opcode:4-category, start:uint8, end:uint8)
```

with inclusive source endpoints. The serializer state is

```text
r = (
  read_cursor:17-category,
  seen_nonzero:bit,
  serializer_halted:bit,
  write_cursor:18-category,
  status:{RUN, ACCEPT, REJECT},
  output_tokens:uint16[17]
)
```

The host loop invokes the same executor transition 136 times and the same
serializer transition 17 times. The host loop index does not choose an opcode,
program counter, operand, cursor, output, or stop time.

## 4. Executor semantics

### 4.1 Instruction entry and source cursor

At phase zero, a non-`HALT` instruction is structurally valid exactly when

```text
opcode in {LOAD, ADD, SUB}
0 <= start <= end < source_length <= 256
end - start + 1 <= 16
```

The initial cursor is `end`. At phase `i < 16`, the expected cursor is
`end - i` while that position is at least `start`; otherwise it is the pad
cursor. Any mismatch between retained and expected cursor sets sticky
`invalid = 1`. A source token outside the frozen decimal codebook also sets
`invalid = 1`. Phase 16 always supplies right digit zero and requires the pad
cursor.

Over-width spans are rejected, never truncated. Reversed, negative, high, and
nondigit spans are rejected by the candidate machine, not skipped by the test
harness.

### 4.2 LOAD

At phases 0 through 15, `LOAD` writes the addressed source digit or zero padding
to the corresponding accumulator cell. At phase 16 it writes zero. Thus a new
source operand always initializes the entire 17-digit accumulator.

### 4.3 ADD and SUB

For `op in {ADD, SUB}`, every phase consumes

```text
(op, accumulator[phase], source_digit, carry_or_borrow)
```

and produces

```text
(new_accumulator_digit, new_carry_or_borrow).
```

There are exactly

```text
2 operations * 10 left digits * 10 right digits * 2 carry states = 400
```

local contexts. One table is tied across all 17 positions and all eight slots.
Phase 16 is an ordinary transition with right digit zero. It therefore consumes
the carry from phase 15 instead of dropping it. This repairs v2's chaining bug:
`9 + 9 = 18`, followed by `+ 9`, produces `27` because the high accumulator
cell participates in the second operation.

After phase 16, a remaining ADD carry or SUB borrow sets sticky `invalid = 1`.
A negative subtraction therefore executes all 17 SUB transitions, then rejects;
the harness does not prefilter it.

### 4.4 Program advance, HALT, and masking

After a valid non-`HALT` phase 16, carry resets to zero, `pc` increments,
`phase` resets to zero, and the next instruction cursor initializes from its
endpoint. Executing a non-`HALT` instruction in slot seven sets `invalid = 1`
and `halted = 1`, because the program omitted `HALT`.

`HALT` acts only on phase zero. Its first cycle sets `halted = 1`; every
remaining executor cycle is masked but charged. Once either `invalid` or
`halted` is set, all subsequent executor calls leave machine state unchanged.
Instructions after `HALT` are never validated or observed.

## 5. Serializer semantics

Invalid or non-halted machine state deterministically returns `REJECT` with
length zero. No modular accumulator digits are exposed after rejection.

For a valid halted state, the serializer scans accumulator cells from position
16 to position zero. Its complete context is

```text
(seen_nonzero, digit, at_last)
```

with 40 possible contexts. Its complete categorical outcome is

```text
(emit, next_seen_nonzero, halt, symbol_digit).
```

The factorized full-logit family is therefore

```text
W_emit   : 40 x 2
W_seen   : 40 x 2
W_halt   : 40 x 2
W_symbol : 40 x 10
```

The canonical table suppresses leading zeroes, emits one zero for the all-zero
register, and halts exactly at the final cell. Early halt, missing final halt,
wrong emission, wrong state update, or wrong symbol is visible to the
independent whole-program scorer.

## 6. Candidate/reference separation

The candidate receives its 400 executor outcomes and 40 serializer outcomes as
mandatory constructor inputs. It never calls the scorer or constructs expected
whole-program answers. The reference interpreter independently:

1. checks each executed source span;
2. decodes the frozen digit token IDs;
3. applies ordinary host integer `LOAD`, `ADD`, and `SUB` semantics;
4. rejects negative or 17-digit-overflow states;
5. requires an executed `HALT`; and
6. constructs the expected output token sequence.

Candidate and reference codebooks are separately constructed immutable maps.
There is no mutable `TRUTH_TABLE` or serializer global. The finite falsifier
poisons the executor alone, serializer alone, and both jointly; every poison
must be rejected while the canonical candidate remains exact.

The canonical candidate tables are oracle-injected. Their lookup count is
reported as external symbolic execution. Candidate/reference independence
prevents a circular pass; it does not make the candidate autonomous.

## 7. Deterministic finite board

The CPU board contains all 400 local executor contexts, all 40 serializer
contexts, and exactly 152 complete program executions:

| Family | Executions |
|---|---:|
| Eight programs at every operand width 1 through 16 | 128 |
| Post-HALT masking variants | 16 |
| Malformed spans | 5 |
| Missing HALT | 1 |
| Seven-ADD maximum bound | 1 |
| Terminal-carry reuse | 1 |
| **Total** | **152** |

The width sweep contains exactly 32 negative subtractions. Each must invoke all
17 SUB transitions before rejection. The malformed board contains a 17-digit
span, reversed span, negative start, high endpoint, and nondigit span. The
seven-ADD case adds seven 16-digit all-nine operands from a zero accumulator
and must serialize `69999999999999993`. The carry-reuse case must serialize
`27`.

Every complete execution invokes 136 executor and 17 serializer cycles. The
board therefore charges exactly 20,672 executor cycles and 2,584 serializer
cycles, including all masked cycles.

A finite pass proves no scale extrapolation or learnability. It only makes the
bounded semantics executable and falsifiable before any neural work.

## 8. Exact parameter ledger

The immutable Shohin base has 125,081,664 parameters. The canonical minimal
neural witness **within the declared factorized full-logit family** is:

| Component | Parameters |
|---|---:|
| Slot embeddings `8 x 128` | 1,024 |
| Global projection `576 x 128 + bias` | 73,856 |
| Source key `576 x 128` | 73,728 |
| Start/end query projections `2 x 128 x 128` | 32,768 |
| Opcode head `128 x 4 + bias` | 516 |
| Executor factorized logits `400 x 10 + 400 x 2` | 4,800 |
| Serializer factorized logits `40 x (2+2+2+10)` | 640 |
| **Additional** | **187,332** |
| **Total** | **125,268,996** |
| **Headroom below 149,999,999** | **24,731,003** |

This is not a globally minimality claim. It is only the smallest listed member
of one frozen realization family.

## 9. State and information ledger

The packed program/private state is

```text
program   144 bits
machine    88 bits
serializer 14 bits
total      246 bits = 31 bytes after padding
```

One byte-addressed realization uses 53 bytes: 24 bytes for eight opcodes and
their uint8 endpoints, 24 bytes for executor-private state, and 5 bytes for
serializer-private state. Other fixed buffers are:

| Buffer | Bytes |
|---|---:|
| Immutable source `256 x uint16` | 512 |
| Output tokens + length + status | 36 |
| Digit codebook `10 x uint16` | 20 |
| Executor temporary | 53 |
| Serializer temporary | 69 |
| Post-base compiler temporary peak | 750,232 |
| Compiler phase including source and codebook | 750,764 |
| Post-compiler serializer live set | 688 |

The base model activation/allocation peak is unknown and must be measured by a
future executable preregistration. No exact end-to-end VRAM peak may be claimed
from these sidecar numbers.

One program compiler target contains

```text
8 * (2 opcode bits + 8 start bits + 8 end bits) = 144 bits.
```

The executor targets contain `400 * 5 = 2,000` bits. Serializer targets contain
`40 * 7 = 280` bits. Executor plus serializer targets therefore contain 2,280
bits. Charging their context fields as well yields 6,520 bits.

## 10. Compute ledger

The canonical compiler matrix-MAC ledger is:

| Component | MACs |
|---|---:|
| Global projection | 73,728 |
| Source keys | 18,874,368 |
| Start/end queries | 262,144 |
| Opcode head | 4,096 |
| Pointer scores | 524,288 |
| **Compiler total** | **19,738,624** |

The dense one-hot equivalent for 136 executor calls is 652,800 MACs. The dense
one-hot equivalent for 17 serializer calls is 10,880 MACs. Their total is
663,680 MACs, and the full non-base dense equivalent is 20,402,304 MACs.

Python table lookup is not a neural MAC and is separately labeled external
symbolic execution. These figures define the neural matched-control budget; they
do not describe the CPU artifact's wall-clock cost.

## 11. Collapse and matched-control result

The complete bounded object is a deterministic finite-state transducer. Its
246-bit packed state gives an elementary state-count upper bound of `2^246`, and
its recurrent execution unrolls into a finite acyclic circuit at fixed bounds.
That fact alone is not a resource-preserving rejection under the R12 charter.

The stronger rejection is direct: rename the ten digit categories by one fixed
permutation. The compiler remains a pointer classifier, and the tied 400-context
update plus 40-context serializer remains a finite Mealy/NPI controller. A
Pointer Network plus tied Mealy/NPI control can receive the same source, program
labels, transition labels, serializer labels, 246 retained bits, 136+17 cycles,
187,332 sidecar parameters, and 20,402,304 dense-equivalent MACs. The behaviors
are conjugate under the fixed digit permutation.

Therefore VAMT v3 has no primitive-level or inherent architectural resource
advantage over this favorable control. The only reopenable empirical question
is whether initializing the digit symbols in Shohin's frozen vocabulary yields
better optimization or sample efficiency than the matched permuted control.
That is a training-protocol conjecture, not a new reasoning mechanism.

## 12. Gates and authorization boundary

The following must all occur before a neural preregistration may even be
drafted:

1. the new CPU source and tests pass Ruff, `py_compile`, and every supplied
   test;
2. the canonical JSON report is deterministic and hash-bound;
3. a fresh exact-byte hostile reviewer regenerates the report and reproduces
   all tests;
4. the reviewer confirms candidate/reference independence, complete machine
   execution, exact cycle counts, and every ledger;
5. the reviewer explicitly returns theory and CPU GO for the narrow
   optimization/data-efficiency conjecture.

Even that review would authorize only a separate neural preregistration. It
would not authorize fitting. A future preregistration must freeze fresh
confirmation generation, grouping, overlap policy, seeds, custody, favorable
Pointer/Mealy control, permuted-vocabulary control, examples, oracle calls,
training FLOPs, inference FLOPs, state, source bytes, precision, and stopping
rules before any H100 job.

Current authority remains:

```text
CPU mechanics implementation: allowed for falsification
Neural preregistration:       NO-GO pending independent review
Neural implementation:       NO-GO
Data generation:             NO-GO
Fitting / H100:              NO-GO
Reasoning claim:             NO-GO
Novel primitive claim:       rejected by matched-control isomorphism
```
