# R12 VAMT v2 Independent Review Result

**Status:** REJECTED. No neural implementation, fit, accelerator allocation, or
reasoning claim is authorized by this result.

## Exact reviewed tuple

| Artifact | SHA-256 |
|---|---|
| `R12_VOCABULARY_ALIGNED_MICROCODE_TRANSDUCER_THEORY.md` | `69d736c6a6f8e5504e0b11674ffc2b46dc1664901418660aec3936f7ab583e06` |
| `pipeline/vamt_symbolic_falsifier.py` | `37c0b6610ef70cf430dd62d205da0f9b367f7167b10b1cd4b5b462f49abf3c38` |
| `pipeline/test_vamt_symbolic_falsifier.py` | `537b719104546491ce99390167a23565f0c5ce65115dfcbae2ec4ce60b93e6cf` |
| `scratchpad/vamt_symbolic_falsifier_v2.json` | `28364d691a34425ec29de8ae8e9da4623c962a941602b2546761b3858b299e15` |

The report's embedded payload SHA-256 was
`28a31ada9a2ead122fd5d9dc3557dbd6be52c53a593efb6c252a2a7fc6fd6225`.
The independent reviewer matched all hashes, regenerated the report byte for
byte, and reproduced 15 passing supplied tests.

## Verdicts

| Surface | Verdict |
|---|---|
| Theory | **NO-GO** |
| CPU mechanics | **RESTRICTED GO** for counterexample and isolated local-kernel work only |
| Permission to draft executable neural preregistration | **NO-GO** |
| Neural fitting | **NO-GO** |

## Blocking findings

1. The CPU artifact does not implement the declared machine. It receives a
   host-selected operation and two prepared equal-width digit tapes. It does not
   execute a program counter, `LOAD`, `HALT`, source spans, masked inactive
   cycles, invalid state, or chained accumulator updates.
2. Negative subtraction is skipped by the audit harness rather than rejected by
   the machine. Terminal borrow is not connected to an invalid state or to a
   serializer rejection path.
3. The carry-slot recurrence is wrong for chained operations. A `W`-digit result
   plus terminal carry requires the next operation to consume `W+1` accumulator
   digits, or an equivalent explicitly proved transition. The v2 ledger charges
   only `W` cycles. Its capability bound is width 16 while its gates demand width
   64.
4. Reference checks are circular. Candidate and expected behavior share mutable
   `TRUTH_TABLE` and serializer globals, so jointly poisoned reference semantics
   can preserve `all_pass`. The position count repeats table equality rather
   than propagating machine state.
5. Pointer and host-boundary claims are not exercised. The harness supplies
   relocated spans manually, silently truncates over-width spans, and accounts
   only one three-digit `ADD`, omitting compiler, program, serializer, and
   invalid-state paths.
6. State, compute, and collapse certificates are incomplete. The state pass
   checks only two byte sums, target information is a Boolean rather than a
   count, maximum MACs omit a declared projection, and the Mealy state bound
   omits mutable machine fields while returning `pass=True` unconditionally.
7. Controls and fresh-split policy are directionally responsible but not frozen
   enough for preregistration. Shuffled labels are sanity controls, not causal
   comparators, and confirmation generator bytes, seed, cardinalities, grouping,
   overlap policy, and custody remain unspecified.

## Surviving evidence

The v2 CPU artifact remains useful only as evidence that:

- one exact categorical add/sub transition table has 400 local contexts;
- a tied single-operation kernel can replay bounded unsigned addition and
  admitted nonnegative subtraction when the host has already selected the
  operation and prepared both tapes;
- one tied unsigned serializer table can suppress leading zeros on a prepared
  well-formed result tape;
- the proposal collapses to ordinary finite-state and recurrent machinery and
  therefore supports no new-primitive claim.

It is not evidence for compilation, complete program execution, host-free
inference, exact resource accounting, autonomous arithmetic, or reasoning.

## Next permitted work

1. Specify one complete bounded machine with exact instruction timing, terminal
   carry consumption, invalid-state propagation, serializer state, and one-call
   host boundary.
2. Replace shared mutable reference tables with an independent immutable oracle
   and adversarial joint-poison tests.
3. Execute full `LOAD`/`ADD`/`SUB`/`HALT` programs over source spans, including
   chained carry, negative-subtraction rejection, over-width rejection, inactive
   cycle charging, and terminal serialization.
4. Recompute exact state, target-information, parameter, and operation ledgers at
   one consistent capability width.
5. Freeze confirmation generation and matched-control contracts only after the
   repaired theory survives another independent review.

No Shohin checkpoint, dataset, scheduler state, remote resource, or accelerator
is authorized by this document.
