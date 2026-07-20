# R12 SD-CST Complete Physical-Record Front-End v1.1 Audit Preregistration

**Status:** closed. Exact source `a9a8d9a06a4a16c82385ae31ce346edda0d25d2f`
produced sole job `694209` on H100 `evc23`; full result:
`R12_SD_CST_COMPLETE_PHYSICAL_RECORD_BUS_V1_1_AUDIT_RESULT.md`.

This post-hoc audit reads only the same 2,000-semantic consumed-training
heldout partition already evaluated by v1.1. Development and confirmation remain
unreachable. It reconstructs exact endpoint SHA-256
`46697b3942fdfd2edfec06cea6cb119ad507adcdee4a99330fd06bc79e5b3e88`
and does not optimize or modify any tensor.

For each of four heldout renderers and each of the six declaration queries, it
records top-one target-span exactness, probability mass on the target span,
target NLL, and a seven-way confusion: the three binding occurrences, three
initial-list occurrences, or other bytes. It also reproduces all-three binding
and all-three initial exactness.

The audit decides only the next representation hypothesis:

- initial queries selecting binding occurrences supports separate
  binding/initial key banks;
- selecting the wrong initial occurrence supports stronger occurrence/position
  conditioning;
- selecting other bytes supports a declaration-local nonlinear token parser;
- diffuse low target mass everywhere rejects a readout-only repair and requires
  adapting declaration-local token memory.

No threshold is promoted, no failed run is rescored, and no reasoning or fresh
generalization claim can follow from this audit.

Observed result: all 48,000 top-one decisions select one of the six true entity
spans and none select other bytes, but the middle initial occurrence is only
0--10.35% exact. This admits an occurrence-role classifier, not more v1.1
epochs or a broader encoder.
