# R12 SD-CST Complete Physical-Record Front-End v1.1 Audit Preregistration

**Status:** deterministic read-only source draft; no audit output exists

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
