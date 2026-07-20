# R12 ER-CST Neural Adapter Preregistration Amendment v1.1

**Protocol:** `R12-ER-CST-v1.1-neural-adapter`

**Status:** closed pre-board after the subsequent audit found eight event slots
cannot represent depth eight plus explicit pre-apply HALT. No board seed, training
seed, H100 job, output, development access, or confirmation access exists. V1.2 is
defined by `R12_ER_CST_NEURAL_ADAPTER_PREREG_AMENDMENT_V1_2.md`.

## Closed v1 defect

Exact source commit `0159bd4` froze a parameter-audited compiler for initial
state, rule cards, opcode bindings, and HALT. A subsequent board-interface audit
found that its public compilation result omitted the late-query category and
query pointer. It could therefore evaluate terminal state but could not invoke
the preregistered categorical reader to produce an answer.

V1 is closed before any board seed or scored byte existed. This is an interface
omission, not a neural result.

## Sole v1.1 change

`compile_rule_program` now accepts a separate late-query byte tensor and validity
mask, invokes the inherited confirmed `compile_query_with_evidence` path exactly
once, and includes its three-class query logits and source pointer logits in the
compiler result. The query path is frozen; it receives no new parameter and no
new trainable tensor.

Every other contract remains unchanged:

- same confirmed parent and byte-identical reconstruction;
- same twelve physical program records;
- same thirteen new compiler tensors;
- same 98 compiler plus four motor trainability tensors and contract hash;
- same 192,421,167 complete parameters, 11,715,616 trainable parameters, and
  7,578,833 headroom below 200M;
- same source-deletion boundary, now explicitly including the categorical query;
- same permitted supervision, controls, optimizer boundary, and development
  gates; and
- no final-state, answer, trajectory, executor, or scorer supervision.

The full adapter tests must prove query/category/pointer shape, source-only query
compilation, gradient isolation, exact parent reconstruction, and unchanged
parameter certificates before v1.1 source can freeze. A board seed may be drawn
only after the repaired source and builder are committed and pushed.
