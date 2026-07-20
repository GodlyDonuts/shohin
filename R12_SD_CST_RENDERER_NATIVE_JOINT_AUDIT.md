# R12 SD-CST Joint Renderer Compiler Post-Hoc Audit

**Status:** implemented before execution

**Purpose:** distinguish dead optimization from cascading record-address errors
after the preregistered joint control failed complete-record gates

This is an explicitly post-hoc diagnostic over the already-consumed renderer
orbit. It is not a score, a model-selection gate, or a reasoning claim. It may
read only the consumed training rows, exact rejected orbit checkpoint, and exact
rejected joint checkpoint. Development and confirmation are unreachable.

The audit reports per-slot rather than all-slots-at-once exactness for line
address, event address, kind, amount, and identity at exact initialization and
the rejected endpoint. Two model-state interventions are diagnostic only:

1. uniform pooling over the gold source-line span before the unchanged native
   slot decoder, measuring kind and amount; and
2. uniform pooling over the gold event-name span before the unchanged frozen
   fingerprint matcher, measuring identity.

These interventions cannot establish capability. They localize whether errors
come from line addressing, event addressing, or downstream field readout. The
complete deployed system remains 179,826,564 parameters, strictly below 200M.
No training, rescore, threshold, development read, or confirmation read is
authorized by this audit.
