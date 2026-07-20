# R12 ER-CST Fresh Board Preregistration Amendment v1.1

**Status:** sole pre-write name-allocation repair. No admitted board, training seed,
H100 job, output, development access, or confirmation access exists.

After source commit `c06eab3e2476d9805bf1079c698143410e25eef5`, raw board
seed `2459068742837489615` was drawn. The full 52,096-row in-memory audit passed
every semantic, leakage, distribution, byte, and control gate except global opaque-
name uniqueness. Independent 32-bit hash truncation produced birthday collisions at
the full 195,360-name scale. The builder exited before creating its output directory,
so no train, development, confirmation, report, or access bytes exist. That seed is
closed permanently.

V1.1 changes only name allocation. It maps each unique `(split, family, role, slot)`
integer through a seed-keyed 32-bit XOR bijection and retains the same one-prefix plus
eight-hex-character surface width. This makes uniqueness exact without changing row
counts, renderer compositions, semantics, distributions, model architecture,
supervision, controls, gates, or byte limits.

A full-scale unit test enumerates all 195,360 planned names and requires exact
uniqueness. The repaired source must be committed and pushed before a new board seed
is drawn. The failed seed may not be reused.
