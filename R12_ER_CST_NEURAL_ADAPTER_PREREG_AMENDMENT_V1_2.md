# R12 ER-CST Neural Adapter Preregistration Amendment v1.2

**Protocol:** `R12-ER-CST-v1.2-neural-adapter`

**Status:** locally admitted before source freeze, board seed, training seed, H100
job, output, or scored access.

## Closed v1.1 defect

V1.1 restored the categorical query interface but retained eight event slots.
Because the tied executor tests HALT before applying a card, eight slots can encode
at most seven active updates followed by HALT. The frozen task requires balanced
depths one through eight. V1.1 is therefore closed before any board seed or data
bytes existed.

## Sole v1.2 change

V1.2 increases event slots from eight to nine and physical records from twelve to
thirteen. This permits eight active card applications followed by one explicit
persistent HALT. No recurrent rule, supervision type, optimizer, control, evaluator,
or threshold changes.

The thirteen-record role head and embeddings add exactly 769 parameters:

| Quantity | V1.2 |
|---|---:|
| immutable Shohin trunk | 125,081,664 |
| compiler | 67,336,999 |
| tied categorical motor | 2,438 |
| categorical reader | 835 |
| **complete deployed system** | **192,421,936** |
| **trainable parameters** | **11,716,385** |
| **headroom below 200M** | **7,578,064** |

The compiler whitelist remains 98 tensors and the motor remains four tensors,
but the changed role tensor shapes produce trainability contract SHA-256
`1e637f3ddc09c1f89d5af9d8d258eb212218517991249ca6ff6e62ced9931eec`.
The exact confirmed-parent state and excluded-state digests remain
`cfb3d8bdf712bd0ed51e35c015b8a106b4b48b6112418585fc1df1139c3b49d9`
and `1ad33273f7db5073b4cfac0e79031544a74b74914050026642110aee484c94e7`.

The board builder must enforce exactly one HALT, depths one through eight, thirteen
nonempty physical program lines, and exact executor agreement before any scored
split is sealed. A board seed may be drawn only after the v1.2 architecture and
fresh-board source are committed and pushed.
