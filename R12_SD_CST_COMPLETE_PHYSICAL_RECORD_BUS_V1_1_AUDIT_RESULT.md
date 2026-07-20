# R12 SD-CST Complete Physical-Record Front-End v1.1 Slot Audit Result

**Decision:** occurrence-role confusion; admit a nonlinear local six-class
pointer head as the next training-only falsifier

**Claim boundary:** deterministic read-only audit of already-consumed training
heldout rows. Development and confirmation access remain `0/0`. This is neither
a rescore nor reasoning evidence.

## 1. Receipt

- exact audit source commit:
  `a9a8d9a06a4a16c82385ae31ce346edda0d25d2f`;
- endpoint SHA-256:
  `46697b3942fdfd2edfec06cea6cb119ad507adcdee4a99330fd06bc79e5b3e88`;
- sole job: `694209`, H100 `evc23`, completed cleanly in 25s;
- report SHA-256:
  `b09122a95808e066583bc38445ac4dae3ce1df863a0fa7050b14bfc7cbe63f63`.

## 2. Six-slot result

Minimum top-one target-span exactness across the four heldout renderers is:

| Query slot | Minimum exact |
|---|---:|
| binding role 0 | 80.80% |
| binding role 1 | 99.40% |
| binding role 2 | 73.50% |
| initial occurrence 0 | 47.05% |
| initial occurrence 1 | **0%** |
| initial occurrence 2 | 16.70% |

Every one of 48,000 top-one decisions lands inside one of the six true entity
occurrence spans. The `other` category is exactly zero for every slot and
renderer. The failure is therefore not inability to localize entity evidence.

The middle initial query exposes the dominant confusion. On declaration
renderer d0 it never chooses its target; its 2,000 predictions split mainly
among binding role 2 (1,082--1,106), initial occurrence 0 (591--620), and
initial occurrence 2 (241). On declaration renderer d1 it reaches only
10.10--10.35%, with most errors again split across those three alternatives.
Mean target probability for that slot is only 15.28--15.41%.

The other initial slots are also occurrence-sensitive: occurrence 0 ranges
47.05--64.60%, and occurrence 2 ranges 16.70--52.25% depending on declaration
renderer. Binding role 1 is nearly exact, while roles 0 and 2 vary with the
same declaration factor. The shared bilinear key/query geometry finds entity
spans but aliases their surface roles and repeated positions.

## 3. Consequence

The audit rejects both an evidence-missing story and a deterministic grammar
shortcut. It also does not support adding epochs to v1.1. The next distinct
training-only system may load the exact successful v1 query path and physical
event bus, then replace declaration dot-product queries with a model-logit-only
nonlinear token classifier that emits six local occurrence logits per byte.

That head must remain local, train without state/answer/executor feedback, keep
the same consumed partition/schedule/gates, preserve all successful paths, and
keep the complete system strictly below 200M. A pass still authorizes only a
separately committed fresh board.

## 4. Artifact

- local report:
  `train/sd_cst_complete_physical_record_bus_v1_1_audit_a9a8d9a/report.json`;
- committed report:
  `artifacts/r12/sd_cst_complete_physical_record_bus_v1_1_slot_audit_a9a8d9a.report.json`;
- Newton report:
  `/lustre/fs1/home/sa305415/shohin_sd_cst_complete_physical_record_bus_v1_1_audit_a9a8d9a/report.json`.

Local and Newton hashes match exactly.
