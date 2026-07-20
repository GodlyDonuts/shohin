# R12 SD-CST Complete Physical Fresh-Board Preregistration

**Status:** source implementation complete before source freeze, board seed, or
training seed. No fresh board, GPU output, development access, or confirmation
access exists yet.

**Claim class:** bounded fresh renderer/name compilation into the retained
source-deleted categorical executor. A pass is not broad natural-language or
general reasoning.

## 1. Fixed hypothesis

Complete Physical-Record Front-End v1.2 solved all twelve consumed-training
mechanics gates at 192,129,179 complete parameters. The remaining uncertainty
is transfer: can the same physical-record/local-field/nonlinear-occurrence
architecture learn new lexical atoms and compose renderer factors that were
never paired during training?

The fresh board uses no consumed renderer sentence. Its declaration atoms use
`Manifest`/`Directory`, event atoms use `Instruction`/`Command` with
`slide`/`carry`, new direction and amount words, and two new query forms. Four
even-parity declaration/event/query combinations are training views. The four
odd-parity combinations are development and sealed-confirmation views. Every
individual factor value occurs in training; only the scored compositions are
withheld.

## 2. Fresh board and custody

After this source is committed, one independent board seed will generate:

- 12,000 latent training programs x four views = 48,000 compiler-only rows;
- 512 disjoint development programs x four views = 2,048 rows;
- 512 disjoint sealed-confirmation programs x four views = 2,048 rows.

All opaque names and operation sequences are split-disjoint and disjoint from
the consumed parent train/development records. The board requires zero exact
prompt, 13-gram, name, and sequence overlap across fresh splits, zero prior
prompt/name/sequence overlap, and zero prior scored 13-gram overlap. Every row
must independently simulate, contain exactly nine bounded physical records and
one HALT, and remain below the 144-byte record/query windows. Training contains
no state, answer, or trajectory oracle. Confirmation is mode `0600` and cannot
be opened without a passing development result and a separately committed
authorization.

The evaluator writes an immutable board-local `O_EXCL` development ledger
before opening development bytes. The gate config and trained checkpoint are
written before that ledger. Development may be read once; confirmation remains
at zero.

The score-bearing pilot emits a source-free evidence capsule containing hard
packets, compiler pointer indices, target byte ranges, renderer IDs, and source
poison certificates. A separately committed assessor recomputes packet,
pointer, state, answer, executor, control, hash, parameter, and custody gates
from the immutable checkpoint/config/packet/evidence/executor/ledger artifacts.
The pilot summary is not sufficient to authorize confirmation.

## 3. Exact architecture and parameter contract

The endpoint reconstructs the exact joint, retained independent physical bus,
complete-local v1 query state, and v1.2 occurrence state from hash-bound
checkpoints. The two obsolete bilinear `local_declaration_*` parameter tensors were not
serialized by v1.2 because its forward path cannot consult them. This contract
zeros them deterministically and keeps them frozen; focused tests must preserve
their forward-dead status.

The treatment and matched label control train exactly:

- the physical-record byte/position embeddings, four local record layers, two
  record-set layers, role/kind/amount/entity motors;
- the eight local-query tensors; and
- the six nonlinear occurrence-head tensors.

The old bilinear declaration path, every global source/orbit/native path,
fingerprint matcher, tape/executor, motor, reader, and Shohin trunk are frozen.

| Quantity | Exact count |
|---|---:|
| immutable Shohin trunk | 125,081,664 |
| complete compiler | 67,027,474 |
| fresh-trainable compiler parameters | 12,152,855 |
| categorical motor | 19,206 |
| categorical reader | 835 |
| **complete deployed system** | **192,129,179** |
| **strict-200M headroom** | **7,870,821** |

No parameter may be added after source freeze.

## 4. Matched arms and optimization

Treatment receives the true compiler fields. The equal-parameter control gets
one of the two deterministic three-cycle entity-role derangements per latent
family; all four renderer views share that false mapping and no control family
retains any true entity role. Both arms receive byte-identical
initialization, the same 48,000 source rows, 3,000 updates, family minibatch
order, AdamW schedule, and complete parameter count.

The treatment and control each train exactly the same 102 tensor names and
12,152,855 parameters. The false-label arm changes labels only; source bytes,
family order, renderer views, initialization, optimizer, and compute are equal.

Each arm trains two epochs, family batch eight (32 rendered rows/update), AdamW
lr `2e-4`, betas `(0.9, 0.95)`, weight decay `0.01`, warmup 100, cosine decay,
gradient clipping `1.0`, and renderer-consistency weight `1.0`. Only epoch two
is score eligible. Neither arm may read development before both endpoints and
the gate config are immutable.

## 5. Source deletion and controls

The compiler emits exactly 25 categorical program bytes plus one query byte.
Program and query source tensors are poisoned after packet sealing. A separate
typed process receives only the hard packets and the hash-bound categorical
execution core. Gold packets must execute exactly. Exact compiled packets must
condition execution exactly.

Controls include equal-parameter row-shuffled labels, uniform packets, shuffled
packets, reset, freeze, post-HALT perturbation, force-alive after HALT, query
rotation, and initial/kind/identity/amount interventions. Post-HALT perturbation
must be invariant; negative controls must collapse according to the thresholds.

## 6. Immutable development gates

All gates must pass:

1. treatment minimum fit-renderer packet at least 99%;
2. development packet at least 90% overall and 85% per renderer;
3. development state, answer, and joint each at least 90% overall;
4. development joint at least 85% per renderer;
5. each packet field at least 95% overall;
6. each line/binding/initial/event pointer at least 90% overall;
7. treatment packet at least +50 points over row-shuffled labels;
8. row-shuffled packet at most 25%;
9. gold and conditional execution exact;
10. post-HALT perturbation invariant;
11. shuffled-packet state at most 35%;
12. reset and freeze state each at most 75%;
13. source deletion/poisoning and separate execution pass;
14. frozen state is byte-identical;
15. complete deployed system is strictly below 200M; and
16. development/confirmation access is exactly `1/0`.

Passing yields `authorize_one_sealed_confirmation`. Any failure yields
`reject_complete_physical_fresh_v1`; no threshold repair, rescore, extra epoch,
or confirmation read is allowed on that board.
