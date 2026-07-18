# R12 SCERT CPU-First Execution Preregistration

**Protocol:** `R12-SCERT-EXECUTION-CPU-v1`
**Theory:** `R12_SELF_CANONICALIZING_EPOCH_RETIREMENT_THEORY.md`
**Theory SHA-256:** `d315b107b6ce3d486a83e091168f027356007c41dd17dd0f8f2f9d7441281dc4`
**Status:** CPU MECHANICS IMPLEMENTATION ONLY. H100 execution is unconditionally
`NO-GO` pending a byte-exact independent hostile review, external hidden-board
custody, immutable manifests, and a separate hardware authorization.

This document instantiates the CPU-first portion of Section 16. It does not
create a fitted model, hidden board, confirmation secret, score, reasoning
result, or hardware authority. The only files in this implementation package
are:

1. `R12_SCERT_EXECUTION_PREREG.md`
2. `pipeline/generate_scert_boards.py`
3. `pipeline/test_generate_scert_boards.py`
4. `train/scert.py`
5. `train/test_scert.py`
6. `train/jobs/scert_newton.sbatch`

No existing file is modified by this protocol. A later source manifest must
bind all six files above plus the theory, `train/model.py`, `train/muon.py`,
`train/digitwise_protocol.py`, the tokenizer, the parent checkpoint, and the
PyTorch AdamW source. A source change invalidates every review and receipt.

## 1. Claim boundary and hard stops

The CPU package may establish only that the declared board construction,
packing, event machine, source intervention, failure accounting, and toy
mechanics are executable and internally consistent. It cannot establish learned
state retirement or reasoning.

The following are hard stops:

- no command in this package launches, fits, evaluates, or requests a GPU;
- the Slurm wrapper requests no GPU and clears `CUDA_VISIBLE_DEVICES`;
- no local command generates, signs, decrypts, reveals, or scores `H_B`, `H_M`,
  `H_A`, or `H_O`;
- no partial hidden metric, transcript, label, residual, or aggregate may be
  opened before the reveal gate;
- no overwrite, resume, retry-selected checkpoint, or development-selected seed
  is permitted;
- malformed, missing, capped, duplicate, unreadable, or non-EOS rows remain in
  their frozen denominator as failures;
- parsed text, arithmetic, width, operation, carry, state validity, schedule,
  answer, gold token, repair, retry, reranking, or host-selected boundary may not
  enter an autonomous runtime call.

## 2. Frozen inputs and parameter ledger

| Object | Frozen identity |
|---|---|
| Parent checkpoint | `train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt`, SHA-256 `d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459` |
| Tokenizer | `artifacts/shohin-tok-32k.json`, SHA-256 `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| Model source | `train/model.py`, SHA-256 `45fc0dc46ceb0f91d08e3f671cbe9ef202ea212e72d5bba8b77356c3fb0983d4` |
| Muon source | `train/muon.py`, SHA-256 `863e79aaaaebb681382f0c88078390b5683ab39be79ac7df60f26d1c04b21762` |
| AdamW source | PyTorch `2.10.0` `torch/optim/adamw.py`, SHA-256 `54299056b7745c162192132bb6028f3387c05ff4203518ff0240058584968312` |

The tied parent has `125,081,664` unique parameters. SCERT adds only:

```text
carry motor:   576*8 + 8 + 8*2 + 2 = 4,634
boundary head: 576*2 + 2             = 1,154
added total:                              5,788
deployment total:                   125,087,452
```

The true and shuffled heads are alternative artifacts and are never resident
together. Total deployment parameters remain strictly below `150,000,000`.

## 3. One effective-logit surface

At every valid next-token position, `h` is the post-final-norm 576-vector and
`ell_base` is the tied unembedding output. The rank-8 motor has no router:

```text
delta = W_up * SiLU(W_down * h + b_down) + b_up
ell_eff = clone(ell_base)
ell_eff[v0] += a_M * cast(delta[0], ell_base.dtype)
ell_eff[v1] += a_M * cast(delta[1], ell_base.dtype)
y = lowest_token_id_argmax(ell_eff)
E = (y == EOS_ID_0)
```

The frozen tokenizer gives `v0=28`, `v1=29`, neutral ASCII-space ID `233`, and
EOS/dummy ID `0`. `a_M` is exactly zero or one. Motor-off still computes and
receipts the same delta, then multiplies it by zero. No coordinate except 28 and
29 may change. Emission, EOS event, vocabulary margins, stage-one CE, and z-loss
all use this same `ell_eff`; a pre-motor EOS check or second argmax is fatal.
The runtime token/event transitions and objective accept the `EffectiveLogits`
receipt rather than a free logits or token argument, recompute its lowest-ID
argmax, and reject any non-motor coordinate mutation.

The FP32 stage-one objective is exactly:

```text
(sum CE(ell_eff.float(), label)
 + 1e-4 * sum logsumexp(ell_eff.float())^2) / supervised_token_count
```

There is no implicit library shift, per-lane weighting, or second EOS loss.

## 4. Boards and custody

Local code deterministically constructs only:

- `T_14`, `T_15`, and `T_16`: 2,048 width-4 episodes each, balanced as
  `2 operations x 8 intermediate carry/borrow patterns x 128`;
- `D_256`: 256 width-4 episodes balanced as
  `2 operations x 8 patterns x 16`;
- public `D_12`: the frozen ordered 12 IDs with two episodes in every
  `(width 4/6/8, add/sub)` cell.

Every episode freezes canonical bytes, exact `P/X` token IDs, lane action,
canonical row SHA-256, token-row SHA-256, semantic episode key, and semantic
transition keys for all successor and terminal-answer lanes. Addition operand
order is canonicalized for semantic keys;
subtraction order is preserved. Exact, token-row, episode-semantic, and
transition-semantic intersections must all be zero among every local board.

The hidden registry is specification-only:

| Board | Frozen denominator |
|---|---:|
| `H_B` | 384 episodes, 2,688 rows: 2,304 COMMIT + 384 HALT |
| `H_M` | 384 one-dispatch cases |
| `H_A` | 768 autonomous episodes |
| `H_O` | 432 observational candidate pairs in 18 strata |

`H_B` has width row counts `640/896/1152` and 64 episodes in every
`(width,operation)` cell. The local builder emits only those specifications and
local key-set commitments. It has no signing/private-key API and no hidden-board
generator.

An external custodian must return a detached Ed25519 receipt bound to an
already provisioned read-only, single-link public-key trust anchor. The receipt
contains only the public declared board-count specification, encrypted-byte
length/hash, exact/semantic key-set commitments, a zero-intersection certificate,
and an unrevealed fit gate. It is invalid if the
custodian identity equals the builder identity, the signature is absent or
wrong, any hidden plaintext/key/label appears, any count/specification changes,
or any exact/semantic intersection is nonzero. The domain keys and ciphertext
remain unavailable to builders, trainers, operators, and experiment code until
all three stage-one checkpoints and all six true/shuffled head artifacts are
immutable.

## 5. Exact tensors

The scaffold tokenization is frozen at 70 `G_L` IDs and 3 `G_R` IDs. All dense
rows have length 2,048 and position IDs `0..2047`. Dummy slots contain token ID
zero with validity false.

Stage one is exactly:

```text
[0,70)       G_L
[70,582)     invalid retired-source dummies
[582,1094)   P then invalid dummies
[1094,1097)  G_R
[1097,1609)  X then invalid dummies
[1609,2048)  invalid dummies
```

Teacher-forced EOS is valid at `1097+len(X)` and has no outgoing loss. Labels
are only at position 1096 and the predictors under `X`, giving exactly
`len(X)+1` supervised tokens. One pack is `[5,2048]`; one update is two packs
flattened without reordering to `[10,2048]`. Each seed has one epoch and exactly
1,024 updates.

Stage two is exactly:

```text
[0,70)       G_L
[70,582)     P then invalid dummies
[582,1094)   X then invalid dummies
[1094,1097)  G_R
[1097,2048)  invalid dummies
```

`M_clean` blocks every valid `X <- P` edge; `G_R` reads `G_L`, valid `X`, and
prior `G_R`, never `P`. `h_probe` is always the post-final-norm residual at
position 1096. The clean/stale reconstruction-mask XOR must be exactly the
valid `X <- P` edge family and nothing else. `Keep` retains only valid `G_L`,
`X`, and `G_R` K/V.

## 6. Training and optimizer contract

Stage one owns exactly 210 Muon matrices (`106,168,320` scalars) and 126 AdamW
tensors including the motor (`18,917,978` scalars). Their union has
`125,086,298` unique trainable parameters. `head.weight` is the same storage as
`tok.weight` and receives no duplicate optimizer entry. The boundary head is
absent.

FP32 parameters and optimizer states are mandatory. The forward is BF16; the
objective is FP32; there is no scaler or shadow copy. Both optimizers are
zeroed, one loss is backpropagated, one global FP32 norm is computed over the
union, and all gradients are multiplied once by `1/max(1,norm)` when needed.
Every gradient must be finite and present. Muon steps first, AdamW second.

For update `u=1..1024`:

```text
s(u)=u/50,                                           u<=50
s(u)=0.1+0.9*0.5*(1+cos(pi*(u-50)/(1024-50))),      otherwise
lr_muon=0.001*s(u)
lr_adam=0.0002*s(u)
```

Muon and AdamW hyperparameters, operation order, source hashes, and all update
receipts are those frozen in the theory and serialized in the immutable
`stage1_optimizer_binding`; fused/foreach/capturable substitutions are false and
Muon precedes AdamW. Stage two freezes base/motor and fits
one true and one shuffled 1,154-parameter head from byte-identical initialization
with AdamW LR `0.01`, betas `(0.9,0.95)`, epsilon `1e-8`, zero decay, batch 512,
10 epochs, 200 updates, clip 1.0, and no warmup. The shuffled labels rotate each
five-lane episode from `CCCC H` to `H CCCC`.

## 7. Runtime state and dispatch

`Q_e` has exactly these fields and no semantic text:

```text
phase, commit_count, candidate_count, epoch_token_count,
total_token_count, replay_slot_cursor, generation_slot_cursor,
cap_constants, failure_flag, rng_state_and_cursor,
deterministic_tie_state, publication_receipt_cursor
```

Caps are eight COMMITs, nine EOS candidates, and 512 generated tokens per epoch.
Only effective argmax EOS creates an event. Every event independently performs
one clean classification replay, exactly one declared 576-to-2 head forward,
and one position-matched reconstruction. A tie selects HALT. COMMIT atomically
installs retained K/V, replaces `P` with exact authored `X`, clears `X`, updates
all counters/cursors, and continues from position 1097. HALT accepts the same EOS
and stops. Empty COMMIT, cap, malformed receipt, post-HALT token, or missing
state field fails closed.

## 8. Target-switch diagnostic

The only fixed target-switch arms are:

- `TS-C1M1`
- `TS-C0M1`
- `TS-C1M0`
- `TS-post-hoc-M1`

Each row freezes same-length `X_nom`, `X_carry`, and `X_result`; each treatment
differs from nominal at exactly one predeclared index/ID. `E_1=True`,
`D_1=COMMIT`, complete `Q_1`, weights, caps, RNG, tie rule, and endpoint are
cloned. Each arm runs one clean classifier and one true-head forward, then
discards the observed action and applies the cloned COMMIT. An observed action
change increments `boundary_action_changed` and makes every paired endpoint fail.
The arm emits only until the next effective EOS or cap and does not classify a
second boundary. A paired carry switch requires both event/action receipts,
exact nominal and carry targets, different frozen targets, and output movement
in that direction.

## 9. Failure-inclusive accounting

Every ledger is initialized with its immutable expected denominator. Case IDs
may be recorded once only; extra or duplicate cases are fatal. Summary counts
missing records as failures:

```text
failures = expected_denominator - successes
missing_as_failures = expected_denominator - observed
```

Structural `do(P)` and observational common-support ledgers are separate.
`H_O` admission remains exactly `>=216/432` overall and `>=12/24` in every
stratum; structural pairs can never fill an observational denominator.

## 10. CPU mechanics and independent reference

The finite board is exactly `2 wrappers x 8 stale-state edit pairs x 16 latest
spans = 256` cases. The main runtime and an independently written implementation
in the board module execute an explicitly initialized two-layer, width-16,
two-head, 32-token float64 causal Transformer. Fixed weights plant a nonzero
`P -> H(X) -> future` path. Their clean probe, retained K/V, next logits, and
argmax must be bit-identical.

All 13 Section 11 gates are noncompensatory: clone/mask XOR; independent clean
K/V; common classifier; universal `do(P)` plus structured stale controls;
dropped-source poisoning; fixed positions; exhaustive FSM; token-only API; one
head call; autonomous arm-local divergence; fixed policies; motor-off single
surface; and finite-state collapse. The report contains no learned-model score.

## 11. Runtime and publication custody

The CPU runtime manifest binds Python executable bytes, platform, locale, exact
Torch/tokenizers module bytes and versions, deterministic-algorithm state,
thread counts, TF32/cuDNN settings, the explicit float64 reference path, and only
an allowlist of non-secret environment switches. Authentication values, `.env`
content, and arbitrary environment variables are never read into a report.

Every artifact uses a current-UID-owned directory and a same-filesystem random
temporary file. It is file-fsynced, moved with OS-level no-replace rename,
directory-fsynced, reopened without symlink following, schema/hash checked,
required to be a single-link regular file, chmod 0444, and directory-fsynced
again. Existing finals, partial aliases, symlinks, hard-link aliases, and
substitution fail. There is no overwrite or resume mode.

## 12. Gates before any future H100 request

All of the following remain required and currently prospective:

1. exact six-file source hashes and one clean reviewed commit;
2. independently approved source, optimizer, runtime, parent, tokenizer, and
   public-board manifests;
3. signed external custody receipt and encrypted hidden-board commitments;
4. independent replay of every local board and all 13 CPU mechanics gates;
5. hostile review of theorem alignment, leakage, self-attestation, substitutions,
   denominator integrity, EOS order, publication races, aliases, and controls;
6. a separately named bounded H100 preflight with exact node/GPU/runtime binding;
7. only after the preflight, a separate explicit hardware authorization.

Until then the only valid decision is `CPU MECHANICS CANDIDATE; H100 NO-GO`.
