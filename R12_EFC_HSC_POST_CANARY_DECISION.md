# R12 EFC HSC Post-Canary Decision Protocol

## Status

Pre-fit decision design only. The exact v6 two-update measurement canary is the
only authorized GPU run. This document does not authorize a qualification fit,
development or confirmation access, a native-reasoning claim, or continuation
pretraining.

## Why a training loss is not evidence

The maximum Hankel-shift compiler has 64,407,956 trainable source-compiler
parameters. Its 384-source train package contains 242,688 independently
measured target bits and 1,966,080 deterministic signature-expansion bits.
That is approximately 265 trainable parameters per independent target bit.
Complete base and derivative rollout supervision therefore permits a
source-specific interpolation strategy. Neither low train loss nor exact train
codebooks, alone, establish a reusable shift law.

The first admissible qualification must distinguish:

1. optimizer or numerical failure;
2. deep-path optimization failure;
3. invalid Hankel decoding;
4. raw-source compilation failure;
5. source-specific memorization;
6. renderer transfer;
7. unseen-world transfer; and
8. genuine causal use of prefix-shift incidence.

## Train-only partition

Use only the existing 384 frozen train sources from 96 worlds. Development and
confirmation artifacts must not be mounted.

Before reading outcomes, select worlds independently within each of the six
action families by a domain-separated hash of the frozen world identity:

- **Fitting (`F`)**: 12 worlds per family and three renderers per world,
  216 sources total.
- **Renderer holdout (`R`)**: the fourth renderer of the same 72 worlds,
  72 sources total.
- **World holdout (`W`)**: four untouched worlds per family and all four
  renderers, 96 sources total.

The split receipt must bind the domain separator, ordered world identities,
ordered source hashes, family counts, and renderer factors. No outcome may
influence the split.

## Diagnostic matrix

| Arm | Input | Decoder | Purpose |
|---|---|---|---|
| `O` | free train-only provisional logits | direct | optimizer/loss runtime |
| `GD` | supervisor-normalized train evidence | direct | lawful direct mechanism |
| `GH` | supervisor-normalized train evidence | Hankel shift | shift decoder viability |
| `RD` | raw candidate source | direct | source compilation without shift claim |
| `RH` | raw candidate source | Hankel shift | complete treatment |
| `RP` | raw candidate source | position-scrambled shift | ordered-incidence control |
| `RB` | raw candidate source | stable word bag | commutative control |

Every scored arm must share the same eligible source bytes, independent target
bits, initialization lineage, update schedule, optimizer semantics, precision,
and persistent 1,536-byte machine. `GD`, `GH`, and `O` are train-only
diagnostics and cannot contribute capability scores.

## Sequential gates

### Gate 0: measurement

The authenticated v6 canary must pass the independent closeout in
`train/assess_episode_functor_hankel_canary.py`:

- exact package, source, arm, initialization, and canary hashes;
- exact two-update schedule;
- one H100 under Landlock and a loopback-only network namespace;
- finite positive gradient norm on both updates;
- changed trainable state;
- positive timing, memory, and optimizer-state measurements; and
- zero persisted weights, scored visibility, fit authority, or pretraining
  authority.

Failure leaves every fit and architecture escalation closed.

### Gate 1: optimizer replay

Run `O` over 16 train sources and three seeds for at most 256 updates. Require:

- every component cross-entropy below `0.01`;
- total loss reduced by at least `100x`;
- 16/16 exact machines;
- 16/16 exact base and derivative codebooks; and
- zero nonfinite values or hard ties.

Failure diagnoses optimizer/loss/runtime mechanics. It is not evidence for
undercapacity or against HSC.

### Gate 2: deep-path microfit

For one fitting world, fit all four renderers with `GD`, `GH`, `RD`, and `RH`
for at most 512 updates. Each arm must reach 4/4 exact keys, record labels,
machines, base codebooks, and derivative codebooks.

If `O` passes but all deep arms fail, diagnose deep-path conditioning. Do not
increase parameter count.

### Gate 3: oracle-evidence mechanism

`GD` and `GH` must each reach 100% exact machines and codebooks on their
train-only evidence. `GD` passing while `GH` fails rejects learned Hankel
decoding independently of raw-source parsing.

### Gate 4: raw fitting

Under one frozen 4,096-update budget, every scored seed must reach on `F`:

- at least 212/216 exact machines;
- at least 99.5% transition and observer cells;
- at least 90% complete base and derivative codebooks; and
- zero unhardenable rows.

Failure after Gates 1--3 pass diagnoses raw-source compilation.

### Gate 5: train-only transfer

Without further optimization:

- `R`: 72/72 exact machines;
- `W`: at least 92/96 exact machines; and
- `W`: at least 14/16 exact machines within every action family.

Report exact key, record-type, occurrence-role, answer, transition, observer,
base-codebook, and derivative-codebook metrics separately.

### Gate 6: causal mechanism

On paired `W` rows, `RH` must beat the strongest of `RD`, `RP`, and `RB` by at
least ten percentage points in exact-machine accuracy. The paired 99%
confidence lower bound must exceed five points, and every seed must pass.

Failure rejects HSC as causally necessary under this board even if raw
compilation succeeds.

## Architecture-escalation triggers

No traditionally fixed transformer component may change merely because
headroom exists.

- **Attention:** admit an attention treatment only if local record parsing is
  strong while distant-record binding is at least ten points worse, and a
  train-only oracle cross-record summary rescues at least ten points.
- **Position/RoPE:** admit a position treatment only if semantics-preserving
  offsets or chunk-boundary shifts cost at least five paired points while
  oracle-normalized evidence remains exact.
- **Normalization/precision:** admit a norm or precision treatment only if
  FP32 rescues at least ten points, or early-to-late gradient RMS remains below
  `1e-3` for at least 80% of updates across seeds.
- **Recurrence:** admit contradiction feedback only if parser and individual
  machine cells exceed 99.5% while complete machines remain below 95%, and a
  parameter-free contradiction revision improves at least ten points
  monotonically over one through three cycles.

Every admitted parent or compiler intervention is a new named
`adapted_base` treatment. It must preserve the protected checkpoint tensors,
account every added parameter and FLOP, stay below 200M complete parameters,
and include a parameter/compute-matched control that destroys only the proposed
causal signal.

## Reserved machine-revision candidate

If parser fields and individual machine cells are strong but global consistency
fails, the first bounded candidate is the **Adjoint Causal-Syndrome Observer
(ACSO)**.

For provisional transition and observer logits, ACSO recomputes the depth-three
behavioral closure and separates base and derivative Jensen-Shannon
innovations. An explicit reverse dynamic program maps those innovations back
to the machine cells that caused them. A shared recurrent preconditioner sees
only recoding-equivariant local logit, probability, adjoint, magnitude,
moment, entropy, and cell-type features. It emits a positive bounded step
along the true negative-adjoint direction for four tied revision cycles. Only
the corrected ordinary 1,536-byte machine survives sealing.

One exact candidate budget is 3,995,137 parameters:

- `10 -> 384 -> 384` feature encoder: 152,064;
- `GRUCell(384, 768)`: 2,658,816; and
- `LayerNorm(768) -> 1536 -> 1` step head: 1,184,257.

The complete HSC+ACSO system would contain 199,488,246 parameters and leave
511,754 headroom. Its matched control cyclically scrambles multi-step
action-position residuals before the same adjoint, preserving depth, tensor
geometry, one-step information, residual norms, parameters, and four-cycle
compute while destroying only true multi-step blame routing.

The prerequisite exact CPU audit is
`artifacts/r12/episode_functor_causal_syndrome_20260724.json`. Across all 200
frozen worlds, all 26,400 permutation-preserving transition swaps and
balance-preserving observer swaps produce nonzero and collision-free
depth-three fingerprints: 132/132 unique faults per world, with 120--704
changed behavioral coordinates. Report payload SHA-256 is
`be260fda48585ff8aacc13369e8b01d80023729c944454388fe65cc37038b254`;
file SHA-256 is
`4aaf86536899214c2d2bcce3516a65c6021a267989040a2f0b4982dc24c35ef2`.
This proves exact single-swap identifiability only. It does not show correction
under noisy learned signatures or neural learnability.

The ACSO mechanics and exact controller constructor are now implemented in
`train/episode_functor_causal_syndrome_observer.py`; integration and fitting
remain unauthorized. The explicit adjoint is a hand-derived reverse dynamic
program through the finite behavioral closure and does not invoke runtime
autograd. The control is the exact gradient of its own scrambled objective and
decreases that objective under a small negative-adjoint step; it is not a
rerouted nonconservative vector field. The explicit adjoint runs under
`no_grad` and cannot retain a target or compiler graph. A hand-calculated
noncommutative machine anchors word-order semantics independently, the total
is derived from the live HSC capacity receipt, and `seal_primary_machine`
emits only the existing source-free `HardFunctorMachine` fields.

Two hostile-review rounds rejected earlier drafts for a nonconservative
control, retained autograd graphs, an absent hard seal, coordinate-dependent
hardening on tied rows, and incomplete independent derivative-prefix coverage.
The current implementation closes those findings: sealing fails closed on
every tied transition or observer row, tie-free hardening is exactly recoding
equivariant, and the independent oracle covers both causal and cyclic
derivative prefixes. Twenty-two focused tests and the complete 468-test relevant
suite pass with 63 known nested-tensor warnings. Final hostile review reports
no remaining P0/P1/P2, passes 40 additional randomized recoding trials
exactly, and bounds manual-adjoint disagreement against autograd below `4e-9`
across depths zero through five in both modes. It authorizes a mechanics-only
commit, not integration or fitting.

The cyclic objective is retained as a structural falsifier but is not the
oracle-recovery matched control because a correct causal machine is generally
not its fixed point. The source-frozen recovery protocol is
`R12_EFC_ACSO_ORACLE_RECOVERY_PROTOCOL.md`. Its oracle-fixed one-step control
computes the full closure but masks evidence beyond one action before the
reverse dynamic program. V2 audits all 672 frozen-board faults that are
immediately observation-equivalent yet separable within suffix depth three,
across three margins and deterministic recodings. V1 is void. The consumed v2
audit tests only target-informed deep-fault recovery mechanics, not HSC
integration or fitting.

**Consumed update:** v2 returned `deep_fault_oracle_no_go` under exact source
freeze `27d5c4b`. Causal treatment and one-step control each recovered 0/672
faults at all three margins despite monotonic innovation and exact recoding.
Post-hoc analysis found adverse correct-vs-wrong gradient contrast on all 672
faults; an oracle positive per-cell gate still failed all margin-0.20 faults.
The present four-cycle positive-gradient ACSO preconditioner is killed. Any
signed causal-retraction or second-order replacement requires a new named
protocol and cannot inherit ACSO's mechanics authorization.

These results admit ACSO mechanics only. They do not establish that HSC emits
usable noisy signatures, that the learned preconditioner improves a machine,
or that a revised machine transfers. Kill ACSO if oracle machines cease to be
fixed points after integration, innovation fails to fall monotonically, or it
fails to beat both unmodified HSC and the cyclic-control adjoint by the frozen
paired margin. Runtime autograd remains inadmissible in the deployed candidate
unless separately accounted as a fixed inference resource without
uncontrolled higher-order training.

## Reserved source-attention candidate

If and only if the attention trigger fires, the bounded candidate is
**Hankel-Syndrome Reentrant Attention (HSRA)**:

- encode per-cell evidence/HSC disagreement, Hankel distance, distance margin,
  and row entropies with a shared `8 -> 192 -> 512` map;
- pull feedback to source records only through the model's own soft role and
  key assignments;
- re-run the eight tied source-encoder layers once;
- add low-rank contradiction-derived biases to attention logits; and
- harden and seal only the second-pass machine.

One exact candidate uses 4,303,048 new parameters, for 199,796,157 complete
parameters and 203,843 headroom. Its isoparametric control deranges feedback
across source records while preserving the exact vector multiset, magnitude,
two-pass compute, initialization, and target budget.

HSRA remains unimplemented and unauthorized. Kill it if second-pass
exact-machine accuracy improves by less than five points, treatment fails to
beat record derangement by five points, feedback ablation costs less than five
points, or initially incorrect rows do not reduce median syndrome by at least
25%.

## Decision boundary

A complete train-only pass would establish that HSC learns transferable
source-to-machine compilation within the frozen train world, renderer, and
family support. It would not establish natural-language reasoning, broad task
transfer, native language-model consumption of the sealed machine, or general
reasoning. Those require separately frozen development, confirmation, and
cross-task protocols.
