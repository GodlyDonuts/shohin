# R12 Mixed-Difference Residual Transducer CPU Preregistration

**Status:** FROZEN MECHANICS PREREGISTRATION. No Shohin neural fit, H100 job,
production-data build, architecture promotion, capability result, or novelty
claim is authorized by this document.

**Protocol:** `R12-MDRT-CPU-v1`

**Frozen scope:** one deterministic, standard-library CPU falsifier on the
finite board below. The planted positive arm is deliberately supplied with the
complete transition law inside fixed tail resources. A pass can validate only
the mixed-difference algebra, autonomous finite-state interface, erasure
contract, collapse audits, and resource accounting. It cannot establish that
Shohin contains the required interaction or can learn it.

## 1. Evidence boundary and capability conjecture

The post-DRS workspace probe found a strong late-layer digit actuator, but it
replaced a full 576-dimensional last-position residual inside a matched source
prompt. It did not show that a compact residual is source-portable, closed
under update, internally scheduled, or consumed after use. Source-scheduled
reasoning and typed-controller experiments separately locate the missing
behavior in autonomous selection, state update, and composition.

The bounded conjecture frozen here is:

> **MDRT-C1.** On a finite source-deleted noncommutative program board, a
> transition cell whose only successor channel is the mixed finite difference
> of a state input and an internally selected action can exactly update and
> consume a persistent state when that mixed interaction contains the complete
> transition law. The same allocated cell with zero mixed interaction cannot
> update, and a state/depth shortcut that does not retain the program cannot
> solve more than one action word per depth and initial value.

This is a mechanics proposition, not a learnability conjecture. It is
deliberately falsifiable by implementation errors, source leakage, stale-state
reuse, incorrect minimization, unmatched budgets, or an unexpectedly capable
shortcut.

## 2. Axiomatic primitive

Let `S` be a finite causal-state set, `A` a finite action set, `c` a fixed
source-free carrier, `B` a state injection, `e_a` an action injection, and
`Phi` a frozen tail. Define

```text
M(z, a) = Phi(c + Bz + e_a) - Phi(c + Bz)
          - Phi(c + e_a) + Phi(c)

a_t     = policy(z_t)
z_(t+1) = decode(M(z_t, a_t))
```

The action-only term `Phi(c + e_a) - Phi(c)` may be cached. The charged runtime
still reserves two live tail calls per transition: one state-plus-action call
and one state-only call. There is no additive state bypass around `M`.

The compiler may read `(initial_value, program)` once and emit a sealed state.
After commitment, the updater receives only that sealed state. It receives no
source bytes, source pointer, source length outside the state, event stream,
cursor oracle, KV cache, external executor output, verifier, or repair signal.
At each boundary the successor is written to a fresh sealed object and the old
state is unavailable to the next call. HALT is selected by the same policy
when the retained program suffix is empty.

### Finite-state proposition

Let a deterministic task have transition `delta`, policy `alpha`, output
`omega`, terminal predicate `tau`, and an injective encoding `E:S -> Z`. If,
for every reachable `s`,

```text
policy(E(s))                    = alpha(s)
decode(M(E(s), alpha(s)))       = E(delta(s, alpha(s)))
output(E(s))                    = omega(s)
halt(E(s))                      = tau(s),
```

then induction gives exact autonomous execution for every finite task path.
Because only `E(delta(s, alpha(s)))` survives the boundary, mutations of the
source or old state after commitment cannot affect future behavior.

At finite precision this realization is a deterministic Moore transducer with
at most `2^b` physical states for `b` retained bits. The proposition does not
define a new computational class.

### Mixed-interaction cancellation lemma

If the frozen tail is additively separable on the reachable domain,

```text
Phi(c + u + e) = F(u) + G(e) + constant,
```

then `M(z,a)=0` identically. A nonzero mixed term is therefore necessary for
this primitive. It is not sufficient: its successor and late-consumer
signatures must separate the task's causal quotient.

## 3. Frozen finite board

All arithmetic is over `F_17`. The source is `(x, w)` where `x` is in
`{0,...,16}` and `w` is a nonempty word over `{A,B}` of length at most eight.

```text
A(x) = x + 1 mod 17
B(x) = 2x mod 17
```

The actions are noncommutative: at `x=0`, `B(A(x))=2` while `A(B(x))=1`.
At a nonterminal state, the correct action is the first retained symbol. A
correct transition applies it and consumes exactly that symbol. Any wrong
action or premature HALT enters one absorbing invalid state. At an empty
suffix, HALT self-loops and either arithmetic action enters the invalid state.

The exhaustive source partitions are frozen as:

| Partition | Lengths | Cases |
|---|---:|---:|
| train-named mechanics slice | 1--4 | `17 * 30 = 510` |
| development-named mechanics slice | 5--6 | `17 * 96 = 1,632` |
| evaluation-named mechanics slice | 7--8 | `17 * 384 = 6,528` |
| complete nonempty board | 1--8 | `8,670` |

The names do not authorize fitting. Every arm is evaluated exhaustively on all
`8,670` sources. The complete causal machine contains

```text
17 * sum_(l=0)^8 2^l + 1 invalid state = 8,688 states.
```

Its observable Moore output is `(value, required_action, terminal, invalid)`.
This output plus the total transition table over `{A,B,HALT}` must minimize to
exactly `8,688` equivalence classes.

## 4. Frozen arms

1. **Planted mixed-interaction positive.** `Phi` contains large deterministic
   carrier, state-only, and action-only nuisance terms plus a mixed term that
   encodes the exact successor state. Four-term subtraction must cancel every
   nuisance coordinate and leave the successor code exactly.
2. **Zero-interaction negative.** Byte-for-byte equal allocated dimensions,
   state capacity, table capacity, precision, and charged tail calls. Its mixed
   term is zero. Successor decoding must fail closed.
3. **State/depth shortcut.** Retains the current value and remaining depth but
   not program symbols. It follows one fixed alternating action schedule per
   depth, consumes one depth unit, and receives padding to the treatment's
   allocated state and fixed-tail budgets. Exactly one of `2^L` words can
   match its action trajectory at each length and initial value, so its frozen
   exact-trajectory count is `17 * 8 = 136` of `8,670`.
4. **Exact task machine.** A transparent hard-register upper bound used only
   for transition-oracle and Moore-partition audits. It is not a treatment to
   beat and carries no neural claim.

No arm receives training examples, optimizer updates, gradients, random seeds,
network access, subprocesses, or accelerator libraries.

## 5. Resource-vector equivalence dossier

Every arm reports the ordered resource vector

```text
(trainable_parameters, allocated_persistent_state_bits,
 utilized_persistent_state_bits, precision_bits,
 allocated_transient_vector_bits, allocated_fixed_tail_table_entries,
 allocated_fixed_tail_table_bits, charged_tail_calls_per_transition,
 source_bits_read_at_compile, source_bytes_retained_after_compile,
 oracle_calls_at_inference, training_examples, optimizer_updates,
 training_flops, external_memory_bits, external_execution_calls,
 sequential_depth_per_task_step)
```

Treatment, zero-interaction, and shortcut arms must have identical **allocated**
budgets. Utilized bits and semantic tail calls are reported separately and may
differ; padding is never presented as useful computation. Audit-oracle calls
used to score the finite board are reported outside the inference vector and
are identical across arms.

Equivalence boundaries:

- the inference mechanism is a constrained recurrent transformer / finite
  transducer and is exactly simulable by tied recurrence;
- a fixed-width latent scratchpad is an equivalent state carrier, while a
  growing token/KV scratchpad has a different retained-memory vector;
- Universal Transformer or ACT recurrence can simulate the cell and HALT;
- an externally supplied action, address, cursor, or stop decision invalidates
  autonomy and collapses to scheduling;
- quantized MDRT states are conjugate to hard registers after exact Moore
  minimization; semantic digit labels or supplied successor states collapse to
  SRR/ACW-style supervision;
- visible chain-of-thought can simulate a finite path by retaining emitted
  tokens and KV, but those bytes and sequential depth are not free;
- finite unrolling is always available at fixed maximum depth and defeats any
  primitive-level novelty claim.

The only possible later hypothesis is resource-bounded: a frozen Shohin tail's
pre-existing state-action interaction might expose a useful updater with less
new training or fewer new parameters than a matched generic recurrent adapter.
This CPU board does not test that hypothesis.

## 6. Exact collapse and audit conditions

The falsifier must fail closed if any condition holds:

1. The planted arm's mixed difference does not exactly equal its encoded
   successor on every one of `8,688 * 3` state-action cells.
2. The zero-interaction arm produces any nonzero mixed coordinate or any valid
   successor decode.
3. The task or positive transition machine minimizes to other than `8,688`
   Moore classes, or the positive table differs from the task table.
4. The zero or shortcut machine preserves the full task quotient.
5. The shortcut solves other than exactly `136` complete source trajectories.
6. A post-commit runtime object contains source/program fields outside its
   sealed state identifier.
7. Mutating source variables or replacing the old baton after a successor is
   committed changes continuation from the successor.
8. Donor-state interchange fails to make continuation follow the donor.
9. Allocated budgets differ across the three executable arms, any source byte
   survives compilation, or any inference oracle/external executor is used.
10. Board counts, action algebra, deterministic serialization, or audit
    recomputation drift from this document.

## 7. Frozen mechanics gates

The deterministic audit is admitted only if all gates hold simultaneously:

- positive mixed-cell exactness: `26,064 / 26,064`;
- positive complete-trajectory exactness: `8,670 / 8,670`;
- zero mixed coordinates and valid decodes: exactly zero;
- zero complete-trajectory exactness: exactly zero;
- shortcut complete-trajectory exactness: exactly `136 / 8,670`;
- task and positive Moore classes: exactly `8,688` each;
- source and stale-state erasure: bit-identical on every audited continuation;
- donor following: exact on every ordered donor case audited;
- allocated resource vectors: exactly equal across executable arms.

A test-suite pass is an implementation check against these frozen mechanics,
not a scientific result. No threshold may be changed after execution to make
an arm pass.

## 8. Prior-art and claim boundary

Every computational component is established machinery: deterministic finite
transducers and Moore minimization, tied recurrent networks, Universal
Transformers and adaptive computation, finite-difference interaction terms,
predictive-state representations, latent recurrent scratchpads, neural memory,
hard registers, and chain-of-thought unrolling. The four-term interaction is a
discrete mixed derivative, not a new algebraic primitive.

The narrow Shohin-facing idea is only this conjunction: use the already
observed late-layer digit actuator as the readout branch, isolate frozen-tail
state-action curvature by exact mixed subtraction, require that curvature to
write the entire next source-deleted baton with no additive bypass, and erase
the prior baton before the next autonomous action. No world-first,
primitive-novelty, general-reasoning, or SoTA claim is allowed from this board.

## 9. Authority boundary

Authorized after this freeze:

1. implement `pipeline/mdrt_cpu_falsifier.py` exactly to this contract;
2. implement exhaustive deterministic tests in
   `pipeline/test_mdrt_cpu_falsifier.py`;
3. run targeted unit tests, Ruff, `py_compile`, and diff checks.

Not authorized: any Shohin checkpoint load, neural fit, H100 or other GPU job,
production board, runbook edit, result document, architecture promotion, or
capability claim.
