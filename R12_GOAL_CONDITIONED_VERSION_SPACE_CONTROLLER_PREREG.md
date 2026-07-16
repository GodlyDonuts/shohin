# R12 Goal-Conditioned Version-Space Controller

**Status:** architecture theorem and CPU falsifier preregistration; no empirical
result and no Shohin fit is authorized by this document.

## 1. Question

Track S asks whether a learned hard packet can preserve and update state after
the source is deleted. That is necessary infrastructure, but it is not
reasoning: the event and destination address are supplied from outside.

Track C asks a stricter question:

> Can a bounded neural controller choose an informative operation, commit that
> operation before observing its consequence, update its own persistent state,
> and decide when the state is sufficient to halt?

The bounded claim is autonomous closed-loop adaptive inference. It is not a
claim of general reasoning, language transfer, asymptotic memory, or novelty.

## 2. Separation theorem

Track S has the form

```
p_0 = E(source)
p_{t+1} = U(p_t, event_t, supplied_address_t)
answer = O(p_T, query)
```

An exact coordinate transducer can pass arbitrary-depth state reconstruction,
source deletion, packet donation, new readers, output recoding, and legal-write
checks while using the controller `HALT` at every step. Therefore no conjunction
of Track S transport tests implies autonomous operation selection or halting.

Track C requires a closed loop:

```
(operation_t, write_t, halt_t) = Policy(p_t, observation_t, goal)
receipt_t = Commit(operation_t)
consequence_t = Environment(receipt_t)
p_{t+1} = Update(p_t, consequence_t, write_t)
```

For a task family with hidden world `theta`, success requires a stopping time
`tau <= B` and a verifier-accepted answer. A valid experiment must also exhibit
a named comparator bound below the treatment result and causal mediation by the
packet and selected action.

## 3. Smallest finite falsifier

The hidden world is a threshold

```
theta in {0, ..., 15}
```

independent of every source byte. The sufficient state is an interval `(L,U)`
stored as two literal F_17 registers. Initially `(L,U)=(0,15)`. A probe `k`
returns one bit:

```
z = 1[theta <= k]
```

The sound version-space update is

```
z = 1: U <- min(U,k)
z = 0: L <- max(L,k+1)
```

and `HALT(y)` is sound exactly when `L=U=y`. There are 136 valid interval
states, so two F_17 registers are sufficient for this finite experiment.

The minimax action is the midpoint `floor((L+U)/2)`. It identifies every world
in at most four probes. A fixed four-probe schedule partitions 16 thresholds
into at most five answer cells, so its exact success ceiling is `5/16 = 31.25%`.
An answer/source motor has ceiling `1/16 = 6.25%` because the hidden world is
independent of source. These are theorem bounds, not empirical baselines.

## 4. Treatment architecture

The treatment is a sub-20k-parameter hard-packet controller with no transformer
context cache after an action is committed:

1. A two-register categorical workspace persists `(L,U)` as two `uint8`
   symbols in `0..16`.
2. A policy reader consumes only the hard packet, the most recent observation,
   and a constant identify-the-threshold goal token.
3. It emits either `PROBE(k)` for one of 16 values or `HALT(y)` for one of 16
   values.
4. The action bytes and current packet hash are committed before the environment
   releases `z`.
5. A learned updater chooses `L`, `U`, or no-write and emits the replacement
   symbol from `(packet, committed action, z)`.
6. No schedule, destination address, symbolic midpoint, information-gain score,
   state label, action label, or halt label enters the treatment.

Training uses terminal verifier success minus a fixed per-probe cost. Any
auxiliary action, state, address, midpoint, information-gain, or halt target
invalidates the treatment. Policy-gradient, exact expected-return, or another
estimator may be chosen only before the first scored seed and must then remain
fixed across all arms.

## 5. Controls

Every neural control receives the same world sampler, observation timing,
terminal reward, probe budget, parameter cap, optimizer evaluations, and seeds.

- hard-packet controller with packet clamped to its initial value;
- fixed open-loop four-probe schedule plus learned answer head;
- source-retained answer motor;
- continuous-state addressed controller;
- parameter-matched GRU controller;
- parameter-matched adaptive-computation controller;
- shuffled-observation controller;
- exact midpoint/version-space program as a non-neural upper bound.

The GRU and adaptive-computation controls remain eligible if they win. A tie
rejects an ACW-specific advantage while retaining only the bounded autonomous
control finding.

## 6. Causal predictions

From `(0,15)`, the first action must be `PROBE(7)` up to a committed action-label
permutation. Intervening on its observation gives distinct futures:

```
do(z=1) -> packet (0,7) -> next probe 3
do(z=0) -> packet (8,15) -> next probe 11
```

Required interventions:

- donor-swap `(0,7)` and `(8,15)` packets while holding source, goal, and latest
  observation fixed; the next actions must swap between 3 and 11;
- clamp the first action to a non-midpoint probe; the controller must perform a
  truthful update and recover, with at most one additional step when feasible;
- shuffle observations after action commitment; success must collapse toward
  the theorem-bound control;
- hold the packet fixed while changing source bytes; action logits must remain
  identical;
- force `HALT` when `L<U`; exact success is at most `1/(U-L+1)` and is recorded
  as an ambiguous-halt violation.

## 7. Pass and kill criteria

The treatment passes the finite mechanism gate only if all conditions hold on
every one of the 16 worlds for every scored seed:

1. 100% terminal answer accuracy within four probes under the ordinary policy;
2. zero ambiguous halts and zero action-after-halt events;
3. every committed update preserves `L <= theta <= U`;
4. every nonterminal ordinary action strictly reduces the version space;
5. the two branch interventions and packet-donor action swap are exact;
6. shuffled observations score no more than 31.25% plus one world;
7. packet-clamped and source-motor controls score no more than their frozen
   theorem ceilings plus one world;
8. the treatment exceeds every valid learned equal-budget control by at least
   two worlds, otherwise the architecture-specific claim is rejected;
9. action receipts prove the environment response was unavailable at commit;
10. code, seeds, resource ledgers, transcripts, and the decision artifact are
    commit-bound and independently replayable.

Any supplied schedule/address/operator, retained transcript/KV, hidden symbolic
executor, reward shaped by the correct action/state, post-score seed change, or
missing control rejects the run. Passing this 16-world task still proves only a
bounded closed-loop mechanism because the complete policy can be unrolled into a
finite circuit.

## 8. Scaling theorem and fixed-packet boundary

An arbitrary interval over `N` ordered worlds has `N(N+1)/2` possible states.
A fixed four-symbol F_17 packet has `17^4 = 83,521` states and becomes
information-theoretically insufficient at `N=409`, where the interval count is
83,845. Any scaling claim must therefore grow packet dimension as

```
d(N) >= ceil(log_17(N(N+1)/2))
```

or explicitly account for external memory. Fixed-packet success at `N=16`
cannot be extrapolated past that bound.

## 9. Prior-art boundary

Version spaces, binary search, recurrent state, adaptive computation, neural
Turing machines, predictive-state representations, and active identification
are established ideas. The project does not call any primitive world-first.
The open empirical contribution is the measured conjunction of hard minimal
state, commit-before-consequence action choice, learned sparse update, learned
halting, theorem-bounded shortcuts, and causal packet/action interventions.

## 10. Shohin admission

No language-model modification follows a CPU no-go. A CPU pass authorizes only
an isolated Shohin sidecar proposal in which the immutable 300k transformer is
frozen, the sidecar emits typed tool/probe actions, action bytes are committed
before tool results arrive, and the recurrent packet is the only cross-step
state exposed to the controller. Language transfer, preservation, and manual
interaction remain separate gates.
