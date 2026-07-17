# R12 Operator-Balanced Commit Bisimulation preregistration

**Status:** CPU structural falsifier only. No neural fit, H100 job, or Shohin
checkpoint modification is authorized before the already-running carry-only
writer experiment has a sealed score. A CPU pass establishes internal
consistency and rejects named shortcuts; it is not evidence that Shohin learned
the mechanism.

## 1. Question and empirical boundary

Shohin's post-DRS evidence is consistent with a narrow transaction failure:

- the frozen model often computes the active digit correctly;
- a late residual linearly exposes carry;
- forcing carry and digit tokens repairs the serialized state;
- the next call responds to a carry intervention in most, but not all, cases;
- autonomous repeated execution still fails.

The live carry-only motor tests the smallest immediate hypothesis: repair only
the carry writer and leave the frozen reader untouched. OBCB-1 is a conditional
successor, not a competing launch. It asks whether a source-independent one-bit
commit protocol is easier to learn when supervision is allocated by the
algebraic carry operator rather than by marginal output labels.

The allowed claim is deliberately narrow:

> Equal allocation over the three decimal carry transformations, paired
> counterfactual supervision, and a hard one-bit source-deletion boundary may
> improve learnability and closed-loop composition at matched data and compute.

OBCB-1 is not a new computational primitive, a new model class, or evidence of
general reasoning.

## 2. Exact finite machine

For operation `op`, decimal digits `a,b`, and incoming carry or borrow
`c in {0,1}`, define

\[
T_{op,a,b}(c)=(d,c').
\]

For addition,

\[
s=a+b+c,\quad d=s\bmod 10,\quad c'=\mathbf{1}[s\ge 10].
\]

For subtraction,

\[
s=a-b-c,\quad d=s\bmod 10,\quad c'=\mathbf{1}[s<0].
\]

For fixed `(op,a,b)`, the carry update is exactly one of

\[
K_0(c)=0,\qquad I(c)=c,\qquad K_1(c)=1.
\]

The partition is:

| Operation | `K0` | `I` | `K1` |
|---|---|---|---|
| addition | `a+b <= 8` | `a+b = 9` | `a+b >= 10` |
| subtraction | `a > b` | `a = b` | `a < b` |

Each operation has exactly 45 `K0`, 10 `I`, and 45 `K1` digit pairs. Under
composition, `{K0,I,K1}` is a three-element transformation monoid. `I` is the
identity; composing a constant map after any map returns that constant map.

The two carry states are minimal. For example, addition with `(a,b)=(0,0)`
emits different digits for `c=0` and `c=1`. One bit is therefore necessary and
sufficient when operation, operands, and cursor remain visible.

## 3. OBCB-1 allocation and boundary

The 200 `(op,a,b)` events are partitioned by monoid element before sampling.
Counterfactual rows for `c=0` and `c=1` are inseparable pairs. The smallest
uniform integer allocation over every underlying digit pair uses the least
common multiple of 45 and 10:

- each `K0` pair is repeated twice;
- each `I` pair is repeated nine times;
- each `K1` pair is repeated twice.

This produces, per operation, 90 paired examples and 180 rows for each monoid
element. Across addition and subtraction the frozen allocation has 540 paired
examples and 1,080 rows. No pair may be split between arms or batches used for
the matched comparison.

After one transition, only

```text
CommitBit(bit: bool)
```

may cross the boundary. The prior event, source digits, incoming bit, emitted
digit, generated history, KV state, cursor history, and step number are deleted.
The next transition receives only the next visible event and `CommitBit`.
Gradients stop at this discrete semantic boundary in any future neural
realization.

The protocol adds zero trainable parameters. A future learned realization may
use an ordinary adapter only if every learned arm receives the same adapter,
initialization, update budget, and optimizer. OBCB does not claim the adapter.

## 4. Counterfactual bisimulation law

Let `J(c)=1-c`. For each local event `x=(op,a,b)`, the two factual outputs must
obey:

\[
T_x(Jc)=\Psi_x(T_x(c)),
\]

where `Psi_x` swaps exactly the two valid outputs. More concretely:

- addition changes the digit by `+1 mod 10` under a carry flip;
- subtraction changes the digit by `-1 mod 10`;
- `K0` leaves outgoing carry zero;
- `K1` leaves outgoing carry one;
- `I` flips outgoing carry with the input.

The carry signatures are

\[
K_0J=K_0,\qquad K_1J=K_1,\qquad IJ=JI=J.
\]

If all 400 local cells are exact, every output packet satisfies the one-bit
contract, and every one of the 40,000 same-operation two-step edges is closed,
then exact iteration follows by induction for any finite sequence. The CPU
falsifier checks the induction base and every length-two composable edge; it
does not infer neural learnability from that fact.

## 5. Frozen CPU falsifier

`pipeline/obcb_falsifier.py` must deterministically:

1. enumerate all 200 local events and classify them as `K0`, `I`, or `K1`;
2. verify the 45/10/45 counts separately for addition and subtraction;
3. verify monoid closure, identity, and all 27 associativity triples;
4. enumerate all 400 `(op,a,b,c)` cells and compare against an independent
   decimal oracle;
5. verify all 200 paired carry-flip signatures;
6. construct and hash the 1,080-row operator-balanced paired allocation;
7. enumerate all 40,000 tuples
   `(operation, first_pair, second_pair, initial_carry)`;
8. verify factual and flipped two-step execution, exact packet closure, source
   poisoning invariance, and absence of machine-internal retained state;
9. inspect packet structure rather than trusting a self-reported resource
   ledger; and
10. reject every named negative control.

The CPU decimal oracle is explicitly an external verifier used by the
falsifier. It is forbidden at neural inference and cannot support a learned
reasoning claim.

## 6. Required negative controls

The finite board must reject:

1. **Commit-ignoring:** computes every transition as if incoming carry were
   zero.
2. **Stale-source replay:** is behaviorally exact while its retained source is
   intact, but recomputes the committed bit from the prior event. It must fail
   structural source deletion and change under source poisoning.
3. **Shuffled state:** deterministically swaps carry labels for a balanced
   subset of event pairs while retaining the one-bit packet shape.
4. **Result history:** carries the correct bit plus the prior result digit. It
   may be behaviorally exact but must fail the one-bit packet contract.
5. **Hidden step:** carries the correct bit plus a hidden transition counter.
   It may be behaviorally exact but must fail the one-bit packet contract.

A favorable cheating control is allowed to remain behaviorally exact. It is
still rejected if it retains forbidden information. Conversely, a
resource-matched one-bit control is rejected only by a frozen behavioral gate.

## 7. Equivalence dossier and allowed novelty

OBCB-1 has no favorable expressivity separation from established methods:

- **Ordinary counterfactual paired SFT:** same zero-loss function class. OBCB
  differs only in operator allocation and the enforced deletion boundary.
- **One-bit recurrence / two-state automaton:** exactly isomorphic to the
  minimal carry transducer and the strongest favorable architectural control.
- **RNN, GRU, or tied transformer recurrence:** can realize the same state
  update; at fixed width it can be unrolled into a finite circuit.
- **Adapter or LoRA implementation:** ordinary implementation machinery. If
  needed, the same parameter budget must be granted to every learned arm.
- **A 400-cell table:** exactly realizes the local transition and is a required
  upper-bound control, not a learned systematicity claim.
- **Visible state tokens or KV memory:** can realize the bit but must charge
  token and KV resources. They are favorable controls with a larger retained
  information vector.
- **External executor or result tape:** can solve the task but violates the
  source-deletion and no-external-execution contract.

The only reopenable conjecture is a learnability and sample-allocation claim:
balancing by transition-monoid element and enforcing one-bit causal closure may
outperform outcome-balanced paired SFT at equal examples, parameters, updates,
and inference steps.

## 8. Frozen resource vector

For the OBCB mechanism relative to its matched learner:

| Resource | Value |
|---|---:|
| added trainable parameters | 0 |
| retained dynamic state | 1 bit |
| source bytes after commit | 0 |
| retained result-history symbols | 0 |
| hidden step bits | 0 |
| external memory bytes | 0 |
| external execution calls at inference | 0 |
| additional inference steps | 0 |

Training examples, optimizer updates, FLOPs, precision, adapter parameters, and
base checkpoint must be identical in every future neural arm. The CPU
falsifier's oracle calls are reported as verification work, not inference.

## 9. Neural matched arms, conditional only

No arm below may launch until the carry-only writer result is sealed and
interpreted.

1. existing frozen reader plus successful writer only;
2. ordinary outcome-balanced counterfactual paired SFT;
3. operator-balanced paired SFT without deletion enforcement;
4. OBCB-1 with operator balance and hard one-bit deletion;
5. favorable explicit one-bit recurrent register;
6. equal-budget generic rank-8 adapter;
7. shuffled state within identical nuisance strata;
8. 400-cell table upper bound.

All arms share data identities, total examples, optimizer updates, batch order,
base, adapter capacity, and decode policy. Development can reject but cannot
confirm. A one-shot held-out board must be frozen before any score is read.

## 10. Kill criteria

Reject the finite OBCB contract if any of these occur:

- a local cell or carry-flip signature fails;
- an operation does not have exact 45/10/45 monoid counts;
- one of 40,000 composable edges fails factual, flipped, or source-poisoned
  execution;
- any packet retains more than the exact boolean bit;
- any machine object retains source, result history, or step state;
- a named negative control is admitted;
- the deterministic report changes across repeated runs.

If a later neural experiment is authorized, reject the OBCB hypothesis if:

- fresh writer accuracy is below 99%;
- identity-class carry accuracy or flip selectivity is below 99.5%;
- any operation, style, width, or carry stratum is below 99%;
- source poisoning or irrelevant sham interventions alter continuation;
- the frozen cycle remains below 45/50;
- full traces remain below 90% separately at widths 4, 6, 8, and 10;
- OBCB gains less than 10 points over operator-balanced paired SFT on cycles or
  less than 15 points on unseen-width full traces;
- shuffled labels retain more than 25% of treatment gain; or
- local gates pass while closed cycles fail. That outcome falsifies one-bit
  sufficiency for the deployed interface and forbids silently adding a tape.

If ordinary paired SFT, the one-bit recurrent control, or the matched adapter
ties OBCB, retain the simpler method and reject any OBCB-specific advantage.
