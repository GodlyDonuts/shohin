# R12 Finite State-versus-Motor No-Go

**Status:** exact identifiability boundary. No finite challenge board, by
itself, can distinguish a reusable state from every finite motor table.

## 1. Behavioral object

Let a deterministic task be

```text
M = (S, G, Q, A, delta, output),
```

where `G` contains update generators and `Q` contains late consumers. Define
the residual behavior of state `s` by

```text
rho_s(w,q) = output(delta_w(s), q),  w in G*.
```

Two states are behaviorally equivalent exactly when

```text
s ~ t iff rho_s(w,q) = rho_t(w,q) for every w,q.
```

A source-deleted realization has writer `W:S->Z`, update maps `T_g`, and reader
`D`. It is reusable over the declared system exactly when

```text
D(T_w(W(s)), q, pi) = pi(rho_s(w,q))
```

for every reachable state, generated update word, consumer, and supplied
output recoding `pi`.

## 2. Finite-protocol table theorem

For any finite evaluation protocol `P`, including adaptively chosen but
finite-support challenges, construct a machine with states

```text
Z_P = {(source_id, tested_update_prefix)}.
```

The writer stores `source_id`; each tested updater appends its label; the
reader returns a table entry for `(source_id,prefix,consumer)` and then applies
the supplied output recoding. Every untested transition enters a failure sink.

This machine can pass all of the following on `P`:

- physical source deletion;
- multi-step continuation;
- consumers and updates hidden from the scorer until after commitment;
- arbitrary supplied output recodings;
- packet swaps and complement ablations.

It has no behavior beyond the finite protocol tree. Secret scoring prevents
manual board tuning but does not turn a finite board into a universal theorem.
Any valid claim must therefore bound description length, retained state,
hypothesis class, or scale dependence.

## 3. Four exact counterexamples

1. **Consumer insufficiency:** `(a,b)` and fitted parity consumers admit the
   one-bit packet `a XOR b`, which cannot answer a held-out query for `a`.
2. **Unseen operator:** two worlds may agree on all observations while a new
   symbol denotes identity in one and bit-flip in the other. Its semantics
   require a declared grammar, examples, or an oracle.
3. **Finite horizon:** a machine may match every continuation through depth
   `L` and deliberately fail at `L+1`.
4. **Output recoding:** withholding `pi` makes two recodings jointly
   impossible; supplying `pi` lets a motor table recode too. Recoding rejects
   token-specific actuators, not arbitrary answer bundles.

## 4. Weakest conditional sufficiency

The weakest non-circular exact criterion is generator-complete,
separator-complete bisimulation:

1. declared generators span every admitted update;
2. a consumer core separates all residual states;
3. source packets are complete before late challenge disclosure;
4. every generator update commutes with the packet realization;
5. every separating consumer reads the correct answer;
6. source deletion and output recoding are process-enforced.

Induction proves every generated continuation. The resulting minimal reachable
realization is isomorphic to the Myhill-Nerode residual quotient. A complete
answer bundle closed under every generator is behaviorally a state; its
internal ontology is not separately identifiable.

## 5. What finite experiments may establish

A score-blind experiment can establish a resource-bounded result:

> One uniform mechanism generalizes across post-commit generated interfaces
> and unseen scale using fewer retained bits, parameters, examples, or compute
> than a named motor-table/control family.

It must freeze the complete resource vector, test increasing scales, include a
favorable table/horizon control, and state exactly which hypothesis class was
rejected. A finite pass never excludes unlimited tables or proves universal
reasoning.

## 6. Prior-art boundary

- residual equivalence and minimal deterministic realization are the
  Myhill-Nerode/Moore-machine construction;
- local transition closure is bisimulation;
- predictive-state representations intentionally treat a sufficient vector of
  future-test predictions as state;
- minimal observable/controllable state is unique only up to coordinates in
  classical realization theory;
- IIT and DAS test or install a supplied causal abstraction; they do not prove
  that the abstraction is the unique reusable state.

Primary references:

- Myhill-Nerode: https://doi.org/10.1090/S0002-9939-1958-0135681-9
- Predictive state representations:
  https://papers.neurips.cc/paper/1983-predictive-representations-of-state.pdf
- Interchange Intervention Training:
  https://proceedings.mlr.press/v162/geiger22a.html
- Distributed Alignment Search:
  https://proceedings.mlr.press/v236/geiger24a.html

## 7. Shohin decision

No new Shohin fit may advance from a finite consumer suite, MCBS projection,
J-lens basis, or output recoding alone. The active bounded target is a uniform
post-commit interface protocol with explicit packet and model-description
limits. `R12_POST_COMMIT_INTERFACE_FALSIFIER_PREREG.md` tests only whether its
CPU scorer correctly separates the declared complete-state and motor controls.
