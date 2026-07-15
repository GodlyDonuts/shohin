# R12 Closed Deliberation No-Go

**Status:** exact theorem; reject internal self-querying as an information or
sample-complexity advantage by itself.

## Claim under test

A learner observes a training object `Z`, then spends `T` internal rounds
choosing questions, answering them from its own state, and updating that state
before emitting a hypothesis. The hoped-for claim was that this deliberation
could discover target information unavailable to a one-shot learner with the
same observations.

## Closed-deliberation equivalence theorem

Let `Theta` denote the unknown target system and `R` the learner's private
randomness. Consider any finite computation

```
S_0     = encode(Z, R)
Q_t     = choose_t(S_t)
Y_t     = answer_t(Z, R, S_t, Q_t)
S_{t+1} = update_t(S_t, Q_t, Y_t)
H       = decode(S_T).
```

Assume that no `Y_t` is answered by `Theta` or by any channel containing target
information beyond `Z`. Then the complete transcript and final hypothesis are
deterministic functions of `(Z,R)`, so

```
Theta -> (Z,R) -> (S_0,Q_0,Y_0,...,S_T,H)
```

is a Markov chain and

```
I(Theta; Q_0,Y_0,...,S_T,H | Z,R) = 0.
```

Composing the finite updates gives a one-shot algorithm

```
B(Z,R) = decode(update_{T-1}(...update_0(encode(Z,R))...))
```

with exactly the same output distribution. It therefore has the same risk and
information-theoretic sample complexity. A `T`-round circuit of size `s` can be
unrolled with size `O(Ts)`; a computational advantage requires an explicit
resource restriction on the comparator, not the word "deliberation."

## Oracle dichotomy

If round `t` instead receives `Y_t = oracle_Theta(Q_t)`, the incremental target
information is

```
I(Theta; Y_t | Z,R,Q_0,Y_0,...,Q_t).
```

If every increment is zero, the theorem above applies. If an increment is
positive, the environment supplied new target information and the experiment
is active learning, membership/equivalence querying, or another named oracle
model. Restricting a comparator to precommitted questions creates a round or
adaptivity separation; it does not establish internally generated reasoning.

The smallest strict active-query control is threshold search on four ordered
targets: two adaptive binary questions identify the target, while an exact
nonadaptive strategy needs three. This is a useful sanity check, not an R12
survivor.

## Collapse attacks

- **Self-review and private debate:** every generated critique is a function of
  the same observations and can be composed into `B`.
- **Proof-carrying state:** an internal prover and verifier compose into one
  algorithm. A target-specific prover or verifier is an external channel.
- **Algebraic closure:** closure under a shared hypothesis class is deterministic
  post-processing. Without that class, an off-support patch survives.
- **Self-experiment:** a target-independent simulator adds no information. A
  target-answering experiment is an oracle query.
- **External fixed execution:** any fixed solver using only `Z` can be moved
  outside the learner without changing behavior.
- **Conjugacy:** closed experiments cannot select a canonical hidden coordinate
  system when all observations are invariant under relabeling.
- **Delayed sabotage:** two targets that agree on `Z` and differ on one unseen
  continuation remain indistinguishable after arbitrary closed deliberation.

## Exact boundary

This theorem does not say sequential computation is useless. It can reduce
time, space, activation, communication, or circuit description relative to a
specified comparator. It says that sequentiality alone cannot improve the
information available about `Theta` or establish a sample-complexity advantage
against an equally expressive learner receiving the same prior and data.

The only admissible continuation is a computational generalization theorem:
identical samples, identical structural prior, no target oracle, a uniform
sequential upper bound, and a lower bound against a precisely named comparator
class. That would be a circuit, streaming, data-structure, communication, or
proof-complexity result. No CPU falsifier, Shohin fit, or H100 job is authorized
for closed self-querying alone.

## Prior-art boundary

Target-coupled exact queries are explicit in Angluin's automata-learning model;
observable sequential systems are represented by predictive-state and
observable-operator formalisms; communication-round advantages include
classical pointer-chasing separations. R12's contribution here is the explicit
information audit that prevents those channels from being relabeled as
internally created evidence.

Primary references:

- D. Angluin, *Learning Regular Sets from Queries and Counterexamples* (1987).
- N. Nisan and A. Wigderson, *Rounds in Communication Complexity Revisited*
  (1991).
