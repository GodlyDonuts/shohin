# R12 Active Verifier Query No-Go

**Status:** valid active-learning theorem, rejected as a new closed reasoning
mechanism. Target-coupled verifier feedback is an oracle; target-independent
verification adds no information.

## 1. Access model

Let a candidate be `theta=(M,s)`, a compact residual machine and current state.
An experiment supplies an event word, query, and proposed witness. The verifier
returns one bit

```
V_theta(x) in {0,1}.
```

Two candidates are verifier-equivalent when every possible experiment receives
the same bit. No adaptive policy can distinguish candidates in one verifier
fiber. Hidden-state conjugacies remain in the same fiber; only the behavioral
quotient can be identified.

## 2. Strongest positive theorem

Let the current quotient version space have size `N`. If every non-singleton
version space has a polynomial-time experiment that leaves at least a `beta`
fraction on either side, greedy disagreement identifies the class within

```
ceil(log(N)/-log(1-beta)) = O(log(N)/beta)
```

verifier calls. Binary information requires at least `ceil(log_2 N)` calls in
the best geometry. With independent verifier error `eta<1/2`, majority
repetition adds the standard

```
O((1-2 eta)^-2 log(T/delta))
```

factor. This can exponentially beat passive sampling, as threshold search does.

## 3. Fatal limits

1. A public target-independent verifier has zero mutual information with the
   unknown target.
2. A target-coupled accept/reject verifier is a membership oracle; a returned
   counterexample is an equivalence oracle.
3. Final-answer verification identifies accepted behavior, not internal
   transitions or a unique reasoning path.
4. Extrapolation is justified only when every surviving candidate agrees.
5. Current-state recovery needs separating experiments, resets, cloning, or an
   adaptive homing sequence; experiments can otherwise merge states
   irreversibly.
6. Query count does not imply computational efficiency. Finding a disagreement
   input for succinct circuits can be SAT-hard, balancing a version space can
   require model counting, and unrestricted program equivalence is undecidable.
7. Active experiments learn a model but do not reduce the exact online residual
   memory lower bound.

## 4. Compact exponential obstruction

On `X={0,1}^d`, take

```
H = {h_bottom} union {h_z : z in X},
h_bottom(x)=0,
h_z(x)=1[x=z].
```

Every hypothesis has an `O(d)` description and a small DFA. Against
`h_bottom`, each membership query eliminates at most one `h_z`, so exact
identification needs `2^d` calls. Compactness, determinism, and self-generated
disagreement do not imply polynomial identification.

## 5. Prior-art boundary

Membership plus equivalence queries are exact automata learning. Balanced
version-space splitting is generalized binary search. Candidate synthesis with
verifier counterexamples is CEGIS/OGIS. Self-play chooses experiments and
hypotheses; all target information still comes from the verifier. Verifiable
reward changes search and optimization, not the oracle information boundary.

## 6. Decision

Actively generated disagreement tests remain useful data-engineering doctrine.
They are not a new latent-reasoning mechanism. No CPU falsifier is authorized
without a concrete nonlinear hypothesis class with polynomial consistency and
separator synthesis, polynomial target-coupled queries without latent-axis
labels, joint action-state identification up to behavioral conjugacy, and an
exponential separation from matched active-automata and synthesis controls.
