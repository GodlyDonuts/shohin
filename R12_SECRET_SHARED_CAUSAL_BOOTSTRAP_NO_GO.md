# R12 Secret-Shared Causal Bootstrap No-Go

**Status:** rejected at invention gate 4. No CPU falsifier, neural fit, or GPU
experiment is authorized from this construction.

## 1. Candidate

The proposed curriculum tried to make a persistent state path compulsory by
splitting a target across causally separated views. For a finite group `G`,
sample a uniform pad `U` independently of target `Y` and reveal

```
V = U^{-1} Y
```

only after the mechanism has committed a state from `U` and the source has
been deleted. The intended extension used a sequence of shares and a running
group product. Episode-private relabelings, conjugations, and hidden-state
interchanges were proposed to prevent a fixed local classifier from passing.

The construction does make memory causally necessary. It does not make
reasoning necessary and it does not identify a new mechanism.

## 2. Tight information theorem

Let `|G| = m`. For every prior on `Y`, a uniform independent `U` gives

```
I(Y; U) = 0
I(Y; V) = 0
H(Y | U,V) = 0
Y = U V.
```

The first independence is immediate. For the second, for every `y,v`,

```
P(V=v | Y=y) = P(U = y v^{-1}) = 1/m.
```

The best single-share accuracy is `max_y P(Y=y)`, not `1/m` unless `Y` is
uniform.

Suppose a sequential mechanism reads `U`, commits state `S`, loses access to
`U`, then reads `V` and must return `Y`. Zero error on every pair requires at
least `m` distinguishable states. If `u != u'` produced the same state, then
the decoder given any fixed `v` would have to return both `uv` and `u'v`,
which differ by cancellation. Therefore

```
|S| >= m
B >= ceil(log2 m).
```

For uniform `Y` and error at most `epsilon`, Fano's inequality yields the tight
lower bound

```
I(U; S) >= log2(m) - h2(epsilon) - epsilon log2(m-1).
```

Retaining `S=U` and returning `SV` attains the exact bound. For an ordered
sequence of shares whose product is `Y`, a running product uses `log2(m)`
state bits independent of sequence length, while any `T-1` shares reveal no
information about a uniform target.

This is a valid one-way communication and streaming-memory theorem. It is not
a computational reasoning theorem.

## 3. Transition-mask variant also collapses

The less direct variant masked transitions rather than the final answer. Let
semantic state evolve as `x_t = a_t x_(t-1)`, sample independent uniform pads
`r_t`, and expose

```
c_t = r_t a_t r_(t-1)^{-1}
z_t = r_t x_t.
```

Then the masked state obeys the ordinary recurrence

```
z_t = c_t z_(t-1).
```

For fixed semantic values, the map from `(r_(t-1), r_t)` to
`(z_(t-1), c_t)` is a bijection. Thus the previous masked state and current
masked transition are independent and uniform. A reset or state-free updater
cannot recover `z_t` above chance; an accurate updater must carry the same
Fano-bounded state information.

However, this is a time-dependent change of coordinates, or gauge transform,
of the original group action. It is conjugate to the same residual automaton.
The exact collapse test succeeds: the proposal is ordinary recurrence in
masked coordinates.

## 4. Fatal non-identifiability

### 4.1 The answer-share task is decryption

`V=U^{-1}Y` contains a one-time-padded target. Combining the shares recovers an
encoded answer; it does not infer an answer from independent axioms or facts.
A matched model that receives both shares is exact with one group product.

### 4.2 The representation is not identified

For every bijection `phi:G->G`, the pair

```
S = phi(U)
D(S,V) = phi^{-1}(S) V
```

has identical behavior. Causal success therefore cannot select a privileged
latent algebra, coordinate system, or semantic state.

### 4.3 Masked success gives no unmasked-transfer theorem

If masked and unmasked inputs are distinguishable, two models can agree on
every masked training episode and behave arbitrarily differently on every
unmasked input. No amount of masked accuracy or state mediation removes that
extension ambiguity. Adding unmasked examples makes transfer an ordinary
curriculum problem rather than a theorem.

### 4.4 Relabeling does not repair the gap

An arbitrary unseen episode relabeling is unidentifiable without a supplied
operation table or demonstrations. Supplying that support changes the problem
to episode-level task inference or meta-learning. Consistent group
conjugation is an automorphism; in abelian groups it changes nothing, and in
nonabelian groups canonical recovery requires either the conjugating element
or another ordinary deconjugation step.

### 4.5 Serialization can reintroduce shortcuts

The secrecy statement covers the mathematical shares only. Prompt length,
format, group choice, output frequency, mask reuse, RNG coupling, and query
metadata can leak the target. Every serialized benchmark would require a
separate whole-view audit.

## 5. Equivalence and prior-art boundary

- Two shares are perfect secret sharing / a group one-time pad in Shannon's
  perfect-secrecy framework.
- A running partial product is the minimal `m`-state Cayley automaton and
  ordinary recurrence.
- Hidden-state counterfactual swaps are interchange intervention training.
- Episode relabeling with support is meta-learning or task adaptation.
- Masked-to-unmasked staging is curriculum or transfer learning.

Relevant primary sources:

- C. E. Shannon, *Communication Theory of Secrecy Systems* (1949):
  https://onlinelibrary.wiley.com/doi/10.1002/j.1538-7305.1949.tb00928.x
- Geiger et al., *Inducing Causal Structure for Interpretable Neural
  Networks* (ICML 2022):
  https://proceedings.mlr.press/v162/geiger22a.html
- Finn, Abbeel, and Levine, *Model-Agnostic Meta-Learning for Fast Adaptation
  of Deep Networks* (ICML 2017):
  https://proceedings.mlr.press/v70/finn17a.html

The useful project-level delta is only a balanced causal-memory diagnostic:
it can certify that a retained channel carries required source information.
It cannot certify deduction, transferable axioms, intelligent context
compression, or a new state primitive.

## 6. Decision

Reject Secret-Shared Causal Bootstrap and its transition-mask variant as R12
mechanisms. The exact collapse test already resolves the question, so a CPU
neural falsifier would only demonstrate that a recurrent model can learn group
multiplication. Preserve the theorem as a future causal-memory control, but do
not train Shohin on it and do not claim masked decryption as reasoning.
