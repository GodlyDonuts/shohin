# R12 MDL Identifiability No-Go

**Status:** rejected as an R12 invention. Description length supplies a valid
average-risk regularizer after a representation language is fixed; it does not
identify the intended extrapolating program from finite unrestricted traces.

## 1. Candidate

The candidate was to select the shortest program consistent with a finite set
of generator examples and relations, expecting the compact causal rule to beat
memorization and therefore extrapolate to arbitrary compositions.

Let `U` be a prefix universal machine and `K_U(p)` the code length of program
`p`. For a finite dataset `D`, idealized MDL chooses a minimum-length program
whose predictions agree with `D`.

## 2. Exact characteristic-set condition

Let `p_star` be the intended total target program. A finite dataset `D`
identifies `p_star` by MDL only if every total program `q` with

```
K_U(q) <= K_U(p_star)
```

that is not extensionally equivalent to `p_star` disagrees with `D`. Ties need
an additional deterministic rule or strict inequality. This condition is both
necessary and sufficient for exact off-sample identification inside the
declared program class.

It is not an explanation of extrapolation. It says that `D` must already be a
characteristic teaching set against every shorter incorrect program. Finding
or certifying that set carries the full identifiability burden.

## 3. What survives: an average-risk Occam bound

For iid examples from distribution `mu`, prefix coding and a union bound imply
that every zero-training-error program `p` simultaneously satisfies, with
probability at least `1-delta`,

```
R_mu(p) <= (K_U(p) ln 2 + ln(1/delta)) / n.
```

Standard noisy versions replace this realizable bound by an empirical-risk
term plus a square-root complexity penalty. This is useful regularization and
belongs in matched controls. It is a distributional risk statement, not a
guarantee on adversarially held-out compositions or arbitrary late queries.

## 4. Four no-go attacks

### 4.1 Finite off-support patch

For every finite `D`, an incorrect program can agree on `D` and differ at the
first untested input. No finite consistency objective distinguishes them
without a restricted program class or a characteristic set.

### 4.2 Delayed sabotage is cheap

A program can run the target until composition length `L` and then fail. It
needs only the target code plus a counter and a self-delimiting description of
`L`, an overhead `O(log L)`. Testing longer finite horizons does not create a
qualitative simplicity separation.

### 4.3 Universal-machine dependence

Kolmogorov complexity is invariant only up to a machine-dependent additive
constant. A perverse but universal reference machine can assign a one-bit code
to a chosen malicious extrapolator. At the tiny description differences at
issue, the representation language is a substantive prior, not a neutral law.

### 4.4 Ideal selection is uncomputable

Finding the shortest total program consistent with arbitrary data requires
solving program equivalence/totality problems. Delayed failures can be placed
beyond any computably chosen finite test horizon. The idealized selector is
therefore not an executable training mechanism.

## 5. Prior-art and resource boundary

The surviving theorem is classical Occam/MDL/PAC-Bayes reasoning. Neural
compression, minimum circuit size, program synthesis, and Bayesian priors are
valid controls, but each inherits a representation language and a hypothesis
class. None supplies the missing ordinary-trace theorem that reveals the right
latent causal presentation.

## 6. Verdict

No CPU falsifier is authorized for unrestricted MDL. A later candidate may use
description length only after separately proving:

1. a computable restricted language;
2. a polynomial characteristic set generated without target-specific search;
3. robustness to noise and representation changes;
4. a held-out compositional guarantee stronger than iid average risk.

MDL can rank survivors inside a theorem-backed class. It cannot create that
class or prove reasoning by itself.
