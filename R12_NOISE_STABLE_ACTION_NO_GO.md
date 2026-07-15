# R12 Noise-Stable Action No-Go

**Status:** rejected as an R12 invention. Exact bounded-precision robustness of
a residual state is error-correcting coding; robust execution with noisy repair
is fault-tolerant computation. Nonlinearity does not supply a third mechanism.

## 1. Coding-necessity theorem

Let `R` be an exact residual-state set with separating continuation-query
behavior, and let

```
E : R -> {0,1}^N
```

be a physical representation that tolerates every pattern of at most `t`
Hamming errors.

For all distinct `r,s in R`,

```
d_H(E(r), E(s)) >= 2t+1.
```

Otherwise the two radius-`t` Hamming balls intersect. One corrupted physical
state would then have to decode to both residuals, which some separating future
continuation and query require to answer differently.

The encodings are therefore a classical error-correcting code and obey the
Hamming sphere-packing bound

```
|R| * sum_(i=0)^t binomial(N,i) <= 2^N.
```

For `|R|=2^n` and `t=tau N`, asymptotically

```
n/N <= 1-h_2(tau)+o(1).
```

This lower bound is independent of whether the logical residual update is
linear, nonlinear, recurrent, or neural.

## 2. Converse and exact collapse

Given any code encoder/decoder `(E,D)` correcting `t` errors and any logical
event action `U_a`, define

```
Phi_a(z) = E(U_a(D(z))).
```

Then `Phi_a` realizes a robust physical action under boundary-state corruption
followed by noiseless repair. Thus exact robust residual actions under this
model are coded logical computation. The construction is an exact collapse
test, not an architectural analogy.

If physical noise has full support, a fixed finite system cannot remain exactly
correct forever. Under a binary symmetric channel with `0<p<1`, noise maps one
codeword exactly to another with probability at least

```
beta = min(p,1-p)^N > 0.
```

Exact survival through `T` independent rounds is at most `(1-beta)^T`, which
converges to zero.

## 3. Strong finite construction

Let logical state be `x in {0,1}^n`. An event has control set `I_a`, target
`j_a`, and Boolean rule `f_a`:

```
U_a(x)_(j_a) = x_(j_a) xor f_a(x_(I_a)),
U_a(x)_i = x_i  for i != j_a.
```

Degree-two rules include Toffoli-type nonlinear reversible updates. Encode `x`
with an asymptotically good code of length `N=Theta(n)` that corrects `tau N`
errors. Expander codes already provide constant rate and distance, linear
sequential decoding, and logarithmic parallel decoding
([Sipser and Spielman](https://www.cs.yale.edu/homes/spielman/Research/expanders.html)).

For independent channel noise `p<tau`, a Chernoff bound gives

```
Pr[failure by T] <= (T+1) exp(-N D_KL(tau || p)),
```

so reliability can last exponentially many rounds in `N` with high
probability. This is a strong control, but it is still decode-compute-reencode.

## 4. Nonlinearity can amplify corruption

Toffoli is the smallest reversible nonlinear Boolean gate. The inputs `010`
and `110` differ in one bit, while the corresponding outputs `010` and `111`
differ in two. Nonlinear action by itself can expand Hamming errors.

Likewise, repetition code `000/111` corrects independent single-bit errors but
a correlated `111` fault maps one codeword directly to the other. If the
decoder, fanout, repair, or re-encoder is noisy, the noiseless repair theorem no
longer applies; the problem becomes fault-tolerant circuits or reliable
cellular automata.

## 5. Prior-art collapse

- `E circle D` is an associative-memory attraction map.
- Sparse-constraint message passing is belief-propagation decoding.
- autonomous local repair under noisy repair operations is the Toom/Gacs
  fault-tolerant cellular-automaton problem;
- a learned denoiser is an approximate codeword/MAP decoder;
- `Phi_a` is an ordinary recurrent transition, so a structure-aware recurrent
  comparator can implement the same state and costs.

The positive construction assumes known coordinates, known code geometry,
handed event wiring, bounded or independent faults, and noiseless global
repair. Removing them reintroduces hidden-coordinate non-identifiability,
correlated failure, or established fault-tolerant computation.

## 6. Verdict

No CPU falsifier is authorized. A reconsidered candidate must prove a resource
separation *inside* coded computation: jointly discover action and redundancy,
tolerate noisy repair and correlated faults, and beat structure-aware ECC,
fault-tolerant cellular automata, denoising, and recurrent controls. The current
family does not.
