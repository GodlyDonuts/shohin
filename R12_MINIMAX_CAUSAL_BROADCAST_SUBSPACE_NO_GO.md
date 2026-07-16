# R12 Minimax Causal Broadcast Subspace No-Go

**Status:** **THEOREM-REJECTED AS A REASONING MECHANISM BEFORE CODE.** MCBS may
be useful as a read-only causal-mediation diagnostic, but it cannot identify a
reusable workspace or authorize training a writer/updater in its subspace.

## 1. Proposed object

For latent contrast `delta_i`, downstream consumer `f_ic`, a frozen residual
metric `G`, and rank-`k` basis `B`, define

```text
P_B = B (B^T G B)^(-1) B^T G
Delta_ic   = f_ic(h_i + delta_i) - f_ic(h_i)
Delta^B_ic = f_ic(h_i + P_B delta_i) - f_ic(h_i)
```

The proposed objective minimizes the worst-consumer normalized error between
`Delta^B_ic` and `Delta_ic`. The metric must be frozen; otherwise the meaning of
the complement changes under residual rescaling.

## 2. Exact collapse theorem

For affine consumers `f_c(h) = A_c h + b_c`, projection preservation and
complement ablation are the same condition:

```text
Delta^B_c = Delta_c
iff A_c (I - P_B) delta = 0
iff Delta^perp_c = 0.
```

Let `D = span{delta_i}` and `N = D intersect (intersection_c kernel(A_c))`.
The minimum exact dimension is

```text
k_min = dim(D) - dim(N).
```

Therefore MCBS recovers only the observable quotient of the chosen consumers.
That quotient need not contain task state, an update law, or information useful
to a new consumer.

## 3. Motor-bundle counterexample

Let the true state be `(a, b)` and every fitted consumer depend only on
`a XOR b`. A one-dimensional parity coordinate preserves every fitted causal
effect, while complement ablation removes every fitted effect. It nevertheless
cannot answer a held-out query for `a`. More generally, a finite set of
consumers can be served by a post-hoc answer bundle `(g_1(s), ..., g_m(s))`.

MCBS cannot distinguish that bundle from a reusable state. Adding more fitted
consumers only enlarges the answer table unless the test also requires closure
under unseen state updates and unseen consumers revealed after the subspace is
frozen.

## 4. Prior-art boundary

The ingredients already overlap established methods:

- projection-based causal subspaces and donor swaps: Distributed Alignment
  Search, https://proceedings.mlr.press/v236/geiger24a.html;
- counterfactual intervention training: Interchange Intervention Training,
  https://proceedings.mlr.press/v162/geiger22a.html;
- causal activation localization: ROME causal tracing,
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html;
- worst-group shared subspaces: Fair PCA,
  https://proceedings.neurips.cc/paper_files/paper/2019/file/2201611d7a08ffda97e3e8c6b667a1bc-Paper.pdf.

The defensible synthesis is a vocabulary-free, consumer-supervised, minimax
causal-localization diagnostic. It is not a new reasoning primitive.

## 5. Decision and successor boundary

No Shohin writer, updater, SFT, or coordinate swap is authorized from MCBS.
Any successor must separate reusable state from a finite answer table by
freezing the representation before revealing both:

1. held-out consumer functions;
2. held-out state-update operators;
3. a committed output-code permutation;
4. source-deleted multi-step reuse.

It must pass a workspace-positive synthetic model and reject a dimension-
matched motor-only model under the same compute and scorer. Readout accuracy,
projection preservation, and complement ablation alone are insufficient.
