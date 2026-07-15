# R12 Counterfactual Cursor-Action Theory

**Status:** theory and identifiability result only. No Shohin fit or H100 job is
authorized by this document.

**Claim class:** a bounded causal-controller training hypothesis. The cursor is
not a new computational primitive. At fixed maximum depth it is exactly a
finite-state transducer and, under fixed-duration steps, a positional table.

## 1. Capability being isolated

Let the operation alphabet be

```
Omega = {add, subtract, multiply, remainder}
```

and let `pi` be a permutation of all four operations. A source `x(pi, r)`
renders the four operation clauses in order `pi` under renderer `r`, while
holding the operation inventory and clause-local operand text fixed. The
control state is

```
c in {0, 1, 2, 3, 4}.
```

The action relation is

```
F(x(pi, r), c) = pi[c]  when c < 4
F(x(pi, r), 4) = DONE.
```

This is intentionally smaller than reasoning. It isolates the missing action
policy diagnosed by `R12_OPERATION_SELECTION_LIKELIHOOD_RESULT.md`. Passing it
does not establish arithmetic execution, state transport, source compilation
outside this grammar, or autonomous halting.

## 2. Exact shortcut theorem

The finite board contains every one of the `4! = 24` permutations, five
renderers, and all five cursor states. It therefore contains exactly 600 cells.
Within every renderer and cursor below DONE, every operation is the target six
times.

The following upper bounds are exact for deterministic unique-top-1 policies:

1. A source-only policy receives five distinct targets for each fixed source
   and can score at most `1/5 = 20%`.
2. A global constant policy scores exactly `120/600 = 20%` because all five
   labels are balanced globally.
3. A renderer-only or source-family-only policy also scores at most 20%.
4. A cursor-only policy can score DONE on all 120 `c=4` cells and one of four
   balanced operations on 30 of 120 cells at each other cursor. Its exact
   ceiling is `(120 + 4*30)/600 = 40%`.
5. A renderer-plus-cursor or clause-position-only policy has the same 40%
   ceiling because operation order is balanced independently inside every
   renderer.
6. Replacing every cursor by zero makes an otherwise exact policy emit the
   first operation in all five cells of each source. It scores exactly
   `120/600 = 20%`.
7. Applying a fixed five-cycle derangement to the cursor gives an otherwise
   exact policy the wrong one of five unique labels in every source and scores
   exactly `0/600`.

The full source-plus-cursor relation has a 600/600 realization. Consequently,
the board identifies joint source/cursor use against these named shortcut
classes. It does not identify the neural coordinates or prove extrapolation.

## 3. Event-triggered cursor

For one-call decoding, define model state `(c, phase)` where `phase` is SELECT
or EXECUTE. It is updated only by emitted token events:

```
SELECT + operation-token -> EXECUTE, cursor := min(c + 1, 4)
EXECUTE + COMMIT-token   -> SELECT,  cursor unchanged
SELECT(c=4) + DONE-token -> HALT-PENDING, cursor unchanged
HALT-PENDING + EOS-token -> HALT
all other tokens         -> state unchanged.
```

Premature DONE at `c<4` and operation tokens at terminal `c=4` leave the state
unchanged and count as protocol failures. The Q intervention is exactly zero
outside SELECT phase, so execution tokens cannot observe or be perturbed by the
already-advanced next cursor.

The operation tokens already exist as single tokenizer IDs for leading-space
`add`, `subtract`, `multiply`, and `remainder` (`820`, `5498`, `4307`, and
`7486` under the current tokenizer). The leading space is part of each token;
the unspaced spellings are different token IDs. The eventual neural
preregistration must freeze one existing single-token COMMIT marker and audit
that it is never ambiguously emitted inside an execution segment. This theory
document does not choose that marker.

The state update must execute inside the model's decoding interface. A host
parser that recognizes prose, decides when an operation ended, chooses the next
cursor, or supplies DONE is an external scheduler and invalidates an autonomous
claim. Standard observation of a generated token ID by the model runtime is
counted in the mechanism and in the retained-state ledger.

## 4. Collapse theorem and claim boundary

For a fixed maximum cursor `T`, the state has `T+1` cursor values and finitely
many phases. Let `e_s` be a one-hot state and let `M_a` be the deterministic
transition matrix for emitted token class `a`. Then

```
e_(t+1) = M_(a_t) e_t.
```

Any cursor-conditioned query perturbation has the form

```
q' = q + A b(c),
```

for fixed code `b`. This is ordinary query projection on augmented features,

```
q' = [W_Q  A] [h ; b(c)].
```

It is therefore exactly a tied finite-state recurrence. When advancement is
independent of emitted token boundaries, `c_t=min(t,T)` and the mechanism
collapses further to a clamped positional embedding or fixed indexed pointer.
The factorization only restricts the cursor table to the span of `b`; it does
not create a new state ontology.

At unbounded `T`, exact cursor storage requires `ceil(log2(T+1))` mutable bits
because the remaining distance to DONE distinguishes every cursor state. At
fixed `T=4`, the standalone cursor requires three bits and SELECT/EXECUTE needs
one additional bit. If an existing decoder position or emitted-token history
already determines the state, those are not free: their cache, source bytes,
and recomputation remain in the resource vector.

The strongest allowed primitive-level statement is therefore negative:

> A self-advancing cursor is a known finite-state/pointer mechanism, not a new
> computational primitive.

The remaining admissible hypothesis concerns training:

> Complete operation-order orbits plus counterfactual cursor interchange may
> make a small frozen-model controller learn the joint source/cursor action
> relation more reliably than information-identical ordinary completion loss.

That is an empirical optimization conjecture, not a theorem and not a general
reasoning claim.

## 5. Orbit-interchange objective

For each fixed source `x`, all five cursor interventions are present. For each
renderer and operand assignment, all 24 clause-order permutations are present.
Every arm receives byte-identical rows and targets.

The proposed treatment adds three relation losses to ordinary full-vocabulary
action cross-entropy. Relation matching is computed only on the five frozen
action-token logits after subtracting each row's mean; this removes an
unidentifiable common-logit offset without renormalizing away relative action
evidence:

1. **Cursor interchange:** swapping `c` while holding the source fixed must
   swap the preferred action to the target at the donor cursor.
2. **Adjacent-order equivariance:** applying an adjacent transposition to two
   source clauses must swap only the affected cursor targets.
3. **Renderer invariance:** sources with the same ordered clauses but different
   renderers must agree on centered restricted action logits at every cursor.

These losses reveal no target that is absent from the ordinary rows. Their only
possible contribution is an optimization bias toward the intended relation.
A relation-sham arm receives the same tensor counts, coefficients, and
computation with every local cursor relation rotated by exactly `+1 mod 5`.
Its numerical loss magnitude is model-dependent and is not asserted equal.

## 6. Identifiability limits

- Natural traces where cursor is a deterministic function of history cannot
  identify cursor use. Interventions must hold source and visible history fixed
  while changing only the internal cursor.
- Correct intervention behavior identifies dependence on the supplied state,
  not its implementation. Cursor embeddings, hard pointers, and tied
  recurrence remain extensionally equivalent.
- Fixed four-operation traces cannot distinguish a genuine variable-length
  halt policy from memorized depth. Variable-length shared-prefix cases and a
  separate DONE/EOS gate are mandatory after selector confirmation.
- Operation selection alone does not establish operand binding, arithmetic,
  semantic-state transport, error recovery, or source-free context
  compression.
- Forcing EOS from the runtime is a hard length clamp. Autonomous termination
  requires the model to emit DONE and then EOS without a host-supplied length.

## 7. Prior-art boundary

The bounded primary-source audit found no exact disclosure of the complete
conjunction below, but every component and the important pairings are known:

- learned or action-conditioned neural instruction pointers: Brooks et al.
  (ICML 2021) and Oh et al. (ICML 2017);
- cursor-conditioned instruction attention: Chiang et al. (2021);
- explicit neural program counters and instruction-pointer propagation: Fox et
  al. (ICLR 2018) and Bieber et al. (NeurIPS 2020);
- pointer/address selection and external neural memory: Pointer Networks,
  Neural Turing Machines, and Neural Programmer-Interpreters;
- sparse per-head query intervention: LoFiT and DISCO;
- counterfactual interchange intervention training: Geiger et al. (ICML 2022)
  and typed IIT for language models (ACL 2023);
- recurrent depth, latent recurrence, and latent-token reasoning: Universal
  Transformers, Looped Transformers, Coconut, and Abstract-CoT.

No claim may attach novelty to a program cursor, action-conditioned
advancement, cursor-conditioned attention, a query vector, interchange
training, a hard pointer, recurrence, or latent tokens. The narrow unverified
delta is the complete system:

> In one uninterrupted autoregressive generation, a model-emitted control
> token updates a finite operation cursor; a centered cursor code perturbs only
> one preregistered query head; same-lexicon operation-order orbits and cursor
> interchange train that causal variable; and the model emits its own DONE and
> EOS while preserving ordinary execution.

The audit supports only "no exact match found in the bounded search." It does
not support a world-first, patentability, or freedom-to-operate claim. Per-head
query vectors in particular have close published and patent prior art.

Primary-source anchors for the boundary include Brooks et al.,
[*Reinforcement Learning of Implicit and Explicit Control Flow
Instructions*](https://proceedings.mlr.press/v139/brooks21a.html); Vinyals et
al., [*Pointer
Networks*](https://proceedings.neurips.cc/paper/2015/hash/29921001f2f04bd3baee84a12e98098f-Abstract.html);
Geiger et al., [*Inducing Causal Structure for Interpretable Neural
Networks*](https://proceedings.mlr.press/v162/geiger22a.html); and Dehghani et
al., [*Universal Transformers*](https://arxiv.org/abs/1807.03819). These are
boundary references, not evidence that the full conjunction is absent from all
literature.

## 8. Resource vector for the first possible neural canary

The first canary conditions only head 0 in the final block. Its centered
three-bit code in `{-1,+1}^3` is projected by a bias-free `3 x 64` matrix and
added to Q after head reshaping but before QK normalization and RoPE. The base
model is loaded strictly and frozen; the 192-scalar adapter is a separate
sidecar bound to the base checkpoint and tokenizer hashes. Centering is
mandatory because raw `{0,1}` bits with no bias make cursor zero
uninfluenceable. Zero initialization gives exact base-model behavior before
training.

Final-block Q-only placement keeps every cached K/V tensor cursor-independent.
Any earlier-layer, multi-head, residual, K, or V intervention is a different
version and may not be substituted after scores. The exact canary ledger must
include:

```
parameters:          192 treatment scalars
retained state:      3 cursor bits + 1 phase bit at T=4
precision:           explicitly frozen in the implementation preregistration
source bytes:        full prompt KV/cache remains resident
training examples:  identical across treatment and controls
oracle calls:        zero beyond frozen labels
training FLOPs:      matched within a preregistered tolerance
inference FLOPs:     cursor projection plus ordinary model decode
sequential depth:    one model decode step per emitted token
external memory:     ordinary KV cache plus four controller bits
external execution: zero for the selector claim; arithmetic is a separate gate
```

The favorable control receives the same cursor, 192 parameters, state, rows,
optimizer, updates, and compute but ordinary completion loss. Additional
controls receive relation-sham pairings, zeroed cursor projection, constant
cursor, deranged cursor, and an equal-parameter source-only adapter. An
unconstrained eight-entry cursor embedding is a 512-parameter favorable ceiling,
not an information-matched denominator.

Teacher-forced alignment is prefix-causal: the state supplied while predicting
token `t` is reconstructed only from events in tokens `<t`. The selected
operation token is predicted under `(c, SELECT)`; only after that token is
observed does the state become `(c+1, EXECUTE)`. In cached decoding, state
updates after sampling and before forwarding the sampled token. Full replay
must prefix-scan the same event table and match cached logits exactly. No mutable
`model.cursor` field or cursor inside the K/V tuple is allowed.

## 9. Advancement rule

The next artifact is the CPU preregistration. A symbolic board pass can only
establish that the test has the claimed geometry and exact collapse scores. It
cannot authorize a Shohin fit until the prior-art boundary and all matched
controls are frozen. Passing the neural selector checks would still leave the
raw atomic executor gate pending and authorize only the
separate one-call action-sequence/DONE test; it would not establish autonomous
reasoning.
