# R12 Post-Commit Fingerprint Transport Theory

**Status:** THEORY CANDIDATE, NEURAL GATE BLOCKED. The v1 exact scorer passed,
but independent audit found that it never invokes a packet updater, does not
enforce process-separated source deletion, and lets the scorer perform output
recoding. `R12_PCFT_ADVERSARIAL_AUDIT.md` is therefore NO-GO for a neural fit.
Only a process-separated exact v2 transport falsifier may reopen a separately
committed CPU neural preregistration. No Shohin adapter, SFT, or H100 job is
authorized.

## 1. Target

Shohin's missing behavior is not merely a decodable answer. It is a packet that
can be written before a late interface exists, updated after the source is
gone, and reused by consumers that were not available to the writer.

Let certified task state be `x_t in F_p^m` and verified event update be
`x_(t+1)=delta_e(x_t)`. The model writes a quantized packet

```text
z_t = Q(W_theta(history_t)) in {0,...,q-1}^k.
```

Only after `z_t` is committed, sample a challenge `r in F_p^m`. The reader
must return the random linear fingerprint

```text
phi_r(x_t) = r^T x_t mod p.
```

After source deletion, a shared learned updater receives only `(z_t,event)` and
must produce a packet that passes fresh independently sampled fingerprints of
`delta_e(x_t)` through long update chains.

## 2. Anti-motor theorem

For distinct states `x != x'` and uniform `r in F_p^m`,

```text
Pr_r[r^T x = r^T x'] = 1/p.
```

Therefore any packet collision is exposed by a random post-commit challenge
with probability `1-1/p`. On a balanced colliding pair, one deterministic
reader has maximum expected exact accuracy

```text
(1 + 1/p) / 2.
```

This does not eliminate unlimited finite tables. It makes the claim
resource-bounded: with a quantized packet near the information content of
`x`, a fixed bundle of public answers cannot cover a challenge family of size
`p^m` without recovering an equivalent sufficient state.

## 3. Proposed training objective

The treatment has four reasoning-only components around a frozen language
backbone:

1. source/history writer `W`;
2. finite-precision packet quantizer `Q`;
3. one shared event-conditioned updater `U`;
4. challenge-conditioned reader `R`.

For independently sampled post-commit challenges `r,r'`, train

```text
L_direct = CE(R(z_t,r), phi_r(x_t))
L_update = CE(R(U(z_t,e),r'), phi_r'(delta_e(x_t)))
L_close  = distance(Q(U(z_t,e)), stopgrad(Q(W(history_t followed by e))))
L = max_or_group_robust(L_direct, L_update, lambda*L_close).
```

The frozen verifier computes scalar fingerprint labels. The model never
receives canonical state coordinates, a rationale, an opcode sequence, a
transition matrix, or the full recursive solution. Packet erasure/noise and an
explicit `k log2(q)` ledger prevent unbounded analog precision.

At evaluation, writer and source process exit before events, challenges, and a
fresh residue-to-output-symbol codebook are generated. A fresh reader process
has no source, source KV, verifier, retrieval path, or scoring feedback.

## 4. Why it differs from failed Shohin paths

- Unlike ordinary answer SFT or additive forks, the writer cannot know which
  functional will be scored.
- Unlike J-lens/MCBS, the method installs and trains a packet rather than
  decoding a presumed raw-model subspace.
- Unlike R11's fixed consumers, the consumer family is generated only after
  packet commitment and is exponentially larger than a fitted answer bundle.
- Unlike a visible scratchpad, no intermediate text is emitted or reread.
- Unlike an external scheduler, event selection and packet update must occur in
  the learned source-free updater.

## 5. Exact collapse and claim boundary

If enough fingerprint labels reconstruct every certified `x_t`, the method is
state distillation under random projections. That is an allowed description,
not a failure to be renamed. It would establish a learned compact update path,
not autonomous discovery of the task ontology.

Reject the candidate if any condition holds:

- packet capacity materially exceeds the certified state information;
- the reader or updater can access source/KV/verifier state;
- direct writer fingerprints pass but source-deleted updates fail;
- unseen history/event compositions or longer depths collapse;
- random/output-token controls match treatment;
- a favorable GRU, MatrixNet/operator, or fixed-consumer control ties under the
  same resource vector;
- normal language bypass is not bitwise identical to the frozen backbone.

## 6. Required process gate before any tiny neural gate

The v1 scorer pass is insufficient. A v2 runner must first execute writer,
stateless one-event updater, oracle, and fresh reader as separate processes;
commit serialized packets before challenge generation; withhold source and
event history from updater/reader roles; and require the reader itself to emit
the late-recoded symbol. Only after that exact process gate passes may a new
preregistration specify a CPU neural experiment. Its minimum design is:

- certified affine register programs over a finite field;
- train depths 1--6; confirmation depths 12, 24, and 48;
- unseen event-parameter compositions and source renderings;
- at least 64 post-commit fingerprints per final packet;
- fresh output recoding after final packet commitment;
- exact-state positive, same-information direct-state and fixed-full-rank
  controls, equal-width rank-two motor bundle, favorable GRU, random packet,
  source-visible, and horizon-triggered controls;
- identical packet bits, parameters, examples, optimizer updates, and compute
  wherever mathematically possible;
- donor packet swaps and direct-versus-incremental packet agreement;
- three frozen seeds and score-blind confirmation.

No Shohin/H100 experiment is authorized by a CPU scorer pass. A learned CPU
candidate must show a preregistered accuracy/resource advantage over
same-information controls at unseen source states, renderings, event
parameters, dimensions, and depths. Tying the exact affine solver is an oracle
gap, not a rejection; beating only the rank-two motor is insufficient.

## 7. Prior-art boundary

The collision theorem is universal hashing. Learned explicit state targets and
recurrent carriers have substantial prior art, as do predictive-state
representations, communication games, and state-reification networks. The
defensible project hypothesis is the combined training protocol:

> finite-precision source-deleted packets trained through functionals generated
> only after commitment, with learned post-source updates and fresh output
> recoding.

This composition may be experimentally new for Shohin. It is not claimed as a
world-first primitive without a broader literature review.

Primary starting points:

- universal hashing: Carter and Wegman,
  https://www.cs.princeton.edu/courses/archive/fall09/cos521/Handouts/universalclasses.pdf
- State-Reification Networks:
  https://proceedings.mlr.press/v97/lamb19a.html
- predictive state representations:
  https://papers.neurips.cc/paper/1983-predictive-representations-of-state.pdf
