# R12 PCFT Adversarial Audit

**Decision:** NO-GO for a neural PCFT preregistration from the v1 scorer pass.
The v1 algebra and exhaustive counts survive; the claimed transport interface
does not.

## 1. What survives

For uniform `x in F_17^4`, a complete packet answers every late linear
functional exactly. A packet containing only `(x_0,x_1)` answers all consumers
whose effective functional lies in `span(e_0,e_1)`, and has exact accuracy
`1/17` whenever the effective functional has a nonzero hidden component. A
bijection of the 17 answer symbols preserves those accuracy counts. The
collision theorem and the 15-cell exhaustive result are correct.

This establishes only that a full vector contains information absent from a
rank-two projection.

## 2. Claim-blocking defects

1. **No packet transport occurs.** V1 composes all future affine events into a
   single effective functional and applies that functional to the initial
   packet. It never invokes a shared updater on `(packet,event)` after source
   deletion, so it does not test the mechanism PCFT needs.
2. **Custody is simulated rather than process-enforced.** Phase one freezes
   packet hashes, but the scorer later re-enumerates sources in the same
   process. Pure packet readers now have no source argument, yet this remains a
   code convention rather than a separate-process boundary.
3. **Scorer-side recoding is not a late-interface test.** Comparing
   `pi(prediction)` with `pi(truth)` is exactly equivalent to comparing raw
   values when `pi` is bijective. A candidate must instead receive the fresh
   codebook and emit the recoded symbol itself.
4. **The rank-two motor is favorable only on the public subspace, not
   information-matched.** Both arms have four fields, but the motor uses only
   289 distinct packets versus the state's 83,521. Padding equalizes tuple
   width, not retained entropy. It is a useful negative control, not the
   decisive neural comparator.
5. **Sixty-four random fingerprints almost surely reveal the complete state.**
   In four dimensions over `F_17`, an overcomplete random linear system is
   full-rank with overwhelming probability. PCFT is therefore state
   distillation through random projections unless a stronger resource result
   is demonstrated. Beating only the rank-two motor would show that richer
   supervision carries more information.
6. **Unseen depth is not unseen scale.** A fixed `F_17^4` board remains
   compatible with finite source tables. A defensible uniformity claim must
   freeze tests across unseen source states, event parameters, renderings,
   compositions, state dimensions, and depths with one variable-size model.
7. **The exact affine solver is an oracle reference, not a control PCFT can
   beat.** A tie at ceiling defines the remaining oracle gap. Decisive controls
   are same-information direct-state and fixed-full-rank supervision under the
   same architecture and resource ledger.

The original v1 horizon flag was also tautological. It was repaired before the
canonical artifact was frozen: the current implementation executes a
source-free horizon-triggered reader over all 15 decisive cells and all 83,521
sources, scoring exact through depth 8 and zero at depth 9. This repair does not
resolve the seven transport/custody defects above.

## 3. Required v2 boundary

Before any neural fit, an exact v2 must provide:

1. separate oracle, writer, stateless one-event updater, and fresh reader
   processes;
2. serialized packet handoffs and packet hashes committed before challenge
   generation;
3. no source, source pointer, event history, verifier, or stale packet channel
   in updater/reader interfaces;
4. independent late output permutations consumed by the reader, with the
   reader emitting the recoded symbol;
5. executed source-visible, source-pointer, query-visible, stale-packet,
   event-history, shuffled-packet, and horizon decoys;
6. direct-versus-incremental state agreement and donor packet swaps;
7. exact ledgers for utilized packet entropy, persistent state, labels and
   independent label rank, parameters, examples, updates, FLOPs, and search
   budget;
8. score-blind confirmation generated only after checkpoint commitment.

The v2 exact process gate may validate custody and transport mechanics. It
still cannot establish learned reasoning.

## 4. Prior-art boundary

Random linear fingerprints are universal hashing and linear sketches. Learned
future-prediction representations overlap predictive-state representations,
successor features, and random action-conditional prediction objectives.
Explicit recurrent state and state reification are established. The only
potentially defensible project contribution is the combined, process-enforced,
finite-precision, post-commit training and evaluation protocol. No world-first
claim is authorized without a broader primary-literature review.

Starting primary sources:

- universal hashing: https://www.cs.princeton.edu/courses/archive/fall09/cos521/Handouts/universalclasses.pdf
- predictive state representations: https://papers.neurips.cc/paper/1983-predictive-representations-of-state.pdf
- successor features: https://papers.nips.cc/paper_files/paper/2017/hash/350db081a661525235354dd3e19b8c05-Abstract.html
- random action-conditional predictions: https://proceedings.neurips.cc/paper_files/paper/2021/hash/c71df24045cfddab4a963d3ac9bdc9a3-Abstract.html
- linear streaming sketches: https://theory.stanford.edu/~matias/papers/ams_stoc.pdf
- state reification: https://proceedings.mlr.press/v97/lamb19a.html

## 5. Shohin decision

Preserve the v1 result as a static positive/control theorem. Do not train a
PCFT neural model from it. Implement only the separately preregistered exact v2
transport/custody harness. No Shohin adapter, SFT, or H100 job is authorized.
