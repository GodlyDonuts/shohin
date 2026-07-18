# R12 Self-Canonicalizing Epoch Retirement Transformer Theory

**Protocol:** `R12-SCERT-THEORY-PREREG-v4`

**Status:** THEORY AND PROSPECTIVE PREREGISTRATION ONLY. This document creates no
implementation, corpus, checkpoint, confirmation board, secret, execution plan,
or result. It authorizes no code change, CPU capability result, model fit, GPU
allocation, H100 job, checkpoint promotion, or scientific claim. H100 execution
is explicitly `NO-GO` until every item in Section 16 is instantiated, hash-bound,
and independently accepted in a later executable protocol.

**One-sentence hypothesis:** a model-authored EOS candidate, a learned
`COMMIT`-versus-`HALT` decision at a fixed clean probe, position-matched
weight-shared reconstruction of the exact latest-state token IDs, and destructive
K/V replacement can remove stale-state mediation that post-hoc cache masking
cannot remove.

## 1. Frozen diagnostic basis

These supplied development observations motivate the architecture. They are not
results of this protocol and cannot be used as confirmation evidence.

1. The exact 12-case board has two cases in every
   `width {4,6,8} x operation {add,sub}` cell.
2. Full stale history `S0 + generated S1` achieved paired carry-causal exactness
   `2/12`, split `0/4`, `2/4`, and `0/4` by width.
3. A fresh source-deleted prompt containing the same latest state achieved
   `10/12`, split `3/4`, `4/4`, and `3/4`. Its nominal target was exact `12/12`,
   counterfactual target `10/12`, and output switch `12/12`.
4. Three post-hoc cache arms failed to reproduce the fresh prompt: latest-state
   K/V only, immutable prefix plus latest-state K/V, and deletion of stale `S0`
   keys while retaining contextualized suffix/latest-state tensors.
5. The negative is mechanistically expected: K/V for `S1` was formed while
   `S1` could attend to `S0`, so `S1` remained a mediator after direct `S0` keys
   were hidden.

The observations identify a stale-context intervention target. They do not show
that emitted `S1` is valid, that the model can author a boundary, or that
retirement improves a complete autonomous trace.

## 2. Question and strongest admissible claim

The question is:

> Given only a frozen instruction scaffold and the model's own latest-state
> token IDs, can a tied Transformer re-encode that state in a clean epoch,
> replace the old contextualized cache, and continue until its learned boundary
> head accepts a model-authored EOS without host semantics, arithmetic, repair,
> scheduling, or gold tokens?

The strongest eventual positive claim is intentionally narrow and post-dispatch:

> On the frozen bounded DWS family and declared runtime, SCERT establishes
> universal post-dispatch equality under `do(P=a/b)` conditional on fixed
> `(X1,E,D1,Q_e,I)`, separately establishes observational independence only if
> its common-support denominator passes, and improves autonomous exact
> continuation under a finite hidden-board total-effect comparison.

This is an architecture-plus-runtime claim. It is not a claim about an ordinary
uninterrupted-KV Transformer.

### 2.1 Append-only context versus overwrite semantics

Ordinary causal context is append-only: a later token can add evidence or mask a
direct key, but it does not rewrite the contextual representation already built
for an earlier token. Algorithmic state machines instead rely on overwrite
semantics: after transition `S0 -> S1`, future execution should consume one
current state, not every historical presentation of that state.

Let `last(H)` be the exact latest-state token string in history `H`, let `P(H)`
be the mechanically retained prior-source token slot, and define

```text
H ~ H'  iff  last(H) = last(H').
kappa_I(H) = Keep(Enc_theta(R_pm(P(H), token_ids(last(H))),
                            M_clean, p_pm)).
```

Because `M_clean` blocks every path from `P(H)` into every retained position,
SCERT maps every admitted history in one `~` class to the single cache
representative `kappa_I(H)`. If the latest-state string is causally sufficient for
all allowed continuations, this same-string partition approximates the relevant
Nerode quotient: histories indistinguishable by every future continuation are
represented once rather than as many stale-context aliases. This may reduce
history-alias sample complexity by making one state string correspond to one
training and inference representation.

That is motivation, not a theorem about learning. The same-string partition can
be too coarse when the string omits necessary state and too fine when multiple
strings are behaviorally equivalent. This document proves no sample-complexity
bound, novelty, general algorithm learning, or SoTA advantage.

## 3. Exact objects and autonomous boundary machine

Let:

```text
theta       shared Transformer and output weights
G_L, G_R    frozen global instruction tokens before and after the state slot
P_e         exact token IDs of the source state consumed during epoch e
X_e         non-EOS token IDs authored during epoch e
E_e         event that the one effective-logit argmax is EOS ID 0
B_phi       learned linear COMMIT-versus-HALT head
D_e         boundary decision in {COMMIT, HALT}
Q_e         complete non-cache runtime state immediately before dispatch
Enc         the ordinary weight-shared Transformer encoding map
KV_e        the active retained per-layer K/V cache after dispatch
```

The frozen tokenizer is `artifacts/shohin-tok-32k.json`, SHA-256
`87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4`.
There is exactly one vocabulary-logit surface at every token decision. Let
`h_t` be the post-final-norm residual, let `U` be the tied unembedding, let
`v0/v1` be the distinct single-character digit IDs bound in Section 9, and let
`a_M` be the frozen motor level for the arm or training stage:

```text
ell_base_t = U h_t
delta_t    = m_psi(h_t)
ell_eff_t  = ell_base_t
ell_eff_t[v0] += a_M * delta_t[0]
ell_eff_t[v1] += a_M * delta_t[1]
y_t        = argmax_lowest_id(ell_eff_t)
```

`a_M=1` in stage-one fitting and every `M1` arm; `a_M=0` in every `M0` arm,
which still computes `delta_t` and discards it. `delta_t` is cast to the dtype of
`ell_base_t` before the two indexed additions. `ell_eff_t`, and no pre-motor or
second surface, determines the emitted token, event detection, every reported
vocabulary margin, and the stage-one vocabulary loss. Thus `E_e` is exactly the
discrete event `y_t=0`, where token ID `0` is `<|endoftext|>`; when `y_t!=0`,
that same `y_t` is appended to `X_e`. The motor cannot write the EOS coordinate
because `v0` and `v1` are distinct from EOS ID 0, but its two additions can
change which token wins against EOS, so event and emission must share this one
argmax. A later executable protocol must reject any tokenizer, motor order,
dtype cast, tie rule, training loss, or decoder that creates a second predicate.
`COMMIT` is not a host-selected delimiter or a vocabulary token; it is
`D_e=COMMIT` after the model itself creates `E_e`.

The exact ASCII scaffold bytes are:

```text
<|system|>
SCERT-DWS-v1.
Current state is the only mutable state.
If it is nonterminal, emit exactly one canonical successor state and end the response.
If it is terminal, emit exactly answer=<integer> and end the response.
Emit no explanation.
<|user|>
Current state:
{STATE}
<|assistant|>
```

`G_L` ends immediately before `{STATE}` and `G_R` begins immediately after it.
Their token IDs and SHA-256 must be frozen before any data build.

The boundary head is the minimal affine classifier

```text
B_phi(h) = W_B h + b_B,  W_B in R^(2 x 576), b_B in R^2.
columns = [HALT, COMMIT]; an exact tie selects HALT.
```

It is evaluated only after `E_e`. Every primary arm first executes the same clean
classification replay defined in Section 4. `h_probe` is the residual at the
fixed final non-padding token of `G_R`, after all `X_e` positions, under
`M_clean`. Thus the probe position and input surface are identical across primary
arms. `D_e=argmax B_phi(h_probe)`. `h_raw`, the residual whose `ell_eff` argmax
created `E_e`, is receipt-only and may feed a separately labeled diagnostic; it
never selects a primary-arm action or cache. The head is therefore a separately
trained clean-state classifier, not a claimed counterpart of `h_raw`.

`Q_e` is not shorthand for phase alone. It is the complete tuple of every
non-cache runtime variable capable of changing dispatch or any future call:

```text
Q_e = (phase, commit_count, candidate_count, epoch_token_count,
       total_token_count, replay_slot_cursor, generation_slot_cursor,
       cap_constants, failure_flag, RNG_state_and_cursor,
       deterministic_tie_state, publication_receipt_cursor).
```

An executable protocol must fail closed if any future-relevant runtime variable
is absent from `Q_e`. The exact source IDs `P_e`, generated IDs `X_e`, model cache,
and write-only transcript are separate declared objects, not hidden fields in
`Q_e`.

The runtime has `ACTIVE` and `HALTED` phases. In `ACTIVE`:

```text
effective argmax is not EOS: append that same token ID to X_e and update Q_e
event E_e occurs:       freeze exact (P_e,X_e,E_e,Q_e);
                        run the common clean classification replay;
                        compute D_e from h_probe;
                        run the arm's position-matched reconstruction replay
  D_e = COMMIT:         suppress candidate EOS as an internal event;
                        atomically install the retained reconstruction K/V;
                        set P_(e+1) := exact X_e, clear X_e;
                        update Q_(e+1) deterministically and continue from the
                        common final-G_R replay endpoint
  D_e = HALT:           accept candidate EOS, update Q_(e+1), enter HALTED,
                        and return the write-only transcript
```

The reconstruction replay is executed in every arm even on `HALT` and then
discarded, so action timing and compute do not depend on the arm. Empty,
malformed, repeated, or semantically impossible spans are replayed and
classified without validation or repair. The caps stored in `Q_e` permit at most
eight `COMMIT` decisions, nine EOS candidates including final `HALT`, and 512
generated tokens per epoch. Reaching a cap stops with protocol failure, never
success. A cap is not a schedule and does not select a boundary.

For a correct width-`w` episode, the model must author exactly:

```text
S1 [event E -> COMMIT]
S2 [event E -> COMMIT]
...
Sw [event E -> COMMIT]
answer [event E -> HALT]
```

EOS alone cannot distinguish a microstep commit from final halt. Always-HALT
stops at the first state; always-COMMIT never accepts a final EOS. The learned
head, not a host parser, must make the distinction. The host does not know or use
`w`, `z`, answer syntax, or terminality during scored generation. Commit count,
state syntax, answer, and width are parsed only after generation has ended.

## 4. Position-matched SCERT reconstruction operator

At `E_e`, the runtime copies IDs without decoding them. Under the frozen tokenizer
and exact scaffold bytes above, `G_L` is exactly 70 token IDs and `G_R` is exactly
3. The runtime pads `P_e` and `X_e` to separate fixed 512-position slots with one
frozen dummy token ID and frozen validity bits. The dummy ID is exactly EOS ID 0
with validity false; a masked dummy never creates event `E_e`. A later executable
protocol must bind all 73 scaffold IDs, assert dummy ID 0, and reject a source or
authored span longer than 512 before scoring. Define:

```text
P_e*      = fixed 512-position source slot containing exact P_e IDs
X_e*      = fixed 512-position authored-span slot containing exact X_e IDs
R_pm      = G_L || P_e* || X_e* || G_R
p_pm      = 0, 1, ..., 1096
Keep      = retain only valid G_L, X_e, and G_R positions; drop P_e* and pads
```

The position ranges are exact and immutable:

```text
G_L       [0,70)
P_e*      [70,582)
X_e*      [582,1094)
G_R       [1094,1097); fixed boundary probe at position 1096
generation starts at position 1097 and may occupy at most [1097,1609)
```

Thus `R_pm` has exactly 1,097 physical positions. A maximum 512-token generated
span ends at position 1,608; its effective EOS argmax creates `E_e` but is not
appended.
No claim-bearing call approaches the 2,048-position context limit.

For a cloned one-dispatch mechanistic event, core `C0` and `C1` receive
byte-identical `R_pm`, validity, positions, weights, and physical shapes. In
autonomous evaluation each arm uses the same schema on its own arm-authored
event, so later semantic IDs need not match. Both masks are causal and identical
except for one declared edge family:

```text
M_clean: valid X_e queries cannot attend valid P_e source positions
M_stale: valid X_e queries can attend valid P_e source positions
```

In both masks, `G_R` queries may attend `G_L`, valid `X_e`, and earlier `G_R`
positions, but never `P_e*`; padding is invisible; and all other allowed edges
are identical. Consequently the only treatment difference is construction-time
`X_e <- P_e` access. Absolute positions, prompt length, framing, valid tokens,
and the final continuation position do not change.

At any arm's own primary event, that arm executes exactly two dense replays:

```text
H_cls       = Enc_theta(R_pm, p_pm, M_clean)
h_probe     = H_cls[fixed final token of G_R]
D_e         = argmax B_phi(h_probe)
H_rec(C1)   = Enc_theta(R_pm, p_pm, M_clean)
H_rec(C0)   = Enc_theta(R_pm, p_pm, M_stale)
KV_rec(Cj)  = Keep(H_rec(Cj).KV)
```

`C1` deliberately re-executes its clean pass rather than reusing `H_cls`, so
the policies have identical forward counts at a given event. On `COMMIT`, the
active arm atomically installs its `KV_rec`, erases all dropped source/pad K/V
and its old active cache, clears `X_e`, sets its next exact source-ID buffer to
its committed `X_e`, and continues at the position immediately after the final
`G_R` token. The candidate EOS is not installed. On `HALT`, the active arm
discards its `KV_rec` and accepts its own event. Cloned mechanistic arms share
that one event by construction; autonomous arms may author different later
events and actions. There is therefore no repeat-EOS versus append-EOS
ambiguity in `C0`.

The replacement is atomic: continuation may observe either the prior complete
cache before dispatch or the complete retained reconstruction after it, never a
mixture. In `C1`, `P_e*` can be computed as isolated dummy work but has no path to
any retained representation and is erased at dispatch. A transcript may be
retained write-only for later scoring, but it is not model input.

This is endogenous replay and destructive cache replacement. `P_e` and `X_e`
are exact runtime token buffers; replay content, event timing, and the learned
action are model-authored. Buffering, padding, masking, argmax, and atomic swap
are fixed mechanical resources. A compact `G_L || X_e || G_R` replay with reset
positions is a separate diagnostic only; it has no primary causal standing.

## 5. Causal graph with complete runtime state

For one boundary, use:

```text
I        = scaffold, theta, head/motor, slot widths, masks, positions, caps,
           tokenizer, numerical environment, tie rule, and runtime mode
S0       = stale source token IDs loaded into P_1*
U1       = decode randomness used while authoring X1 and event E
X1       = exact authored non-EOS token IDs
E        = model-authored EOS-argmax boundary event
Q_e      = complete pre-dispatch non-cache runtime state
H_cls    = Enc_theta(R_pm, M_clean, p_pm)
D1       = B_phi(H_cls[fixed final G_R token]) in {COMMIT,HALT}
H1_C0    = reconstruction under M_stale
H1_C1    = reconstruction under M_clean
Z1_Cj    = complete post-dispatch state (retained K/V, Q_(e+1), buffers, phase)
UF       = future exogenous random bits
Future   = all later token IDs, events, actions, and stops through HALT or cap
```

Before conditioning, stale history is allowed to affect what the model authored,
when it emitted EOS, and the runtime counters:

```text
S0 ---> X1 <--- U1
 |       |
 +-----> E
 +-----> Q_e
 +-----> H1_C0 ---> Z1_C0 ---> Future
```

The claim-bearing `C1` post-dispatch graph is instead:

```text
(X1,I) ------> H_cls ------> D1
   |             |
   +----------> H1_C1
(X1,E,D1,Q_e,I) ----------> Z1_C1 ----------> Future <--- UF
S0 ---> isolated P_1* --X--> every retained C1 position
```

`--X-->` denotes a structurally prohibited edge in `M_clean`, not an observed
zero. This graph supports two different statements: a universal intervention
equality for the dispatch function, and an observational conditional-independence
corollary only where the conditioning event has support.

## 6. Structural dispatch invariance and observational corollary

### 6.1 Universal structural intervention target

Fix `I=i`, model/motor/head bytes, runtime arm `C1`, and any syntactically valid
fixed values `(x,e,d,q)`. Let `a` and `b` be any two source-ID spans of length at
most 512, including spans never produced observationally. Assume:

1. `Q_e=q` contains every non-cache runtime parent of dispatch and future calls;
2. `R_pm`, positions, and `M_clean` are deterministic, all retained queries are
   structurally blocked from all valid source-slot keys, and source-slot validity
   cannot alter a retained mask row;
3. the action is produced only by the clean classifier at fixed position 1096,
   never by `h_raw`;
4. the atomic dispatch map updates counters, buffers, phase, positions, and RNG
   cursor only from `(x,e,d,q,i)` and removes every older cache, residual, source
   pointer, position offset, and hidden seed path;
5. numerical kernels satisfy the bound deterministic contract; and
6. future sampling uses the same law or the same exogenous bits `UF`.

Then the implementation-level intervention equality is universal over valid
source contents:

```text
Z1_C1(do(P_1=a),x,e,d,q;i)
=
Z1_C1(do(P_1=b),x,e,d,q;i).
```

With common `UF`, every later token, event, action, cap, and stop is also equal.
This is a functional software property. It does not require `a` or `b` to have
positive probability under the model's observational trace distribution.

### Proof

Under `M_clean`, changing source IDs or source validity in `[70,582)` cannot alter
any retained `G_L`, `X1`, or `G_R` representation. Therefore `h_probe`, `D1`, and
`KV_rec(C1)` are equal for fixed `(x,e,d,q,i)`. The post-dispatch buffer and
`Q_(e+1)` are the same deterministic image of that tuple, so `Z1_C1` is equal.
Induction from equal post-dispatch state and common `UF` gives equal future
execution. QED.

### 6.2 Observational common-support corollary

The separate observational statement is:

```text
Future _||_ P_e | (X_e,E_e,D_e,Q_e,I), runtime_arm=C1.
```

It is asserted only for tuples with positive probability under the declared trace
distribution. It does not follow from observing off-support `do(P)` pairs. A
hidden observational audit starts with exactly 432 candidate history pairs:

```text
3 edit families x 3 widths x 2 operations x 24 = 432.
```

The custodian freezes candidates before future decoding. A pair is admitted only
when independently generated histories have different `P_e` but exactly equal
`(X_e,E_e,D_e,Q_e,I)` before looking at any post-dispatch tensor or output. The
admitted denominator, rejected case IDs, and rejection reasons are immutable.

An observational independence claim is forbidden unless at least 216/432 pairs
are admitted overall and at least 12/24 are admitted in every one of the 18
`(edit,width,operation)` strata. If that minimum fails, the result is
`OBSERVATIONAL DENOMINATOR NO-GO`; the universal structural intervention theorem
and its 384-case mechanistic assay remain separately reportable.

This section says nothing about whether stale source changes the probability or
timing of `X_e`, `E_e`, `D_e`, or `Q_e`, and nothing about the semantic correctness
or sufficiency of `X_e`.

## 7. Why post-hoc mask-only retirement cannot prove the target

For an `S1` token `i` encoded in full history, one attention layer contains a
term of the form:

```text
h_i = F(x_i, sum_j alpha(i,j) V h_j), where j includes S0 positions.
K_i = W_K h_i
V_i = W_V h_i
```

When any `alpha(i,j)` on stale position `j` is nonzero, `h_i`, `K_i`, and `V_i`
can depend on `S0`. Deleting the stale keys only after these values exist removes
the direct path `S0 -> Future`, but leaves:

```text
S0 -> H(S1) -> K/V(S1) -> Future.
```

The path is open after conditioning on the token IDs of `S1`, because those IDs
do not determine their contextual hidden states.

A scalar counterexample is sufficient. Let a full-history encoder create
`h(S1)=x+s0` and let the future read only `h(S1)`. After masking the `S0` key,
`Future=x+s0`; fixing `x=token_ids(S1)` does not remove dependence on `s0`.
Therefore post-hoc masks, key deletion, and cache slicing cannot establish the
theorem. A construction-time mask that blocks every `S1 <- S0` path at every
layer could establish it, but that is a new clean encoding operation, not a
post-hoc mask over already contextualized K/V.

The same proof applies to boundary control. If `B_phi` reads `h_raw`, then
`S0 -> h_raw -> D_e -> Future` remains open even if installed K/V is clean. The
theorem-bearing arm must classify the fixed final-`G_R` probe from the common
`M_clean` classification pass. A raw-residual head is a useful contaminated
diagnostic, not SCERT and not a primary arm.

## 8. Forbidden runtime resources

During every claim-bearing autonomous evaluation, the model/runtime must do none
of the following:

- decode or parse a DWS field, integer, operation, width, cursor, carry, result,
  terminal flag, answer, or state validity;
- perform arithmetic, an ALU call, solver replay, schedule construction, state
  transition, state repair, canonical reserialization, verification, retry, or
  beam/rerank selection;
- inject, force, replace, or select any gold state, target token, `COMMIT`, EOS,
  answer, head action, or boundary;
- choose a cache arm, checkpoint, seed, epoch, stop, or boundary from model
  output or a score;
- after dispatch, retain old epochs through a hidden cache, residual, source
  pointer, transcript, position offset, random seed, or side channel.

The declared `P_e` source-ID slot is permitted only during the position-matched
reconstruction in Section 4. In `C1` it is attention-isolated from every retained
position and is erased at dispatch; using it anywhere else is a forbidden hidden
source path.

The only online boundary predicate is `y_t=EOS` on the single `ell_eff_t`
surface in Section 3, followed by the frozen head's two-way argmax on the fixed
clean `h_probe`. The carry motor has no router: in SCERT mode it is evaluated
from the current residual at every next-token decision and its declared level
controls only the two indexed additions for the tokenizer's single-character
digit tokens `0` and `1`. It receives no token text, token class, site flag,
parsed field, event flag, or position label. A pre-motor EOS check, post-EOS
motor call, or separate emission argmax is forbidden.

In particular, the scored arm may not parse `z`, recognize a state line, inspect
`answer=`, count expected microsteps, or use width to decide `COMMIT` versus
`HALT`. Those operations are allowed only in explicitly non-autonomous ceilings.

Offline parsing and arithmetic are allowed only after a call has stopped, to
construct training labels before fitting, or to construct explicitly labeled
causal diagnostics and the non-autonomous host ceiling. None may feed a
claim-bearing generation.

## 9. Parameter and runtime resource dossier

The deployment count uses tied embeddings once.

| Component | Unique parameters | Trainable in proposed fit |
|---|---:|---:|
| Parent Shohin GPT | `125,081,664` | `125,081,664` |
| Epoch-local masks, positions, buffer, swap, FSM | `0` | `0` |
| Rank-8 carry motor `576->8->2`, with biases | `4,634` | `4,634` |
| Boundary head `576->2`, with bias | `1,154` | `1,154` |
| Total | `125,087,452` | `125,087,452` |
| Strictly addable while total remains `<150,000,000` | `24,912,547` | not allocated |

Each evaluated arm loads exactly one boundary head. The true and shuffled heads
are alternative 1,154-parameter artifacts and are never resident together; the
control replaces the true head rather than adding another head.

The motor count is exact:

```text
576*8 + 8 + 8*2 + 2 = 4,634.
576*2 + 2 = 1,154.
```

Let `v0` and `v1` be the frozen tokenizer IDs that each decode to exactly the
single ASCII character named by the variable. At every SCERT next-token
decision, `m_psi(h)=W_up*SiLU(W_down*h+b_down)+b_up` is computed and only
`ell_eff[v0]` and `ell_eff[v1]` may differ from `ell_base`. The later executable
protocol must bind `v0` and `v1`; this document does not confuse them with EOS
ID `0` or the tokenizer's unrelated padding ID `1`. The protocol's masked dummy
remains EOS ID 0 with validity false; tokenizer padding ID 1 is never substituted
for it. No caller may expose `ell_base` as an alternate event, emission,
target-loss, or scoring surface.

No learned epoch embedding, reset vector, canonicalizer, position table, parser,
verifier, scheduler, motor router, or gate other than the declared boundary head
is permitted. Any such tensor creates a new version and parameter ledger.

At BF16, one full active Shohin K/V cache with `L` real token positions occupies:

```text
30 layers * 2(K,V) * 3 KV heads * 64 head width * L * 2 bytes
= 23,040 * L bytes.
```

At `L=2,048`, this is `47,185,920` bytes. SCERT model-visible retained memory is
bounded by the current reconstructed prompt, never cumulative epoch history. The
runtime also holds one 512-ID source slot, one 512-ID uncommitted span slot, the
complete `Q_e`, and write-only receipts. Actual allocated bytes, transcript
bytes, K/V bytes, masks, `Q_e` bytes, RNG state, model calls, attention-pair
counts, and measured FLOPs must be reported per arm; logical bounds are not
substitutes for measured resources.

Every model-selected EOS event, including final `HALT`, spends two full
weight-shared re-encodings: the common clean classification pass and the
position-matched reconstruction pass. Thus SCERT trades compute for autonomous
clean boundary control and destructive context reset. It is not a free cache
optimization.

## 10. Equivalence and internal re-prompting boundary

### Endogenous replay

SCERT is closest operationally to internal re-prompting:

```text
ordinary prompt:       host supplies state and starts one decode
fresh host re-prompt:  host selects state IDs and starts another decode
SCERT:                 model supplies state IDs and EOS candidate; its frozen
                       head selects COMMIT or HALT; fixed runtime replays
```

SCERT does not semantically summarize history. It copies the exact authored IDs
into the same frozen state slot and reruns the same weights. Its distinction from
fresh host prompting is provenance and control: the model authors replay
content, the EOS candidate, and the learned boundary action, while the host
performs a fixed mechanical dispatch and reset. That fixed reset remains an
essential runtime resource.

### Bounded computational equivalence

With finite vocabulary, bounded epoch length, at most eight commits, BF16 K/V,
finite parameters, finite runtime state, and a finite RNG state, the set of
reachable SCERT configurations is finite. Greedy SCERT is therefore extensionally
a deterministic finite-state transducer/Mealy machine. Sampled SCERT is a finite
probabilistic transducer. An RNN with sufficient finite state can simulate the
same transition system, and a bounded unrolling can simulate it with a fixed
feed-forward circuit.

Destructive replay may improve optimization, interference control, and practical
memory scaling. It does not create a new computability class. No claim extends
to unbounded precision, unbounded state length, unbounded commits, or an
asymptotic separation from RNNs, FSTs, recurrent Transformers, or re-prompting.

## 11. Finite CPU mechanics falsifier

Before any executable neural or H100 protocol may be reviewed, an independent
CPU harness must pass all gates below. The harness is a falsifier of mechanics,
not evidence of learned capability.

The finite board has exactly:

```text
2 frozen instruction wrappers
x 8 equal-length stale-state edit pairs
x 16 fixed latest-state token spans
= 256 paired cases.
```

It uses an explicitly initialized two-layer, width-16, two-head toy causal
Transformer over a 32-token vocabulary in deterministic float64 CPU execution.
Its fixed weights must make the path `S0 -> H(S1) -> Future` nonzero. Board,
weights, masks, positions, and expected outputs must be independently generated
and hash-bound in any later implementation.

All of these are noncompensatory:

1. The one-dispatch harness must clone exact pre-classification `(P,X,E,Q,I)`
   before applying an arm. Each clone must execute its own clean classifier and
   exactly one declared head forward; no probe, logits, or action may be copied
   between arms. Core `C0-true` and `C1-true` receipts must have byte-identical
   surfaces, validity, positions, weights, probe, action, and physical shapes in
   `256/256`; their reconstruction-mask XOR must equal exactly valid `X <- P`
   edges.
2. Clean `C1` retained K/V must be bit-identical, layer by layer, to an
   independent reference encoding of the same `R_pm`, positions, `M_clean`, and
   `Keep` projection in `256/256`.
3. At that sole shared dispatch, `C0-true` and `C1-true` must have bit-identical
   `h_probe`, one-head logits, `D`, event transition, `Q` input, and final-`G_R`
   endpoint before reconstruction K/V is selected. The assay stops after first
   post-dispatch `ell_base`, motor delta, `ell_eff`, and token.
4. Under arbitrary `do(P=a/b)`, `C1` post-dispatch state, next `ell_base`, motor
   delta, `ell_eff`, and effective-argmax token must be identical in `256/256`.
   `C0-true`, `C0-neutral`, and `C0-shuffled`
   must have identical open edges; the planted fixture must make only structured
   true content move output toward its source-implied target in `256/256`.
5. Random-byte or NaN poisoning of every dropped source position and retired
   cache tensor after swap must leave `C1` outputs unchanged in `256/256`.
6. Positions must be fixed and equal across mechanistic arms. Any inherited
   history offset, compact-position substitution, shifted probe, or changed
   surface length must fail an exact receipt assertion.
7. Every `ACTIVE/HALTED` transition and complete `Q_e -> Q_(e+1)` map must be
   exhaustively checked, including non-EOS tokens, empty span, consecutive EOS
   events, both actions, all caps, and post-HALT tokens. Only the Section 3
   effective argmax can create `E_e`.
8. The runtime API may accept only exact token IDs, validity, model/cache,
   complete `Q_e`, prebound arm ID, and RNG state. Decoded text, parsed fields,
   gold token, boundary index, or schedule must be impossible by schema.
9. Every arm must execute exactly one 1,154-parameter head forward per event:
   true for treatment, shuffled for shuffled control, true-and-discard for fixed
   or oracle controls, and true-on-raw instead of true-on-clean for the separate
   raw diagnostic. A two-head receipt must fail.
10. A two-event autonomous toy must deliberately diverge after its first
    reconstruction and prove each arm subsequently consumes only its own tokens,
    `E`, `Q`, action, and stop. Any equality-enforcement or cross-arm copy fails.
11. Always-HALT, always-COMMIT, and fixed-count policies must show exact
    precomputed finite action traces. No policy may inspect token text.
12. Motor-off must compute and discard the same delta at every token decision.
    Between motor-off and motor-on, every `ell_eff` coordinate except the two
    frozen single-character digit IDs must be bit-identical. There is no site
    predicate or alternate event surface.
13. A finite-state enumerator must reproduce every toy transition and stop state
    exactly, recording finite-state collapse rather than claiming a new primitive.

One failed item is mechanics NO-GO. Passing opens only review of an executable
preregistration; it does not authorize it.

## 12. Frozen training proposal

The proposed parent is
`train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt`, SHA-256
`d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459`,
with 30 layers, width 576, 9 query heads, 3 KV heads, and context 2,048.

Training has exactly 2,048 width-4 add/sub episodes per independent seed:

```text
2 operations x 8 intermediate carry/borrow patterns x 128 = 2,048.
```

For episode `n`, lane `j in {0,1,2,3,4}` has exact current-state token IDs
`P[n,j]`, exact target-span IDs `X[n,j]`, and action label `A[n,j]`. Lanes 0-3 are
the four `current state -> successor state` transitions with `A=COMMIT`; lane 4
is `terminal state -> answer` with `A=HALT`. The data builder rejects any
`P[n,j]` or `X[n,j]` longer than 512 IDs. It never truncates, reparses, or
reserializes a span after tokenization.

### 12.1 Stage-one base/motor tensor

Every lane is one dense tensor row of exactly 2,048 token IDs, position IDs
`0..2047`, boolean valid-key bits, a frozen attention mask, and shifted labels.
The dummy/padding token is exactly EOS ID 0 with validity false. Valid IDs are
left-aligned in their slots. The row is:

```text
positions       content
[0,70)          exact G_L IDs
[70,582)        512 dummy IDs; all invalid (empty retired-source slot)
[582,1094)      P[n,j] IDs, then invalid dummy IDs
[1094,1097)     exact G_R IDs
[1097,1609)     X[n,j] IDs, then invalid dummy IDs
[1609,2048)     invalid dummy IDs
```

EOS target ID 0 is inserted as one valid input token at position
`1097+len(X[n,j])`, which ranges from 1097 through 1609; that position replaces
the first dummy after `X`. It exists only for teacher forcing and receives no
outgoing loss. The apparent overlap at position 1609 when `len(X)=512` is
intentional: `[1097,1609)` contains 512 target IDs and position 1609 contains
EOS. Every row remains below context 2,048.

The base-training attention mask is fixed by regions, not content:

1. valid `G_L` queries attend causal valid `G_L` keys;
2. valid current-state queries in `[582,1094)` attend all `G_L` and causal prior
   valid current-state keys, never `[70,582)`;
3. `G_R` queries attend `G_L`, valid current-state keys, and causal prior `G_R`,
   never `[70,582)`;
4. valid target and teacher-forced EOS queries attend `G_L`, valid current state,
   `G_R`, and causal prior target keys, never `[70,582)`; and
5. invalid queries attend nothing and invalid keys are visible to no query.

This is the exact clean active-epoch surface used after `C1` dispatch: the old
source slot is empty, the current state occupies the second 512-position slot,
and generation begins at position 1097. It is not the two-state boundary replay
surface used to fit the head.

All labels initialize to ignore index `-100`. Let `x=X[n,j]` and `m=len(x)`.
The only supervised predictor positions and labels are:

```text
labels[1096]       = x[0] if m>0 else EOS_ID_0
labels[1097+k]     = x[k+1] for 0 <= k < m-1
labels[1097+m-1]   = EOS_ID_0 when m>0
```

Thus each lane contributes exactly `m+1` full-vocabulary next-token losses: all
target IDs and the model-authored EOS event target. Prompt, dummy, and EOS-input
positions have zero loss. Stage one fixes `a_M=1` and forms only `ell_eff` from
Section 3. If `T_sup` is the exact set of supervised predictor positions in the
full update, its FP32 objective is

```text
L_stage1 = (sum_(t in T_sup) CE(ell_eff_t.float(), labels[t])
            + 1e-4 * sum_(t in T_sup) logsumexp(ell_eff_t.float())^2)
           / |T_sup|.
```

There is no per-lane reweighting, pre-motor CE, or separately evaluated EOS
loss. The trainer evaluates the displayed loss directly at the listed predictor
positions; a library-side second label shift or implicit second model loss is
forbidden. The EOS target, digit targets, event predicate, and greedy decoder
therefore all train and read the same post-motor effective-logit surface.

The motor site is exact. At every valid model position, including zero-loss
prompt positions, `h_t` is the 576-vector after the final Transformer norm and
immediately before the tied unembedding. The base computes `ell_base_t`;
`m_psi(h_t)` is then evaluated without a router and stage-one `a_M=1` creates
`ell_eff_t` exactly as in Section 3. Objective gradients reach the motor only
through `T_sup`; all other positions are outside both CE and z-loss sums. No
parsed digit, carry, site, event, or position indicator enters this path.

One logical episode pack is tensor shape `[5,2048]`, not five 768-token lanes.
One update contains exactly two packs, shape `[2,5,2048]`, flattened without
reordering to `[10,2048]`. Stage one uses one epoch and all 2,048 packs per seed,
for exactly 1,024 updates. Seeds are `2026071814`, `2026071815`, and
`2026071816`; none may be selected or dropped. Full model weights and the carry
motor are trainable under the token-mean full-vocabulary LM loss above. No
boundary-head gradient enters the base or motor.

These are genuine training replicates, not three orderings of one corpus. A
domain-separated counter-based PRNG keyed by the declared seed independently
determines operand instances within each fixed balance stratum, carry-motor
initialization, stage-one pack permutation, stage-two head initialization, and
stage-two minibatch permutation. Training rows must be disjoint across seeds by
canonical row hash. The parent checkpoint, tokenizer, optimizer hyperparameters,
and evaluation boards remain common by design. Any other stochastic source must
be either separately domain-keyed by seed and receipted or disabled.

### 12.2 Stage-two clean boundary-head tensor

Stage two freezes the fitted base and motor. For every one of the 10,240 lanes per
seed it constructs one separate dense `[2048]` replay row:

```text
[0,70)          exact G_L IDs
[70,582)        P[n,j] IDs, then invalid EOS-ID-0 dummies
[582,1094)      X[n,j] IDs, then invalid EOS-ID-0 dummies
[1094,1097)     exact G_R IDs
[1097,2048)     invalid EOS-ID-0 dummies
```

Positions are exactly `0..2047`. Attention is exactly `M_clean`: valid `X`
queries cannot attend valid `P`; `G_R` can attend `G_L`, valid `X`, and prior
`G_R`, but never `P`; invalid positions are invisible. The only extracted feature
is the post-final-norm residual at fixed position 1096. This is `h_probe`.

Stage two extracts exactly 10,240 `h_probe` rows and fits the 1,154-parameter
boundary head by mean two-way CE. It uses AdamW LR
`0.01`, betas `(0.9,0.95)`, epsilon `1e-8`, zero weight decay, batch 512, ten
epochs, exactly 200 updates, no warmup, and gradient clip `1.0`. The head is then
frozen before any autonomous score is opened. No decoded text or syntax-derived
feature enters the head.

A separately frozen shuffled-label head starts from byte-identical initialization
and sees the same residual rows, minibatch indices, optimizer hyperparameters,
update count, and compute. Its numerical parameter updates may differ because its
labels differ. Within each five-lane episode its boundary labels are rotated once:

```text
true:     COMMIT, COMMIT, COMMIT, COMMIT, HALT
shuffled: HALT,   COMMIT, COMMIT, COMMIT, COMMIT
```

The shuffle preserves four `COMMIT` and one `HALT` label per episode while
breaking the intended terminal relation. It may not share fitted parameters with
the true head.

### 12.3 Exact stage-one optimizer partition and update

The stage-one parameter partition is by exact object identity, not by a future
name heuristic. For every block index `l in [0,29]`, Muon owns exactly:

```text
blocks.l.attn.q.weight       [576,576]
blocks.l.attn.k.weight       [192,576]
blocks.l.attn.v.weight       [192,576]
blocks.l.attn.o.weight       [576,576]
blocks.l.mlp.gate.weight     [1536,576]
blocks.l.mlp.up.weight       [1536,576]
blocks.l.mlp.down.weight     [576,1536]
```

That is 210 tensors and exactly `106,168,320` scalar parameters. AdamW owns
exactly the following disjoint identities:

```text
tok.weight, identical by object identity to tied head.weight   18,874,368
blocks.l.n1.w and blocks.l.n2.w for l in [0,29]                    34,560
blocks.l.attn.qn.w and blocks.l.attn.kn.w for l in [0,29]           3,840
norm.w                                                                 576
motor.down.weight, motor.down.bias, motor.up.weight, motor.up.bias   4,634
```

AdamW therefore owns 126 tensors and `18,917,978` scalar parameters. The union
is exactly `125,086,298` unique trainable stage-one parameters. The tied
`head.weight` alias receives no second parameter entry or second update. The
boundary head is absent from both stage-one optimizers. Any missing, duplicated,
renamed, reshaped, frozen, or additional trainable tensor is optimizer-partition
NO-GO rather than an implementation choice.

The parent parameter storage and both optimizer state families are FP32; model
forward runs under BF16 autocast, `ell_eff` is cast to FP32 for the Section 12.1
objective, and there is no gradient scaler or shadow master parameter. At each
update, both optimizers are zeroed, one full effective-logit loss is backpropagated,
and the single global FP32 L2 norm over the union above is computed. A nonfinite
gradient is fatal. If that norm exceeds `1.0`, every gradient in both partitions
is multiplied once by its exact reciprocal; otherwise no clipping multiplication
occurs. Muon steps first and AdamW second. Because their identity sets are
disjoint, neither optimizer may read or mutate the other's parameters or state.

There are exactly `U=1024` updates indexed `u=1..1024`. Both base learning rates
are multiplied by the same frozen scalar:

```text
s(u) = u/50                                                     for 1 <= u <= 50
s(u) = 0.1 + 0.9*0.5*(1 + cos(pi*(u-50)/(1024-50)))            for 51 <= u <= 1024
lr_muon(u) = 0.001  * s(u)
lr_adam(u) = 0.0002 * s(u)
```

Muon uses momentum `0.95`, Nesterov on, five BF16 Newton-Schulz iterations,
coefficients `(3.4445,-4.7750,2.0315)`, normalization epsilon `1e-7`, no weight
decay, and matrix scale `sqrt(max(1,rows/cols))`. Its zero-initialized buffer and
update are exactly those in `train/muon.py`, SHA-256
`863e79aaaaebb681382f0c88078390b5683ab39be79ac7df60f26d1c04b21762`.
AdamW uses betas `(0.9,0.95)`, epsilon `1e-8`, weight decay `0`, bias correction,
and `amsgrad=False`, `maximize=False`, `foreach=False`, `fused=False`,
`capturable=False`, and `differentiable=False`. Its source is PyTorch `2.10.0`
module `torch/optim/adamw.py`, SHA-256
`54299056b7745c162192132bb6028f3387c05ff4203518ff0240058584968312`.

For completeness, let `g_u` be the globally clipped gradient and let all state
start at zero. For each Muon matrix, the exact recurrence before the bound
Newton-Schulz routine is

```text
b_u       = 0.95*b_(u-1) + g_u
g_nes_u   = g_u + 0.95*b_u
p_u       = p_(u-1) - lr_muon(u)*sqrt(max(1,rows/cols))*NS5(g_nes_u).
```

For each AdamW tensor, with elementwise square and square root,

```text
m_u       = 0.9*m_(u-1)  + 0.1*g_u
v_u       = 0.95*v_(u-1) + 0.05*(g_u*g_u)
mhat_u    = m_u/(1-0.9^u)
vhat_u    = v_u/(1-0.95^u)
p_u       = p_(u-1) - lr_adam(u)*mhat_u/(sqrt(vhat_u)+1e-8).
```

Weight decay contributes exactly zero. Every listed identity must have one
finite gradient at every update; a missing gradient is fatal rather than a
silent optimizer skip. The source hash controls operation ordering, casts,
transpose choice, and Newton-Schulz implementation where the equations above do
not encode tensor-level evaluation order.

The model source is `train/model.py`, SHA-256
`45fc0dc46ceb0f91d08e3f671cbe9ef202ea212e72d5bba8b77356c3fb0983d4`.
An executable protocol must byte-match these three sources and reject any
functional, fused, foreach, compiled-optimizer, source-fallback, or scheduler
substitution. BF16 autocast, FP32 objective, TF32 disabled, pack order,
initialization, tokenizer, parent bytes, gradients, clipping coefficient,
optimizer state, and every update receipt must be bound before fitting. No early
stopping, checkpoint choice, or development-driven hyperparameter change is
allowed.

### 12.4 Frozen boundary-head holdout and global board disjointness

Before any stage-one fit, the complete board registry must be materialized and
hash-committed. The boundary-head holdout `H_B` contains exactly 384 episodes:

```text
3 widths x 2 operations x 8 carry/borrow patterns x 8 = 384 episodes.
```

For each width `w`, every episode contributes exactly `w` clean teacher-forced
`COMMIT` candidate rows and one clean terminal `HALT` candidate row, each packed
exactly as in Section 12.2. The immutable denominator is therefore:

```text
width 4: 128 episodes,  640 rows =  512 COMMIT + 128 HALT
width 6: 128 episodes,  896 rows =  768 COMMIT + 128 HALT
width 8: 128 episodes, 1152 rows = 1024 COMMIT + 128 HALT
total:   384 episodes, 2688 rows = 2304 COMMIT + 384 HALT
```

Each `(width,operation)` cell has 64 episodes, `64*w` COMMIT rows, and 64 HALT
rows. `H_B` freezes exact canonical episode bytes, token IDs, masks, positions,
labels, row order, and denominator before fitting. An external custodian builds
and encrypts it, publishes its byte length and SHA-256 commitment, and withholds
its decryption material from builders, trainers, operators, and experiment code
until all three stage-one checkpoints and all six true/shuffled head artifacts
are immutable. Every one of 2,688 rows is scored; missing, duplicate, malformed,
or unreadable rows are failures. Revealing any `H_B` label, residual, prediction,
or aggregate before artifact freeze is custody NO-GO. After reveal, no fit,
hyperparameter change, checkpoint selection, retry, or replacement board is
allowed.

The registry contains the three seed-specific training sets `T_14/T_15/T_16`,
the public 256-case development board `D_256`, the supplied public 12-case board
`D_12`, `H_B`, the hidden 384-case mechanistic board `H_M`, the hidden 768-case
autonomous board `H_A`, and the hidden 432-candidate observational board `H_O`.
Exact disjointness requires zero intersection among all registry entries for
both canonical serialized-example SHA-256 and every tokenized row SHA-256.

Semantic disjointness is separately mandatory. Every valid arithmetic episode
has canonical key

```text
K_episode = (SCERT-DWS-v1, operation, width, canonical_operand_pair),
```

where leading zeros are retained, addition sorts the two width-digit operands
lexicographically to collapse its commutative alias, and subtraction preserves
ordered minuend and subtrahend. Every transition also contributes
`K_transition=(K_episode,step_index,canonical_current_state,
canonical_successor_state)`. A mechanistic intervention contributes the ordered
tuple of operation, width, canonical `P_nom`, `P_cf`, fixed `X`, both frozen
source-implied targets, edit family, and edited index. An observational candidate
contributes the semantic keys of both source histories. Formatting, tokenization,
operand order aliases, or counterfactual presentation cannot create a new key.
Every registry row contributes every applicable `K_episode` and `K_transition`
in addition to its board-specific intervention or history key, so changing key
type cannot hide reuse of an underlying arithmetic episode or transition.
The semantic-key sets of every pair of registry entries, including the three
training seeds, must have empty intersection. `H_M` and `H_A` are therefore both
exactly and semantically disjoint, not merely differently shuffled views.

The external custodian receives immutable canonical manifests for all training
and public boards, generates `H_B/H_M/H_A/H_O` from separately domain-keyed
randomness, rejects collisions before commitment, and publishes a signed
zero-intersection certificate containing board counts and the hashes of every
private key-set commitment without revealing private cases. Any collision,
post-fit resampling, incomplete certificate, or board whose hidden bytes were
read by fitting code is package-level NO-GO.

The two stages teach only one-step clean transitions, EOS candidacy, a
clean-residual boundary classifier, and terminal answer. Autonomous multi-epoch
composition is reserved for evaluation.

## 13. One-dispatch assay and autonomous total-effect factorial

This section freezes two different future experiments and grants no H100
authority. The first is a matched one-dispatch mechanistic assay. The second is
an autonomous total-effect factorial. Their receipts, endpoints, and claims may
not be pooled or substituted for one another.

### 13.1 Matched one-dispatch mechanistic assay

For each assay case and motor level, generation before the first boundary is run
once from the common clean initialization in Section 14. That one shared run
authors exact `(P_0,X_1,E_1,Q_1)`. The pre-classification state is then cloned
byte-for-byte into the mechanistic arms. Each clone independently executes the
same clean classification replay and exactly one forward of its declared head;
no `h_probe`, head logits, or action is copied between arms. The receipts must
show equal `h_probe`, true-head logits, and `D_1` for the core `C0-true` and
`C1-true` clones. Those clones therefore have exactly equal `P_0`, `X_1`, `E_1`,
`Q_1`, `D_1`, weights, motor level, surfaces, positions, validity, and endpoint.
Only the reconstruction mask differs: `M_stale` versus `M_clean`.

Each clone executes one reconstruction, constructs the complete post-dispatch
state, computes the first post-dispatch `ell_base`, motor delta, `ell_eff`, and
effective-argmax token, and stops. It never generates a second boundary.
Equality requirements in the
mechanistic assay apply only to the shared initial dispatch. They are not imposed
on autonomous rollouts.

Four source-content arms use the same cloned `(X_1,E_1,Q_1,D_1)`:

1. **`C1-true`:** true `P_0` IDs with `M_clean`.
2. **`C0-true`:** the same true `P_0` IDs with `M_stale`.
3. **`C0-neutral`:** `M_stale` with every valid source ID replaced by tokenizer
   ID 233, which alone decodes to one ASCII space, while preserving exact source
   validity, length, positions, and all open edges.
4. **`C0-shuffled`:** `M_stale` with valid source IDs in exact reverse order,
   `P_rev[k]=P[L-1-k]`. It preserves source length, validity, token multiset,
   positions, and all open edges while destroying serialized order. A source
   length below two is board-invalid rather than silently left unchanged.

Neutral ID 233 and reversal are committed before model fitting. Neutral IDs are
valid attended tokens, not masked padding. `C0-true`, `C0-neutral`, and
`C0-shuffled` therefore have identical open attention edges and differ only in
source content. This separates structured stale-content mediation from generic
extra-key load or attention dilution.

The hidden assay board has exactly 384 paired source interventions:

```text
3 widths x 2 operations x 8 carry/borrow patterns x 8 = 384.
```

Every case provides same-length `P_nom` and `P_cf`, a fixed `X_1`, and two
different source-implied next-token targets frozen by the offline builder before
model calls. The true source-directed switch endpoint passes only if
`C0-true(P_nom)` emits the nominal source-implied target,
`C0-true(P_cf)` emits the counterfactual source-implied target, the targets
differ, and the outputs switch in that direction. The same endpoint is computed
after applying the neutral and shuffled content transforms. Any content transform
selected from model output is forbidden.

### 13.2 Autonomous total-effect factorial

Each frozen seed produces one base/carry checkpoint plus true and shuffled frozen
head artifacts before scores open. The autonomous core is:

| Factor | Level 0 | Level 1 |
|---|---|---|
| Reconstruction policy | `C0`: use `M_stale` at each arm-authored dispatch | `C1`: use `M_clean` at each arm-authored dispatch |
| Carry-motor actuation | `M0`: execute the learned delta and set `a_M=0` | `M1`: set `a_M=1` in the single `ell_eff` construction |

The four cells `C0M0`, `C0M1`, `C1M0`, and `C1M1` start from byte-identical
initial token IDs, initial cache, `Q_0`, true-head artifact, generation cap, and
greedy tie rule. The policy first acts at the first model-authored event. After
that dispatch, arm-specific caches may produce different tokens. Therefore later
`P_e`, `X_e`, `E_e`, `Q_e`, `h_probe`, true-head actions, event counts, stop
times, and replay contents are explicitly permitted to differ. The factorial
estimates the autonomous total effect of repeatedly applying the reconstruction
policy, including all downstream token/action mediation. It is not a matched
per-dispatch mechanism assay.

Within each arm and at each arm-authored event, the event machine remains fixed:
clean classification, exactly one boundary-head forward, one reconstruction
under that arm's policy, and atomic dispatch. `C0` advances from its own replay
endpoint without appending candidate EOS. No arm may borrow another arm's tokens,
events, action, `Q_e`, or stop.

Motor-off is an acute runtime ablation of the same learned parameters. Because
stage one co-trains the base and motor, `M1-M0` can identify only acute actuation
within a motor-conditioned checkpoint, not the effect of adding or training the
motor. A training-time motor/no-motor claim requires another preregistration.

### 13.3 Head and context controls

Every event in every arm executes exactly one affine `576->2` head forward:

1. treatment and primary-factorial arms load only the true head and use its one
   forward on `h_probe`;
2. the shuffled-head control loads only the shuffled head and uses its one
   forward on `h_probe`; it does not execute the true head;
3. fixed `K4/K6/K8`, always-HALT, always-COMMIT, target-switch fixed-action,
   oracle-syntax, and oracle-width/fresh-state controls load and execute only the
   true head once, discard its action, and apply their declared fixed or oracle
   action; and
4. the raw-residual diagnostic is a separate matched arm that executes the true
   head once on `h_raw` instead of `h_probe`; it never executes both.

Thus every arm loads one equal-sized head artifact and performs exactly one head
forward per event. The true and shuffled learned-head policies are never resident
or executed together. Fixed policies run the complete mixed board and may not be
selected by case width. Oracle controls are non-autonomous ceilings and cannot
feed, select, or rescue a scored arm.

The post-hoc mask-only negative uses the arm's one declared head action, drops
direct stale keys after they have already contextualized `X`, and executes dummy
position-matched reconstruction for compute matching. The compact clean
diagnostic uses compact reset positions and is explicitly compound. Fresh host
prompting injects oracle states and receives no autonomy credit.

For physical matching, every case in every arm allocates eighteen dense
2,048-position replay tensors, one classification and one reconstruction for each
of nine possible events, plus nine 512-position generation budgets. Unused and
post-stop work is masked dummy compute. Unpadding, early kernel exit, and variable
batch shape are forbidden. Forward counts must match exactly and measured FLOPs
within 1%, or the higher-compute control is labeled favorable. Autonomous arms
need not have equal semantic tokens after their first divergent dispatch.

### 13.4 Hidden finite boards and replicate rule

The hidden autonomous board has exactly 768 episodes:

```text
3 widths x 2 operations x 8 carry/borrow patterns x 16 = 768.
```

`H_M` and `H_A` are generated, collision-checked, and hash-committed by the
independent custodian under the exact and semantic disjointness contract in
Section 12.4. They remain hidden until the Section 16 reveal gate. There is no
randomized arm assignment and no assumed exchangeability, so this protocol makes
no sign-flip, McNemar, p-value, confidence-interval, or population-frequency
claim. It uses exact finite-board thresholds only.

The three seeds remain independent training replicates. For each endpoint and
contrast, all three per-seed numerators over the fixed denominator, their minimum,
median, and maximum are reported. Every directional and magnitude gate must pass
in `3/3`; pooling `3N` rows or averaging away a failed seed is forbidden. Public
development and supplied 12-case results cannot satisfy a hidden-board gate.

The future factorial requires one visible `NVIDIA H100 PCIe`, batch-one greedy
evaluation, BF16 model and effective logits, TF32 off, lowest-token-ID tie
break, and complete token, `E_e`, `Q_e`, residual, head, surface, validity, mask,
position, cache, `ell_base`, motor-delta, `ell_eff`, stop, and resource receipts.
These requirements do not
authorize hardware. No partial score may open before every declared arm and all
three seeds are immutable.

## 14. Evaluation and causal interventions

### 14.1 Autonomous total-effect evaluation

Every autonomous arm starts from one byte-identical clean initialization tensor:

```text
[0,70)          exact G_L
[70,582)        invalid EOS-ID-0 dummies
[582,1094)      exact S0 IDs, then invalid dummies
[1094,1097)     exact G_R
[1097,2048)     invalid dummies; generation begins at 1097
```

It uses the stage-one clean active-epoch mask, `Q_0=(ACTIVE,0,0,0,0,0,0,
fixed_caps,False,fixed_rng_start,fixed_tie_state,0)`, and the same initial cache.
The arm then runs only its declared event machine until its one loaded head
accepts a model-selected EOS as `HALT` or a fixed cap fails. No parser,
arithmetic, semantic stop, gold token, retry, or host boundary is available.

After the first dispatch, autonomous arms may and generally will have different
tokens, events, runtime state, and actions. Scoring never forces them back onto a
matched trace. Full-trace exactness requires all ordered states, the expected
number of model-authored `COMMIT` actions, exact answer, and final model-authored
`HALT`. Missing, malformed, extra, premature-HALT, missed-HALT, or post-terminal
spans are failures.

The public development board has 256 width-4 episodes balanced over operation and
carry/borrow pattern. The supplied 12-case width-4/6/8 board remains a public
cross-width diagnostic. The hidden 768-case autonomous board in Section 13.4 is
the only confirmatory autonomous denominator. Width, operation, carry stratum,
longest exact prefix, event and commit counts, first failure, state exactness,
answer exactness, full-trace exactness, action-sequence exactness,
premature-HALT, and missed-HALT are all reported per seed and arm.

### 14.2 Universal `do(P)` one-dispatch assay

The public 12 cases retain three source interventions each:

```text
E_c:  flip the serialized source carry/borrow bit
E_r:  add one modulo 10 to source written result digit r[0]
E_o:  in order a then b, +1 then -1 modulo 10, use the first active-operand
      edit that changes the frozen source-implied next target
```

These 36 public pairs and the hidden 384-case assay are structural interventions,
not observational matches. They hold exact `(X_1,E_1,D_1,Q_1,I)` fixed and set
the source slot by `do(P_0=a)` or `do(P_0=b)`, whether or not either joint tuple
occurs naturally. They stop after first post-dispatch `ell_base`, motor delta,
`ell_eff`, and effective-argmax token and cannot feed an autonomous trace.

For `C1`, every source intervention, including true, counterfactual, neutral, and
shuffled content, must yield bit-identical `h_probe`, head logits/action, retained
layerwise K/V, post-dispatch `Q_2`, next `ell_base`, motor delta, `ell_eff`, and
effective-argmax token. Each cache
must equal an independent reference encoding of the exact `R_pm`, validity,
positions, `M_clean`, and `Keep` projection. This tests the universal structural
property; no common-support filter is applied.

For `C0`, true, neutral, and shuffled content use identical `M_stale` open edges.
The scorer reports any-content change, source-directed paired switch, nominal and
counterfactual target exactness, and effective-logit-margin movement toward the frozen
source-implied target. Directional true-content response, not arbitrary output
difference, is required to establish semantic stale mediation rather than generic
extra-key dilution.

### 14.3 Observational common-support audit

The separate 432-candidate hidden audit in Section 6.2 uses naturally generated
history pairs. Admission is determined solely from pre-dispatch receipts. Every
admitted `C1` pair is decoded from the matched dispatch with common future random
bits through `HALT` or cap; complete post-dispatch state and future receipts must
be exact. The 216-overall and 12-per-stratum minimum is noncompensatory. Structural
`do(P)` results may not fill an observational denominator.

### 14.4 Latest-state target-switch interventions

This is a labeled fixed-action structural reconstruction diagnostic, not an
autonomous arm. It is applied to every `D_12` episode for public sanity and every
`H_A` episode for the fixed hidden denominator. Before fitting, the board
custodian freezes for each episode exact `P_0`, canonical same-length token spans
`X_nom`, `X_carry`, and `X_result`, the one-token edited index and replacement ID
for each intervention, and canonical full next-span targets `Y_nom`, `Y_carry`,
and `Y_result`. `X_carry` differs from `X_nom` only by the serialized carry or
borrow bit. `X_result` differs only by `r[0]` modulo 10. All three spans must have
identical token count, positions, validity, and scaffold surface. A missing,
multi-token, length-changing, or ambiguous edit makes the fixed case fail; the
runtime may not search for another edit.

The arm is exactly `TS-C1M1`: reconstruction uses `M_clean`, the carry motor is
fixed on with `a_M=1` at every vocabulary decision, and the frozen true boundary
head is the only loaded head. The nominal arm teacher-forces `X_nom` on the clean
active-epoch surface and evaluates the next-token `ell_eff` after its last token.
That one effective argmax must create nominal event `E_1`; otherwise all
target-switch endpoints for the case fail. The runtime freezes the complete
nominal `Q_1`, executes the nominal arm's one clean classifier and one true-head
forward, and records nominal `D_1`. `D_1` must be `COMMIT`; otherwise the case
fails. Exact `(P_0,E_1,D_1,Q_1,I)`, weights, motor level, caps, RNG state, and tie
rule are then cloned into the counterfactual arm. Only `do(X_1=X_carry)` or
`do(X_1=X_result)` differs from the nominal arm.

Three controls are separately frozen and use the same board rows, nominal
`(E_1,D_1,Q_1)`, fixed-action rule, head-forward count, generation endpoint, and
failure accounting. `TS-C0M1` changes only reconstruction to `M_stale`;
`TS-C1M0` changes only to `a_M=0` while still computing and discarding the motor
delta; and `TS-post-hoc-M1` uses the Section 13.3 post-hoc mask-only cache. No
other target-switch reconstruction policy, motor level, or control may be added
after fitting.

Each counterfactual arm independently executes its own clean classification
replay and exactly one true-head forward. Its effective EOS receipt must still
equal cloned `E_1`, and its observed head action is recorded. For causal
isolation, both arms discard the observed action after that one forward and
dispatch the cloned nominal `D_1=COMMIT`, exactly like a fixed-action control in
Section 13.3. If a counterfactual observed head action differs from `D_1`, the
`boundary_action_changed` count increments and that pair fails every paired
target-switch success endpoint. A boundary-action change is never credited as
an output switch, target switch, or autonomous success.

Each arm performs its one declared reconstruction or post-hoc operation, installs
its resulting retained cache at the common final-`G_R` endpoint without the
candidate EOS, and greedily emits from its declared `a_M` effective-logit surface
until the next effective EOS argmax or the 512-token cap. The scored output
`Y_hat` is the exact non-EOS token sequence; the diagnostic stops before
classifying or dispatching that next EOS, so there is no second boundary action.
A paired carry target-switch passes only when:

1. both initial effective EOS receipts equal cloned `E_1` and both observed head
   actions equal cloned `D_1=COMMIT`;
2. `Y_hat_nom=Y_nom` and `Y_hat_carry=Y_carry` exactly through the next EOS;
3. `Y_nom != Y_carry`; and
4. `Y_hat_nom != Y_hat_carry` in that frozen target direction.

Full-target exactness, raw output difference, boundary-action change, and paired
target switch are separate counts. Counterfactual exactness alone is invalid.
The written-result arm uses the same frozen `TS-C1M1` procedure and is
corroboration only; it cannot rescue failed carry switch. Because `E_1`, `D_1`,
and `Q_1` are nominal structural interventions, none of these arms receives
autonomy credit or enters the autonomous total-effect numerator.

### 14.5 Clean-direct equality and output preservation

At every actual clean head-authored `COMMIT`, a read-only independent audit call
encodes the exact same `R_pm`, validity bits, `p_pm`, `M_clean`, dtype, and
weights, then applies the same `Keep` projection. Every retained layer K/V, next
`ell_base`, motor delta, `ell_eff`, fixed final-`G_R` `h_probe`, boundary-head
logits, boundary action, and post-dispatch `Q_(e+1)` must be bit-identical to
those used by `C1`. The audit result cannot alter decoding.

A frozen 128-prompt non-DWS preservation set is decoded for 128 greedy tokens
from both parent and fitted checkpoint with SCERT mode disabled. Token IDs must
be exactly equal, with zero SCERT boundary dispatches and zero motor calls.
Within the fitted checkpoint, a frozen teacher-forced set of 512 non-carry DWS
prefixes is scored with the motor off and on. Every `ell_eff` coordinate except
`v0` and `v1` must be bit-identical, and effective-argmax token IDs must be
unchanged. Carry positions are
identified only by the offline scorer after both calls. The boundary head may
never alter `ell_base`, the motor delta, `ell_eff`, or create an EOS candidate.
Any failure is a package-level preservation veto.

## 15. Decision gates

All denominators include malformed, missing, capped, and non-EOS calls as
failures unless a subsection explicitly defines pre-output observational
admission. Every seed, width, operation, and frozen stratum is reported; seed or
case exclusion is forbidden. All thresholds are finite-board requirements with
no p-value interpretation.

### 15.1 Mechanics, separation, and autonomy vetoes

- every CPU mechanics item passes;
- the one-dispatch assay validator proves the core clones share exact
  `(P,X,E,Q,D,I)` at their sole dispatch and stop after one next-token receipt;
- the autonomous validator proves equality is required only at initialization,
  never forces later tokens/events/state/actions to match, and never transfers a
  receipt between arms;
- every `C1` structural source intervention and clean-direct audit is bit-exact;
- event detection, emitted token, motor delta, every vocabulary endpoint, and
  stage-one CE/z-loss all use the one `ell_eff` surface with no second argmax;
- the exact/semantic board-disjointness certificate, 2,688-row `H_B` custody,
  and fixed reveal order in Section 12.4 pass before any fitted score opens;
- the 210/126 optimizer identity partition, source hashes, state, clipping,
  schedule, and 1,024 update receipts match Section 12.3 exactly;
- no online forbidden resource is called;
- exactly one `576->2` head forward occurs per event in every arm under the
  artifact/action rules of Section 13.3;
- the 128-prompt preservation gate passes exactly.

### 15.2 Structural and observational non-interference gates

For each seed, `C1` must be exact on all 384 hidden structural assay cases under
true, counterfactual, neutral, and shuffled source contents: `384/384` equal
`h_probe`, head logits/action, retained K/V, post-dispatch state, next
`ell_base`, motor delta, `ell_eff`, and effective-argmax token. One mismatch is
structural theorem implementation NO-GO.

The observational audit must admit at least `216/432` pairs overall and at least
`12/24` in each of its 18 strata. Every admitted pair must have exact complete
post-dispatch state and future receipts through halt or cap. A smaller denominator
is `OBSERVATIONAL DENOMINATOR NO-GO`; a mismatch is observational independence
NO-GO. Neither outcome invalidates a separately passing universal `do(P)` result.

### 15.3 Autonomous boundary-policy gate

On the exact 2,688-row `H_B` from Section 12.4, in every seed the true head must
be correct on at least `2554/2688` rows overall. Its noncompensatory
`(width,operation)` minima are `288/320` for each width-4 cell, `404/448` for
each width-6 cell, and `519/576` for each width-8 cell. It must also be correct
on at least `2074/2304` `COMMIT` rows and `346/384` `HALT` rows. Relative to the
same seed's shuffled-label head on the identical rows, it must gain at least
`404/2688` correct rows overall, `231/2304` within `COMMIT`, and `39/384` within
`HALT`. These fixed integer thresholds replace rounded percentages; no row can
be admitted, excluded, reweighted, or substituted after reveal.

On the hidden autonomous board, `C1M1` exact complete boundary-action sequence
must be at least `615/768` overall and `205/256` within each width, in all three
seeds. It must exceed the shuffled-head sequence count and each single fixed
`K4/K6/K8` count by at least `154/768` overall and `52/256` per width. No fixed
policy may be selected by case width. Always-HALT and always-COMMIT must have zero
complete sequence credit.

Oracle-syntax and oracle-width/fresh-state results are reported as host-scheduled
ceilings only. They cannot satisfy, replace, or compensate for this gate. Any
scored use of parsed `z`, parsed `answer=`, expected width, or a host-selected
boundary is automatic autonomy NO-GO.

### 15.4 One-dispatch semantic stale-content gate

For `C0-true`, source-directed paired switch must be at least `192/384` overall
and `64/128` within each width, in all three seeds. Its directional count must
exceed both `C0-neutral` and `C0-shuffled` by at least `96/384` overall and
`32/128` per width. True, neutral, and shuffled arms must have identical source
validity and identical open reconstruction-mask edges in every case.

This gate is noncompensatory. If `C0` changes under source content but does not
move toward the frozen source-implied target, the result is generic content or
extra-key sensitivity, not semantic stale mediation. If the directional gaps to
neutral and shuffled controls fail, an autonomous `C1-C0` difference may be
reported only as a reconstruction-policy total effect; it may not be attributed
to removal of semantic stale-source mediation.

### 15.5 Autonomous reconstruction-policy total-effect gate

On the hidden 768-case board, each of these full-trace count differences must be
at least `77/768` overall and `13/256` within each width, in all three seeds:

```text
C1M1 - C0M1
C1M0 - C0M0
C1M1 - post-hoc-mask-only-M1
```

Longest exact prefix and public-board movement are reported but cannot substitute
for these exact hidden finite thresholds. Passing establishes an autonomous total
effect of the reconstruction policy. Semantic stale-mediation attribution also
requires Section 15.4.

### 15.6 Noncompensatory carry veto

On the public supplied 12 cases, the following remain required development sanity
checks but cannot satisfy the hidden gate:

- paired carry target-switch at least `9/12` and at least `3/4` per width;
- counterfactual full-target exactness at least `9/12` and `3/4` per width;
- output switch at least `11/12` and `4/4` per width;
- a paired-switch gain of `TS-C1M1` over `TS-C0M1` of at least 40 percentage
  points overall and at least `2/4` within every width.

On the fixed 768-case `H_A` target-switch overlay, `TS-C1M1` paired carry switch
must be at least `384/768`, exceed both `TS-C0M1` and `TS-post-hoc-M1` by at
least `77/768`, and be at least `154/384` separately for nominal `c=0` and
`c=1`, in all three seeds. Every boundary-action change is already a failed
pair under Section 14.4 and is additionally reported by arm and stratum. Failure
of any carry gate vetoes a carry-use or integrated-mechanism claim regardless of
nominal exactness, answer score, EOS, written-result response, or fresh-host
ceiling. These fixed-action diagnostic counts never receive autonomous credit.

### 15.7 Acute motor-actuation and interaction gate

An acute motor-actuation contribution within the jointly motor-trained checkpoint
may be claimed only if autonomous `C1M1-C1M0` is at least `39/768` in
full-trace count and fixed-action `TS-C1M1-TS-C1M0` is at least `77/768` in
paired carry-switch count, with no negative full-trace difference in any width,
in all three seeds. This does not support a motor-training or
added-parameter claim. A positive acute-actuation-by-reconstruction interaction
requires, per seed,

```text
(N_C1M1 - N_C0M1) - (N_C1M0 - N_C0M0) >= 39 of 768,
```

with a nonnegative interaction count within every width. If reconstruction passes
but these motor gates fail, the result is reconstruction-policy GO and acute
motor actuation NO-GO. If only `C1M1` passes, the allowed conclusion is a joint
package signal with no component attribution. Fresh-host performance cannot
compensate for any autonomous gate.

## 16. Non-executable custody and determinism checklist

This is a requirements checklist, not evidence that any requirement has been
met. Every item is currently unchecked. No CPU capability claim, model fit,
factorial, confirmation call, or H100 job is authorized by this document.

- [ ] **Reviewed clean source:** one clean Git commit binds the exact bytes of
  this preregistration, event runtime, evaluator, trainer, data builder,
  `model.py`, mask/position code, scorer, semantic validator, job wrapper, tests,
  lockfiles, and independent reference implementation. A manifest lists path,
  mode, byte length, and SHA-256; before/after source-tree hashes must match.
- [ ] **Artifact identities:** parent checkpoint, all fitted checkpoints,
  tokenizer, scaffold IDs, `G_L/G_R`, dummy/neutral IDs, `v0/v1`, training,
  development, boundary-head holdout, mechanistic, autonomous, and observational
  boards, every semantic-key-set commitment, confirmation commitment, heads,
  motors, and optimizer states have externally recorded byte lengths and SHA-256
  receipts.
- [ ] **External board custody and disjointness:** the complete Section 12.4
  registry, exact 2,688-row `H_B` composition, encrypted board bytes, domain
  keys, and exact/semantic zero-intersection certificate are committed before
  fitting. The custodian is independent of the runtime process; held-out bytes
  remain unreadable to builders, trainers, operators, and experiment code until
  their declared reveal points. Self-authored, post-fit regenerated,
  collision-bearing, or self-rehashable commitments are invalid.
- [ ] **Optimizer identity:** the 210-tensor Muon and 126-tensor AdamW identity
  sets, tied-parameter deduplication, scalar counts, FP32 storage/state, update
  order, global clip, schedule, hyperparameters, and all three source hashes in
  Section 12.3 match an independent manifest exactly. One tensor in both or
  neither set, an alias updated twice, or any source fallback is fatal.
- [ ] **Deterministic environment:** exact node/GPU model and UUID, driver, CUDA,
  cuDNN, NCCL, PyTorch, compiler, container or environment-lock hash, locale,
  thread counts, and all non-secret environment switches are receipted.
  `torch.use_deterministic_algorithms(True)` is enforced; TF32, dropout, cuDNN
  benchmarking, stochastic data order, and nondeterministic kernels are disabled
  unless an exact deterministic substitute is bound.
- [ ] **Attention backend pin:** SDPA implementation is explicitly pinned. Flash
  and memory-efficient kernels are disabled unless exact same-node repeated-run
  bit identity and an independent reference comparison pass. Matmul precision,
  BF16 casts, softmax path, tie rule, and `CUBLAS_WORKSPACE_CONFIG` are fixed.
- [ ] **Complete runtime state:** a schema enumerates every `Q_e` field and a
  dependency audit proves no omitted counter, phase, RNG cursor, position offset,
  receipt cursor, cap, pointer, or stop flag can influence future execution.
- [ ] **Tensor and semantic validator:** before scores open, an independent
  validator reconstructs every stage-one `[2,5,2048]` update and stage-two
  `[2048]` replay; checks IDs, fixed positions, validity, attention masks,
  shifted labels, ignore masks, supervised-token denominator, residual/motor
  site, `ell_base`, motor delta, `ell_eff`, effective event/emission argmax,
  effective CE and z-loss, optimizer partition, and optimizer update count; and
  checks every evaluation arm/mode/case/seed count. It proves exact clone
  equality only in the one-dispatch assay, permits
  arm-local downstream divergence in autonomous runs, validates structural versus
  observational denominators separately, checks true/neutral/shuffled source
  content with identical open edges, counts exactly one declared head forward per
  event, validates the frozen `TS-C1M1` and control arms including cloned
  `(E,D,Q)` and boundary-action-change failures, and validates `E`, `D`, `Q`,
  stop/cap, `Keep`, re-encoding, and motor receipts. Any second logit surface or
  missing/extra receipt is fatal.
- [ ] **Mechanics/reference agreement:** all Section 11 CPU gates pass against an
  independently written implementation. A bounded H100 preflight then reproduces
  exact event traces and agrees on all semantically discrete outputs before any
  capability board can run.
- [ ] **Crash-atomic publication:** every artifact is written to a same-filesystem
  temporary path, file-fsynced, atomically renamed, directory-fsynced, reopened,
  schema-validated, and hash-verified before immutable permissions are applied.
  Partial files, mutable final paths, hard-link aliases, and overwrite are fatal.
- [ ] **Score blindness:** per-arm outputs, logs, and summaries remain sealed until
  every arm and all three seeds finish and pass integrity checks. No partial
  metric, transcript, checkpoint choice, threshold, retry, or exclusion is
  available to the operator or subsequent jobs.
- [ ] **Independent hostile review:** a reviewer who did not author the runtime
  receives the exact manifest and returns written GO on theorem alignment,
  autonomy, determinism, resource matching, statistics, custody, and final
  validator bytes. A source change invalidates that GO.
- [ ] **Explicit hardware authorization:** only after all preceding receipts are
  immutable may a separate authorization name the exact dry-run or H100 job,
  account, partition, resource request, output root, and allowed board. Absence of
  that authorization is unconditional GPU NO-GO.

Secrets and environment values capable of authenticating services are never
printed, committed, copied into manifests, or written to logs. This checklist
must be instantiated in a later executable protocol; prose assertions in this
theory draft do not satisfy it.

## 17. Claim boundaries and next decision

Even a complete pass would establish only bounded DWS token-state retirement
under this explicit event-driven runtime. It would not establish:

- semantic canonicalization, because the runtime copies IDs without knowing
  whether they are a state;
- a proven Nerode quotient or sample-complexity theorem, because sufficiency of
  the latest-state string is empirical and representation equality alone gives
  no learning bound;
- arithmetic discovery, state repair, planning, schedule learning, source
  compilation, broad language reasoning, SoTA performance, or general autonomy;
- reliable state authorship outside tested widths, values, operations, syntax,
  commit count, precision, or context length;
- ordinary Transformer recurrence, because replay and destructive replacement
  are essential resources;
- a new memory ontology, computational primitive, or separation from internal
  re-prompting, RNNs, FSTs, recurrent Transformers, or bounded unrolling;
- a hidden-set or promotion claim, because this document creates no secret
  confirmation board and authorizes no run.

Nor would the current motor factorial establish that training with or adding the
motor helped. It can establish only acute inference-time actuation within the
motor-conditioned checkpoints. A matched training-time motor/no-motor factorial
would be a new protocol.

The learned head can establish bounded autonomous boundary selection only if its
frozen controls pass. EOS by itself never establishes model-authored retirement:
without the head, distinguishing microstep reset from final halt requires a
fixed schedule or host interpretation of state/answer syntax, both external
resources.

The universal structural theorem can pass while the observational denominator
fails or autonomous behavior remains wrong. That means the dispatch implementation
blocks the declared source path, but says nothing by itself about naturally
matched histories or authored-state sufficiency. Observational independence can
be claimed only after its denominator and exactness gates. Autonomous improvement
without the one-dispatch semantic-content gate is only a reconstruction-policy
total effect, not evidence of semantic stale-source mediation.

The only current decision is `NO EXECUTION AUTHORITY` and `H100 NO-GO`. A later
executable version may seek authorization only after every Section 16 item is
instantiated and independently accepted.
