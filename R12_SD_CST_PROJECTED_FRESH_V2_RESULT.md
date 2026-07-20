# R12 Projected SD-CST Fresh v2 Result

**Decision:** rejected on the sole fresh development read; confirmation remains
sealed and must not be opened

**Scientific source:** `6ca8933c2cfcc2d972733774b26ced9a9b75caef`

**Board/training seeds:** `126281723562431289` / `2943136710636342416`

**Sole score-bearing job:** Slurm `694028` on `evc27`, completed cleanly in
8m18s

## Contract and custody

V2 preserves the 146,057,595-parameter v1 system and changes only the
model-logit discretization of the eight event-kind slots. It computes exact
maximum a posteriori assignment under the public exactly-one-STOP grammar,
exports the raw float32 `8 x 3` logits, and lets the independent assessor
recompute and verify the complete assignment. Only the resulting 25+1
categorical bytes reach the separate source-deleted executor.

The final board was generated only after source commit and after excluding all
operation sequences from both inherited parent training and the consumed v1
development split. It contains 48,000 training, 2,304 development, and 2,304
sealed-confirmation rows. The sealed confirmation file remains mode `0600` and
unopened. Board hashes are:

| Artifact | SHA-256 |
|---|---|
| board report | `050794b34afa949f2d2b2942ef3c0717ac418320a1b2becca7c31ca437d745bb` |
| training | `b7756dbf8d4401dbc5fb897dee53f68758e27200b1ce0d2387631f2f0205ec25` |
| development | `0e0720030f4b5739b7de7320fb45f5817e1e8fadb3f7f12e62b98e2f41593191` |
| sealed confirmation | `c477718c91b22abcfd9dec41f1bb3876294ebc36bfe4c5538a4175ad64d69a07` |

The cross-generation audit finds zero prior consumed-development prompt, name,
or operation-sequence overlap in every new split; zero prior 13-gram overlap in
new training and sealed confirmation; and 34 disclosed renderer-grammar
13-grams in new development. Confirmation access remains zero.

## Training result

The treatment learns the complete consumed training interface:

- exact whole tape, initial state, kind, identity, amount, and query:
  48,000/48,000;
- event pointer: 47,985/48,000; and
- exact STOP grammar on every treatment training row.

The equal-update independently row-shuffled-label arm remains near chance:
934/48,000 exact whole tapes, 1,361/48,000 identities, and 100/48,000 binding
pointers. This is strong evidence that the projected treatment learned the
fresh training binding function rather than inheriting or merely relabeling it.

## Development result

| Metric | Treatment | Row-shuffled labels | Consumed projected |
|---|---:|---:|---:|
| Exact structured packet | 672/2,304 = **29.167%** | 13/2,304 = 0.564% | 672/2,304 = 29.167% |
| Exact recurrent state | 2,055/2,304 = **89.193%** | 416/2,304 = 18.056% | 2,061/2,304 = 89.453% |
| Exact answer | 763/2,304 = **33.116%** | 744/2,304 = 32.292% | 783/2,304 = 33.984% |
| Exact joint | 684/2,304 = **29.688%** | 134/2,304 = 5.816% | 689/2,304 = 29.905% |

All 672 exact treatment packets execute to the exact final state and answer:
**100% conditional execution**. Every frozen packet intervention matches its
independently changed oracle, every observed STOP bucket is exact, source
deletion passes, and negative controls remain below their ceilings. The
recurrent motor, reader, state trajectory, halt semantics, and source-deletion
boundary are therefore not the observed bottleneck.

Treatment compiler fields localize the failure:

| Field | Exact |
|---|---:|
| Initial state | 2,176/2,304 = 94.444% |
| Event kind after structured decode | 2,017/2,304 = 87.543% |
| Entity identity | 2,015/2,304 = 87.457% |
| Amount | 2,020/2,304 = 87.674% |
| **Late query** | **768/2,304 = 33.333%** |

Raw independent kind argmax already has exactly one STOP on 2,206/2,304 =
95.747% and exact complete kinds on 2,016/2,304 = 87.500%. Exact MAP adds one
exact-kind row and guarantees legal construction, but it does not repair
grounding.

## Renderer and depth decomposition

Seven non-paraphrase variants each reach 288/288 exact recurrent state, while
the held-out paraphrase renderer reaches only 39/288 = **13.542%** state and
0/288 exact packets. All non-paraphrase variants are nevertheless stuck at
96/288 = 33.333% packets and answers because the late query is at chance.

State remains between 88.281% and 90.365% at every depth one through six. Packet
accuracy instead follows the query-position alias: depths one/four are 62.5%,
depths three/six are 25%, and depths two/five are 0%. This periodic structure,
together with exact 33.333% query accuracy, is a direct label-position shortcut,
not an execution-depth failure.

Pointers confirm the renderer failure. Source-line localization is 100% on all
seven non-paraphrase variants and 0% on paraphrase; initial-entity localization
is 100% on those variants and 14.236% on paraphrase; event-entity localization
is 0% on paraphrase. The treatment's binding path is causally useful relative
to shuffled supervision, but the common frozen source/query front end caps both
treatment and consumed-projected arms.

## Decision and next hypothesis

The frozen assessor records `reject_projected_fresh_board`; confirmation is not
authorized. V2 establishes a bounded causal decomposition:

1. the model can learn the projected exact-surface binding function on fresh
   training rows;
2. a correct private categorical packet is sufficient and is consumed exactly;
3. renderer-invariant source grounding does not transfer to the held-out
   paraphrase family; and
4. the frozen late-query compiler is exactly at three-way chance.

Do not add executor width, epochs, a different STOP decoder, or a same-board
rescore. The admissible successor is a fresh-board trainable source/query front
end with renderer-orbit supervision and a content-addressed query-to-entity
binding interface. The existing projected executor remains fixed. User
authority permits the complete deployed system to grow only while remaining
strictly below 200,000,000 parameters; parameter increases must be charged to
this isolated front-end hypothesis and matched by equal-budget controls.

## Preserved artifacts

| Artifact | SHA-256 |
|---|---|
| checkpoint | `1d338651e381c6bd36982adca0e0edf36147c54c101ebc63e37ffea431a645fd` |
| gate configuration | `3bac8380892a2c352234a0144f5247376bfe2ff88c395ba111cbb31695617cb2` |
| development evaluation | `9bc40bf93591c02decc41bfe4ce5feeb00a4cfd2d94eaf9373407fa1be8b8d92` |
| packet tensor | `531aab65b101397dab5240a3409eb95afa7c38059a21558d49aa971c259c7e30` |
| executor tensor | `0a823e462d05b2c51422fcda3680d646fe6ccc7a5f00b16ada6e801be37f58bb` |
| assessment | `4c45970899ca65a78e833d48a4c8220231117560b8ae6443a0c139174f74ea4e` |
| development access ledger | `3e0b328b3699edb56b67b628e3d38cc3d6c47d2fa76a576310a654585e719e7a` |

Local mirrors preserve all seven files with these exact hashes. Development and
confirmation custody is `1/0`.
