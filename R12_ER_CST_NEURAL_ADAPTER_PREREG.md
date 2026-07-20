# R12 ER-CST Neural Adapter Preregistration

**Protocol:** `R12-ER-CST-v1-neural-adapter`

**Status:** architecture admitted before scientific source freeze. No board seed,
training seed, H100 job, development read, confirmation read, or neural result
exists.

## 1. Question

Can Shohin infer three episode-local `S_3` permutation laws from determining
before/after witnesses, bind fresh opaque operation names to those laws, delete
the source, and compose the resulting categorical cards with one tied recurrent
motor?

This is the first post-confirmation test in which operation meaning changes in
every problem. It is still a bounded finite-law test, not a claim of arbitrary
program induction or general reasoning.

## 2. Exact inherited parent

The only permitted parent is the independently confirmed SD-CST Complete
Physical Fresh v1.3 treatment:

| Receipt | SHA-256 |
|---|---|
| confirmed checkpoint | `a5888d88541904cfa186a6686012c13c7b555f7d186ba1e3e73f71dbaca462d8` |
| confirmation assessment | `4629a745f6eed2e388eb6e1f78b29dff346ee6939e21275ae6ff1d66719d3cb9` |
| reconstructed parent state | `cfb3d8bdf712bd0ed51e35c015b8a106b4b48b6112418585fc1df1139c3b49d9` |

Initialization must reconstruct all four parent stages, load the confirmed
treatment state, and copy every inherited tensor byte-identically into the
ER-CST subclass. A missing or changed parent tensor is fatal.

## 3. Frozen architecture

The compiler consumes exactly twelve newline-delimited physical records:

1. one declaration/initial-state record;
2. three rule-witness records; and
3. eight event records, including a persistent HALT suffix.

It reuses the confirmed local byte/position embeddings, four-layer line encoder,
two-layer record-set encoder, and nonlinear six-occurrence pointer head. It adds
exactly thirteen tensors:

- a twelve-role physical-record head and role embeddings;
- independent rule and event normalizations;
- a six-class permutation-card head;
- a two-class HALT head; and
- one event-query and one rule-key projection for opcode-to-card binding.

The model emits only categorical initial-state, three card, eight card-reference,
and eight HALT logits plus source-facing pointer logits. After hard packet sealing,
the source, token memory, record representations, and compiler residuals are
destroyed. The executor receives only categorical state/card/reference/HALT
tensors and the motor weights.

The tied motor is a `12 -> 128 -> 6` GELU MLP reused at every step. Its complete
domain is all `6 x 6 = 36` state/card pairs, and its certificate target is exact
permutation composition. It replaces the confirmed parent's 19,206-parameter
fixed motor rather than coexisting with it.

## 4. Exact parameter certificate

| Component | Parent | ER-CST |
|---|---:|---:|
| immutable Shohin trunk | 125,081,664 | 125,081,664 |
| compiler | 67,027,474 | 67,336,230 |
| categorical motor | 19,206 | 2,438 |
| categorical reader | 835 | 835 |
| **complete deployed system** | **192,129,179** | **192,421,167** |
| **headroom below 200M** | **7,870,821** | **7,578,833** |

The new compiler contributes 308,756 parameters while the smaller motor removes
16,768, for a net increase of 291,988. Exactly 11,715,616 parameters are
trainable: 11,713,178 compiler parameters and 2,438 motor parameters. The
compiler whitelist contains 98 tensors; the motor contains four. The canonical
name/shape/count contract SHA-256 is
`f2c6c1debd1e17c43c287b2dc72db1185765ba9fa7d340ed56007c65c0a1bc2b`.

The excluded compiler state digest after exact reconstruction is
`1ad33273f7db5073b4cfac0e79031544a74b74914050026642110aee484c94e7`.
It must remain byte-identical for every arm.

## 5. Permitted supervision

Training rows may supervise only:

- twelve physical-record roles and source pointers;
- declaration bindings and initial-state identities;
- the three six-class rule cards;
- eight opcode-to-card references;
- one HALT decision per event slot; and
- the motor's fixed 36-cell categorical composition certificate.

Training may not expose final states, answers, recurrent trajectories, depth as a
feature, development/confirmation cards, executor output, correctness feedback,
retry/search, or any scorer field. Development and confirmation rows may contain
oracles only in files inaccessible to fitting.

## 6. Matched neural arms

The scientific fit must include at least:

1. **treatment:** true episode-local rule cards and opcode bindings;
2. **family-deranged cards:** same source, initialization, parameters, updates,
   and data order, with all three card labels rotated inside each family; and
3. **equality-ablated witness:** same physical surfaces and budget, but repeated
   witness identities are independently renamed so the relation is unavailable.

Evaluation additionally applies opcode-deranged binding, card-storage reindex,
witness/opcode alpha rename, post-HALT suffix, witness corruption, source-free,
uniform-packet, shuffled-packet, and gold-packet controls. No arm may share an
output directory or read another arm's predictions.

## 7. Pre-board evidence

- 14 focused adapter/mechanics/receipt tests pass.
- Ruff, byte compilation, and diff checks pass.
- Every declared trainable tensor receives a nonzero gradient in a real compiler
  forward/backward; no excluded tensor receives a gradient.
- The tied motor fits all 36 state/card transitions exactly.
- The actual confirmed parent reconstructs and copies byte-identically.
- The exact parameter and excluded-state certificates above reproduce locally.
- The frozen CPU mechanics report passes all seven gates over 10,000 episodes;
  card derangement retains only 15.08% exact final state.

## 8. Ordered custody

1. Commit and push this architecture, implementation, tests, and receipt.
2. Draw one board seed only after that exact commit.
3. Build 48,000/2,048/2,048 train/development/sealed-confirmation rows, audit all
   semantics and split exclusions, and reproduce the build byte-identically.
4. Commit the board receipt, then draw one training seed.
5. Run one development job with atomic access ledger and independent assessment.
6. Open confirmation once only if every frozen gate in
   `R12_ER_CST_EPISODIC_RULE_CARD_THEORY.md` passes.

Any architecture, parameter, supervision, optimizer, board, threshold, or
evaluator change after its corresponding freeze requires a fresh version and
fresh unopened board. A mechanics or fit pass alone is not neural reasoning
evidence.
