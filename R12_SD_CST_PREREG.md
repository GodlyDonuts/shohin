# R12 Source-Deleted Categorical State Transport Preregistration

**Status:** mechanics admitted; neural source, fresh board, and development score not yet frozen

**Track:** SD-CST, the first integration test after S9.2 parser-only anchor closure was rejected

## Question and honest boundary

SD-CST asks whether Shohin can compile a linguistic program into a minimal hard
state machine, delete the source, execute a tied learned update repeatedly, and
answer a query disclosed only after the program packet is committed. It tests a
bounded form of autonomous compositional state transport. It does not by itself
establish open-domain, mathematical, linguistic, or general reasoning.

The test integrates capabilities that were previously established only in
isolation: source-deleted recurrence, language-to-tape compilation, learned
transition semantics, explicit halting, and late state readout. A positive result
must therefore be autonomous and causal. Teacher-forced digit, field, motor, or
reader accuracy alone is not a reasoning claim.

## Frozen task

Each example contains three opaque entity names, an explicit instance-local
alpha/beta/gamma binding declaration, an independently arbitrary initial order,
seven valid state-changing operations, one explicit STOP, and a late query for
the entity at one of three positions. The seven operations and STOP form an
eight-slot semantic tape. STOP occurs after active depth one through six, so at
least one valid operation remains after STOP. All source texts have one roster
line and eight event lines. The eight event clauses are stored in an independently
chosen textual order and carry explicit semantic ordinals. STOP is one of those
randomized clauses, not a privileged final sentence.

An operation moves one bound entity left or right by one or two positions in a
three-entity permutation. The six possible permutations are the complete state
space. Every generated operation changes state even if it is after STOP. The
late query is a separate one-line source and is withheld until after program
compilation and deletion.

Training rows contain only compiler targets:

- one six-way initial-state category;
- eight three-way event-kind categories (`LEFT`, `RIGHT`, `STOP`);
- identity and amount categories for the seven non-STOP slots; and
- one separately compiled three-way late-query category.

Training rows contain no final state, answer, trajectory, episode reward, graph
validity, executor result, or retry feedback. Motor supervision is the complete
finite one-step table, independent of source rows. Reader supervision is the
complete finite state-query table.

## Score-bearing information boundary

Program text is processed once. The only retained payload is:

```text
initial_state: uint8[batch]        # 6 categories
event_kind: uint8[batch, 8]       # 3 categories
event_identity: uint8[batch, 8]   # 3 categories; ignored at STOP
amount: uint8[batch, 8]           # 2 categories; ignored at STOP
```

Exactly one event kind must be STOP in every scored row. Program IDs, masks,
token residuals, attention maps, logits, probabilities, margins, confidence,
pointers, and source text are destroyed before execution. The late query is then
compiled in a separate invocation into one `uint8` category without access to
program text, the committed packet, or recurrent state.

At score time the motor receives only one-hot views of the current integer state
and current integer event categories. Its argmax is converted back to an integer
before the next call. No motor logits or hidden activations cross a step. One tied
motor is called at every one of the eight public slots; there is no timestep
embedding, host operation table, variable Python loop, semantic branch, retry,
beam, repair, or external schedule. STOP closes an `alive` gate; all later slots
are still presented to the same motor but cannot update state. The reader sees
only the final integer state and separately committed integer query.

Soft categorical tensors and straight-through gradients may be used in training.
Only `HardProgramTape`, `HardLateQuery`, and `rollout_hard` may produce a reported
development or confirmation score.

## Finite integration theorem

Let `C` emit the exact initial state and exact eight-slot tape, `M` implement all
72 non-STOP state/action cells, the STOP gate preserve all six states, and `R`
implement all 18 state/query cells. Then hard SD-CST rollout returns the exact
halted state and answer for every board row.

Proof is induction over the fixed eight slots. The base state is exact by `C`.
For a live non-STOP slot, exactness is preserved by the corresponding `M` cell.
At STOP the state is unchanged and `alive` becomes false. Every later proposal is
masked, so the halted state is invariant to the suffix. `R` then maps the exact
state and late query to the exact entity. This is a finite composition theorem,
not evidence that natural-language compilation is exact; that is the empirical
bottleneck under test.

## Architecture and parameter cap

The base is the frozen 300,000-step Shohin checkpoint:

- 125,081,664 parameters;
- 30 layers, width 576, FF width 1,536;
- 9 attention heads, 3 KV heads;
- vocabulary 32,768, context 2,048, one transformer pass;
- checkpoint SHA-256 `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`;
- tokenizer SHA-256 `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4`.

The compiler reads frozen layer-19 residuals, projects them to width 384, uses
five trainable transformer-encoder layers with 8 heads and FF width 1,408, and
cross-attends one initial-state query plus eight event queries. A separate query
slot processes the late query. The compiler adds 9,205,009 parameters. The tied
motor has 19,206 parameters and the reader 835. Total added/trainable parameters
are 9,225,050; the complete system is exactly **134,306,714**, leaving 15,693,286
parameters below the strict 150,000,000 cap. The base remains frozen and in eval
mode throughout.

## Board and causal families

After all source, tests, assessor, launcher, and this preregistration are committed,
one independent board seed and one independent training seed will be drawn. The
fresh board contains:

- 48,000 compiler-only training rows;
- 288 development families, eight rows each, for 2,304 rows;
- 288 sealed-confirmation families, eight rows each, for 2,304 rows.

Each evaluation family contains canonical, query-swap, paraphrase,
binding-recode, order-counterfactual, STOP-shift, storage-order-shuffle, and
post-STOP-suffix variants. Query twins have byte-identical program text. The
generator and independent adjacent-swap auditor must agree. Exact prompts,
13-grams, names, templates, and operation sequences may not cross splits.
Query and answer roles are balanced per variant. Program/query length may not
predict active depth. Confirmation is mode `0600` and remains unopened unless
every development gate passes.

## Frozen optimization budget

The compiler receives four deterministic passes over the 48,000 unique training
rows: batch 64, exactly 3,000 updates, AdamW, peak learning rate `1e-3`, weight
decay `0.01`, 100 linear-warmup updates, cosine decay to zero, and gradient clip
1.0. The per-row loss is the unweighted sum of initial-state CE, event-kind CE,
non-STOP identity CE, non-STOP amount CE, and separately invoked late-query CE.
No outcome loss is permitted.

The motor receives the complete 72-cell non-STOP table and all six STOP states
for 2,000 full-table AdamW updates at learning rate `0.025`, zero weight decay.
The reader receives the complete 18-cell state-query table for 1,200 full-table
AdamW updates at learning rate `0.04`, zero weight decay. These fixed budgets are
charged even if exact fit occurs early. All seeds, update counts, optimizer state,
architecture values, source hashes, board hashes, base hash, tokenizer hash, and
parameter counts must be embedded in the checkpoint and evaluation output.

## Pre-board mechanics decision

The CPU-only falsifier uses only the standard library for generation and audit.
All registered mechanics gates pass:

- independent simulators agree on 72/72 atomic state/action cells;
- execute-through-STOP, textual-storage order, unordered event bag, STOP-blind,
  query-blind, and post-STOP-overrun controls each score 0%;
- length-only halt prediction is exactly chance at 1/6;
- resetting state every step reaches only 63.889%;
- all query leaks, training-answer leaks, nonidentical query twins, and cross-split
  sequence reuse mutations are rejected; and
- confirmation access is zero.

This admits only board mechanics. It is not a neural result.

## Sole fresh-development gates

One immutable development read is allowed. The unchanged run must satisfy every
gate below before confirmation can be opened:

1. Complete-system parameters are exactly 134,306,714 and below 150M; base,
   tokenizer, board, runtime, checkpoint, and access-ledger hashes all match.
2. Motor certificates are 72/72 non-STOP transitions, 6/6 STOP preservation,
   and 78/78 dead-state invariance. Reader certificates are 18/18.
3. Compiler exact tape is at least 95% overall and at least 90% for every causal
   variant and active depth. Initial state, event kind, non-STOP identity, amount,
   and late query are each at least 98% exact.
4. Autonomous exact final state and answer are each at least 90% overall and at
   least 85% at depth six.
5. Conditional on an exact compiled tape and query, execution state and answer
   are 100% exact. Any failure rejects the motor/reader integration theorem.
6. Query swaps preserve final state on 100% of pairs and the answer follows the
   swapped query on 100% of exact-tape eligible pairs, with at least 85% of
   families eligible.
7. Separating state swaps follow the swapped-state oracle on 100% of exact-packet
   eligible rows, with at least 85% of families eligible; freeze and reset
   controls are reported and must not equal the treatment trajectory on
   separating rows.
8. Post-STOP suffix variants preserve state and answer on 100% of exact-packet
   eligible pairs, with at least 85% of families eligible. With the gate forcibly
   held alive, 100% follow the full-suffix oracle instead, proving that the suffix
   is real and the learned halt matters.
9. Binding recodes and paraphrases preserve canonical abstract state/answer on at
   least 95% of mutually exact-packet pairs. Semantic order counterfactuals and
   STOP shifts follow their changed oracle on at least 90% overall.
10. Program-source poisoning after hard commitment, storage reindexing of event
    clauses, and discarded-logit poisoning are bit-identical on 100% of rows.
11. Uniform, source-free, and shuffled hard packets are each at most 25% exact
    state and at most 45% exact answer. Reset and freeze controls are each at
    most 75% exact state and at most 75% exact answer. Every control covers all
    2,304 development rows; the exact ten control names and thresholds are
    hash-bound in the gate configuration before development bytes are opened.
12. Development/confirmation access is exactly `1/0`; the exclusive development
    access ledger matches its preregistered byte hash; and every required row,
    certificate, control, intervention, budget, and hash record is present.

The assessor fails closed on missing evidence, duplicate IDs, nonfinite values,
changed config, or an unregistered score path. Failure closes this board without
rescore or confirmation access. Passing permits one separately frozen confirmation
read with identical bytes, checkpoint, evaluator, thresholds, and assessor.

## What a pass would and would not mean

A pass would establish that a 134.3M-parameter system can autonomously compile
held-out language into a source-deleted discrete program, execute a tied learned
transition across multiple steps, halt on a model-predicted event, and answer a
later query under strong causal interventions. It would be a real bounded native
reasoning result because the host supplies neither the schedule nor the arithmetic
transition.

It would remain a finite three-entity permutation algebra with a closed action
set. It would not establish transfer to unseen operators, larger state spaces,
variable tape lengths, free-form decomposition, or open-domain reasoning. Those
require fresh cardinality/action transfer boards after this integration gate,
not weakened claims on the present board.
