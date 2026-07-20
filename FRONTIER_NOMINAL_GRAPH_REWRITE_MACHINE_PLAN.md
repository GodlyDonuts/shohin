# The central thesis

Shohin should not be pushed toward general reasoning by making the language model “think longer.” It should be turned into a **renaming-invariant compiler for a small, model-owned reasoning computer**.

The evidence already points there. S7 shows that a tiny learned generator can induce and recurrently execute unfamiliar laws when composition is forced through the correct algebraic basis. S8.1 shows that, once a valid graph exists, order, state transition, query consumption, and nil termination work exactly. S9 shows that **identity-before-semantics** nearly solves whole-source grounding: its main failures are proposal/class completeness and operation-name recoding, not downstream computation. In particular, the 1,925 original/recoded cases that remained valid were bit-identical; the 18 failures became invalid before semantics could be compared. 

That suggests the following equation:

[
\text{general reasoning}
\approx
\text{invariant compilation}
+
\text{first-class operation definitions}
+
\text{forced compositional execution}
+
\text{model-owned agenda control}
+
\text{causal language realization}.
]

The deepest architectural move is to stop treating operation names, entity names, node numbers, memory addresses, and primitive labels as meaningful. They are **gauge choices**—arbitrary coordinate systems. Meaning should live in relations, effects, types, and executable definitions.

---

# 1. Define the attainable target correctly

At this parameter scale, “general-purpose reasoning” should initially mean:

> Given facts, definitions, examples, or rules in context, Shohin can construct an anonymous structured problem representation, induce unfamiliar operations when sufficiently specified, decompose the goal, update persistent state, branch, terminate, and produce an answer across math, code, logic, and finite planning tasks.

That is different from possessing broad encyclopedic knowledge or being an excellent conversational assistant. Those are additional language and knowledge problems. The first target is a **domain-general algorithmic reasoner over supplied context**.

A suitable end-to-end architecture is:

[
G_0 = C_\alpha(x)
]

[
\mathcal R = H(G_0^{\text{definitions}})
]

[
a_t = \Pi_\theta(G_t,S_t,\mathcal R)
]

[
(G_{t+1},S_{t+1}) =
U_\theta(G_t,S_t,\mathcal R_{a_t})
]

[
y = D_\theta(G_T,S_T,q).
]

Here:

* (C_\alpha) is an alpha-invariant language compiler.
* (H) turns definitions and demonstrations into executable **rule cards**.
* (\Pi_\theta) is a model-owned agenda/controller.
* (U_\theta) is a tied recurrent executor.
* (D_\theta) realizes the terminal state in natural language.

The host may store tensors, apply model-emitted graph edits, enforce declared types and memory bounds, and invoke learned cells. It may not choose the plan, supply an operation, execute arithmetic, repair a graph, run candidates against the final answer, or select a winning answer externally.

---

# 2. The immediate S9.1 experiment

S9.1 should be a very targeted repair, not a wider network or another generic training run. S9 already passes 20 of 22 gates. Its exact graph score is 94.775%, versus 46.387% for the equal-parameter no-class arm, and its valid graphs are almost always semantically exact. The failed gates are only class completeness and recoding eligibility. 

## 2.1 Fix candidate-space equivariance before adding capacity

The recoding result contains a major clue:

* When both original and recoded sources produce valid graphs, their results are bit-identical.
* The failures occur because recoding changes tokenization and makes some required span/class candidates disappear or become unselected.

So the relation network is already close to equivariant **conditional on proposal validity**. The first repair should therefore be to make the proposal space closed under renaming.

### Replace token-width-bounded islands with byte-aligned nominal islands

The current compiler scores token spans up to width four. That makes semantic eligibility depend on BPE segmentation. Instead:

1. Enumerate bounded **byte-coordinate spans**, perhaps up to 24 or 32 bytes.
2. Map each byte span to its overlapping Shohin token residuals.
3. Pool start, end, mean, and boundary-local byte features.
4. Score the entire span jointly, as S9 does now.
5. Keep the candidate enumeration ignorant of whether the span is an entity, operation, event tag, or irrelevant word.

This does not allow the host to identify names. It merely guarantees that a renamed symbol remains representable even when it changes from one BPE token to six.

A small byte convolution or byte embedding path could be added for boundary robustness, but the semantic relation encoder should remain unchanged.

## 2.2 Make operation symbols alpha-equivariant by construction

After the model proposes operation spans, replace their lexical identity with an anonymous per-example atom.

A practical implementation is:

[
e(o_i)= e_{\text{operation}} + r_i,
]

where (e_{\text{operation}}) is shared and (r_i) is a per-example distinguishing code that is independently permuted or regenerated. Downstream computation sees the shared operation type and relationally distinguishable atom, but not a stable lexical embedding.

A stronger version uses parallel shared-weight streams: each operation class receives its own stream, while aggregation across streams is permutation invariant. Recent symbol-invariant Transformer work gives an exact alpha-renaming invariance construction using precisely this pattern—shared per-symbol streams plus permutation-invariant aggregation. It also shows that the approach can be retrofitted into pretrained models, though computation grows with the number of interchangeable symbols. ([[arXiv](https://arxiv.org/html/2601.23169v2)][1])

For Shohin, the number of operation classes per source is small, so this is an excellent place to spend compute rather than parameters.

## 2.3 Replace independent relation decisions with model-logit-only structured assignment

S9’s residual misses look like structured recall failures: a missing class, missing witness, duplicate relation, or invalid cardinality can destroy the whole graph. Independent local argmax decisions are unnecessarily brittle.

Let the model produce unary and pairwise scores for candidate graph elements. Decode:

[
\hat g =
\arg\max_{g\in\mathcal G}
\left[
\sum_i s_i(g_i)
+
\sum_{i,j}s_{ij}(g_i,g_j)
\right],
]

where (\mathcal G) is the set of graphs satisfying only the declared syntax and typing rules.

Permitted constraints would include:

* exactly one entry event;
* exactly one terminal nil;
* each event has one operation, entity, and successor/nil assignment;
* each rule card has the required witness slots;
* each initial-state position is used once;
* all active event nodes form one reachable nil-terminated chain;
* relation endpoints have compatible structural types.

The decoder must not use:

* executor output;
* final-state or answer agreement;
* gold depth;
* source order;
* a solver-derived semantic label;
* retries based on answer correctness.

This is the same scientific boundary as grammar-constrained parsing: structural well-formedness is an architectural prior, not semantic repair. Recent work on semantic constrained decoding shows that generation can be constrained in program space rather than only token space, but Shohin’s version must deliberately stop at graph grammar and typing rather than equivalence to a target answer. ([[arXiv](https://arxiv.org/pdf/2509.00360)][2])

Mandatory controls should include uniform logits through the same constraint layer, shuffled logits, source-free logits, unconstrained S9 logits, and oracle logits. The board should also report how many valid assignments exist per source, so the grammar cannot secretly determine the answer.

## 2.4 Train on the full renaming orbit

For a random operation-name permutation (\pi), require:

[
C_\alpha(\pi x)
===============

\pi C_\alpha(x)
]

before anonymous class IDs are discarded, and exact equality after canonical class reindexing.

A useful loss is:

[
L =
L_{\text{graph}}
+\lambda L_{\text{orbit}}
+\mu L_{\text{structured-margin}}.
]

`L_orbit` aligns span participation, class membership, and relation logits across source-level recodings after adjusting the byte-coordinate map. `L_structured-margin` requires the gold graph to score above every loss-augmented valid alternative.

The training board should deliberately contain operation names whose tokenization ranges from one to perhaps eight BPE tokens, including same-length strings with different token counts and different-length strings with the same token count.

## 2.5 Keep the S9.1 scientific claim narrow

S9.1 should pass every existing frozen S9 gate on a fresh board, with two strengthened requirements:

* **100% operation-recoding eligibility** for every originally valid graph.
* **Bit-identical canonical graphs** after operation renaming and class reindexing.

Do not add alias/coreference, new algebras, dynamic planning, or a larger executor to S9.1. A clean confirmation here would establish a powerful foundation: exact repeated-reference grounding that is invariant to tokenization and symbol names.

---

# 3. The architectural leap: operations must become data

Even a perfect S9.1 still has a finite graph grammar and a specialized cyclic executor. General reasoning requires eliminating the assumption that the compiler already knows the semantic category of every operation.

The key transition is:

[
\text{operation token}
\quad\longrightarrow\quad
\text{first-class rule object}.
]

Instead of classifying an unfamiliar phrase as `left`, `right`, `add`, or `subtract`, Shohin should construct a **rule card**:

[
R =
(\tau_{\text{in}},
\tau_{\text{out}},
D,
F,
p,
q).
]

Where:

* (\tau_{\text{in}},\tau_{\text{out}}) are input/output types.
* (D) links to definitions, examples, equations, or code.
* (F) is an effect fingerprint on a determining set.
* (p) is a discrete microprogram.
* (q) is uncertainty or a distribution over candidate programs.

This is the most direct generalization of S7. S7’s two witnesses plus learned cyclic generator are already a specialized rule card. The next system should make rule cards generic and composable.

## 3.1 Choose a hypothesis family before predicting its parameters

S6 demonstrated that mathematical identifiability does not make a generic Transformer discover the right algebra. It fit every training law and still learned a lookup surface. S7 succeeded because its representation forced the correct generator composition.

The general lesson is:

> Do not ask a Transformer to emit an opaque operation embedding. Ask it to select a restricted hypothesis family and fill that family’s determining representation.

Examples:

| Hypothesis family       | Determining representation                     |
| ----------------------- | ---------------------------------------------- |
| permutation             | images of basis elements or generator word     |
| affine map              | basis-point images or coefficients             |
| Boolean operation       | truth table or Boolean circuit                 |
| finite-state transition | local transition cells                         |
| stack operation         | typed push/pop/read/write effects              |
| list transformation     | rewrite program over head/tail primitives      |
| graph operation         | local edge/node rewrite program                |
| arithmetic              | digit/limb transition program with carry state |

The family is an architectural and learned prior. The evidence must be sufficient to identify a member. If it is not, the model should preserve multiple candidates rather than hallucinating one.

## 3.2 Use a learned RISC-like reasoning microkernel

Shohin should have a small set of shared, model-owned primitive transitions rather than one neural module per benchmark operation.

A plausible typed instruction basis includes:

* categorical equality and comparison;
* copy, swap, select, and branch;
* successor/predecessor on learned domains;
* read/write of typed registers;
* push, pop, pair, and unpair;
* follow, add, and remove graph edges;
* call, return, commit, and emit;
* local digit/bit transition cells.

Some of these—addressing, equality, graph storage, type masks—may remain disclosed architectural operations. Semantic transitions should be learned cells, confirmed on exhaustive or determining atomic boards, and reused recurrently.

A high-level operation then compiles to a program over these primitives:

[
\text{“rotate twice then reverse”}
\mapsto
[\texttt{ROTATE},\texttt{ROTATE},\texttt{REVERSE}].
]

The operation name is irrelevant. Its definition or behavior determines the program.

This is a **reasoning RISC machine**:

* primitive semantics are small enough to learn exactly;
* complex operations are sequences, not new parameter blocks;
* held-out composition is forced through shared weights;
* programs can be inspected, intervened on, and causally tested.

A hard-coded host interpreter would only be an upper bound. The promoted treatment must use learned primitive cells and a model-generated program.

## 3.3 Make the graph homoiconic

The same graph should represent:

* data;
* program;
* operation definitions;
* goals;
* subgoals;
* control dependencies;
* current state;
* uncertainty.

In other words, **code is data inside the graph**.

A compact universal grammar could use generic node types:

* `OBJECT`
* `VALUE`
* `RULE`
* `APPLICATION`
* `GOAL`
* `STATE_SLOT`
* `RESULT`

And generic relations:

* `BINDS`
* `DEFINES`
* `ARGUMENT`
* `PRODUCES`
* `DEPENDS_ON`
* `TRUE_BRANCH`
* `FALSE_BRANCH`
* `NEXT`
* `QUERY`
* `OUTPUT`

New operations do not require new relation labels. They are new `RULE` nodes containing new microprograms.

This is how Shohin can escape a permanently finite operation ontology without requiring a new architecture for every domain.

## 3.4 Use provided examples as internal unit tests

When a task defines a novel operation with examples, the rule compiler can generate a small number of candidate microprograms and execute them on those examples using Shohin’s own learned executor.

The system would:

1. compile candidate programs;
2. run each on the supplied demonstrations;
3. score agreement;
4. retain or reweight candidates;
5. execute the query using the surviving program distribution.

The demonstrations are part of the source, so no external verifier adds information. The model is simply performing inference over existing evidence.

This extends S7’s witness mechanism to richer operations.

Crucially, underdetermined examples should produce an underdetermined rule state. Training should include one-witness and observationally equivalent cases whose correct response is ambiguity, abstention, or preservation of multiple candidates.

---

# 4. Planning and halting should be an agenda, not a prose trace

The next missing interface is not another hidden scratch vector. It is a model-owned **work graph**.

## 4.1 Replace a linear event list with an obligation graph

Each reasoning node should have a status such as:

* unresolved;
* ready;
* executing;
* committed;
* blocked;
* retired.

A slow planner chooses an unresolved goal and emits one of a small number of graph edits:

* expand into subgoals;
* bind a rule;
* apply a rule;
* branch;
* commit a result;
* retire a completed obligation;
* halt.

A fast executor applies the selected microprogram to the current typed state.

This naturally separates the controller from the executor without letting the host schedule either.

## 4.2 Use two timescales, but over explicit state

Recent HRM and TRM results suggest that small networks can benefit considerably from repeated or hierarchical computation. But independent analysis of TRM found substantial dependence on puzzle identity and aggressive test-time augmentation, and observed that much of the gain appeared early in the recursion. That means recurrence is useful, but it should not be treated as evidence of general reasoning by itself. ([[arXiv](https://arxiv.org/abs/2506.21734)][3])

For Shohin:

* the **fast loop** executes one local state transition or graph rewrite;
* the **slow loop** revises the goal decomposition, chooses a rule card, or opens a branch;
* the two loops have separate parameters, losses, and state fields;
* fast-loop gradients do not rewrite the planner’s ontology;
* planner gradients do not modify primitive transition semantics.

This directly addresses the typed-controller interference already observed.

## 4.3 Make state updates transactional

Each executor step should produce a proposed delta:

[
\Delta_t =
(\text{writes},\text{new nodes},\text{new edges},\text{retirements}).
]

The model also predicts `COMMIT` or `ABORT`. The runtime applies only committed deltas.

The runtime may check:

* type compatibility;
* bounds;
* pointer validity;
* single-writer rules;
* graph well-formedness.

It may not check whether the arithmetic or semantic result is correct.

This two-phase structure helps prevent accidental overwrites, stale-state reuse, and replay loops without pretending to correct common-mode semantic mistakes.

## 4.4 Tie halt to graph state

`DONE` should not be a token learned from textual position.

A terminal state should require:

1. an answer node has been committed;
2. no unresolved required dependency remains;
3. the model emits a terminal nil/HALT transition;
4. the committed state remains stable for one subsequent control check.

The important point is that the model creates and retires the obligations. The runtime only evaluates the literal status structure that the model produced.

Terminal/nonterminal twins should use locally identical answer-like states: one must halt because the agenda is empty; the other must continue because an unresolved dependency remains elsewhere in the graph.

## 4.5 Add multi-hypothesis search only after the single path works

A later controller could maintain four candidate rule/program graphs with model-owned weights. Recent Recursive Inference Machine work interprets recursive reasoners as proposal mechanisms that can benefit from a learned reweighting stage; denoising-recursion work provides a curriculum for recovering useful states through recurrent refinement. ([[arXiv](https://arxiv.org/html/2603.05234v1)][4])

For Shohin, this should be treated as a search and optimization improvement, not as a source of new information.

The four hypotheses should be deliberately heterogeneous:

* relation-first parse;
* effect/fingerprint-first parse;
* query-backward parse;
* syntax-first parse.

A reweighting head may use only the source, model-owned graphs, supplied demonstrations, and predicted state consistency. It may not use final-answer correctness or an external solver.

---

# 5. Connect reasoning to language without textual chain-of-thought

The terminal state should causally drive language generation through a structured interface. Training Shohin to imitate long rationales would risk reproducing the exact failure mode you are trying to avoid: fluent reasoning-shaped text disconnected from the recurrent machine.

## 5.1 Use the graph as a prefix memory

After reasoning terminates, expose the terminal graph and state to selected Shohin layers as structured memory:

* one embedding per graph node;
* edge-type attention biases;
* categorical state embeddings;
* a query node;
* node-order randomization during training.

A recent graph-language architecture showed that graph topology can be injected directly into language-model attention using graph-aware biases while preserving node-order equivariance and retaining fine-grained node text rather than compressing every node to one opaque summary. That is a useful design precedent for Shohin’s graph-to-language bridge. ([[arXiv](https://arxiv.org/html/2605.10247v1)][5])

The bridge should be small—perhaps adapters or cross-attention in the top four to six layers.

## 5.2 Enforce a lexical firewall

Once the source has been compiled:

* operation and entity names become anonymous IDs;
* the executor sees no original lexical embeddings;
* the final realizer receives the terminal graph, query wording, and a name-restoration map;
* name restoration occurs only during serialization.

This cleanly separates semantic computation from surface realization.

It also creates powerful interventions:

* rename every source symbol: computation unchanged, output names appropriately renamed;
* swap two anonymous state nodes: answer follows the swap;
* preserve graph but alter original source wording: answer remains;
* preserve source but zero graph memory: answer collapses;
* supply a counterfactual valid terminal state: language follows the state rather than memorized source associations.

## 5.3 Train answer realization before explanation realization

The sequence should be:

1. structured answer head;
2. exact concise natural-language answer;
3. explanation generated from the executed graph;
4. conversational presentation.

Do not train explanations until terminal-answer generation is causally dependent on the graph.

Explanations should cite graph nodes or operation steps internally, making it possible to verify that each sentence corresponds to an executed dependency. They need not expose private latent reasoning; they should summarize the model-owned graph.

## 5.4 Rehabilitate language without teaching fake reasoning

Shohin’s raw language checkpoint will probably require targeted language adaptation. Use:

* instruction and output-contract data;
* paraphrase-to-graph data;
* definitions paired with typed rule cards;
* code/docstring/AST correspondences;
* entity and coreference binding;
* ordinary conversational replay for preservation.

Larger teacher models can generate surface paraphrases, but the semantic graph should come from the verified task generator—not from teacher reasoning traces. This distills language coverage rather than another model’s apparent thought process.

---

# 6. The training curriculum

The curriculum should progressively remove privileged supervision while preserving independent component gates.

| Stage    | Capability                          | Training object                             | Critical held-out dimensions                                 |
| -------- | ----------------------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| **S9.1** | exact repeated-reference grounding  | alpha-normalized occurrence graph           | operation names, BPE widths, renderer, node order            |
| **S10**  | non-identical reference binding     | learned mention-equivalence quotient        | synonyms, pronouns, aliases, similar distractors             |
| **S11**  | unfamiliar rule compilation         | typed rule cards and discrete microprograms | names, definitions, coordinate systems, program compositions |
| **S12**  | dynamic planning and halt           | agenda/work graph with graph edits          | depth, breadth, branch order, storage order, terminal twins  |
| **S13**  | source-deleted integrated execution | predicted graph, state, agenda              | mixed domains, lengths, values, modalities                   |
| **S14**  | language realization                | terminal graph to answer/explanation        | paraphrases, output formats, donor states                    |
| **S15**  | broad transfer                      | math, code, logic, planning mixture         | entire task families and cross-domain compositions           |

## 6.1 Train across multiple description modalities

Every latent operation or program should be rendered as some combination of:

* controlled natural language;
* free paraphrase;
* equations;
* truth tables;
* input/output examples;
* pseudocode;
* executable code;
* diagrams or graph descriptions.

The same operation must compile to the same anonymous rule card across modalities.

This discourages the model from equating semantics with a single wording family.

## 6.2 Use a contrastive twin matrix

For every program family, generate all four categories:

| Surface relation | Semantic relation |
| ---------------- | ----------------- |
| nearly identical | same              |
| nearly identical | different         |
| very different   | same              |
| very different   | different         |

Include:

* token-bag-identical order twins;
* same operation name assigned different local definitions;
* different names assigned the same definition;
* aliases versus near-alias distractors;
* terminal versus nonterminal local twins;
* same final answer reached through different state trajectories;
* different answers sharing the same superficial trace shape.

These are more valuable than another large undifferentiated SFT corpus.

## 6.3 Use progressive interface deletion

Training can begin with exact intermediate supervision, but progressively delete it:

1. gold graph and gold state;
2. predicted graph, gold state;
3. predicted graph and predicted state, gold microprogram;
4. predicted graph, rule card, state, and agenda;
5. final answer plus structural and causal objectives.

At every transition, freeze a fresh board before removing supervision.

## 6.4 Keep losses physically separated

Use distinct parameter islands for:

* boundary and nominal binding;
* relation graph assembly;
* rule-card compilation;
* primitive execution;
* agenda control;
* halt;
* language realization.

Use stop-gradients at interfaces during component training. Only after every island passes should a small calibration layer be jointly trained.

The executor should never receive language-generation loss. The compiler should not receive final-answer labels during its first gate. The realizer should not be allowed to repair the graph.

## 6.5 Denoise only after semantics are correct

Denoising graph and state corruptions may improve recovery from off-manifold errors. But it cannot determine which valid semantic state was intended. Use denoising for:

* missing edge;
* duplicated node;
* stale status bit;
* perturbed continuous node feature;
* partially erased state packet.

Do not claim it corrects a coherent wrong operation program. The ledger’s coding and reversibility no-gos remain decisive there.

---

# 7. The three quotients Shohin ultimately needs

A useful unifying theory is that general reasoning requires three distinct quotient operations.

## 7.1 Occurrence quotient

Mentions with exactly repeated surfaces are grouped.

S9 nearly establishes this.

## 7.2 Nominal quotient

Different names and surfaces are recognized as representing the same bound object or rule when their relational role demands it.

Examples:

* `x` versus `temperature`;
* “the vessel” versus “it”;
* `combine` versus a later reference to “that operation”;
* alpha-renamed program variables.

This requires learned equivalence edges plus exact transitive clustering. Exact surface equality becomes one high-confidence feature, not the definition of identity.

## 7.3 Causal quotient

Histories or states are grouped only when every admitted future continuation produces the same behavior.

S3/S7 establish bounded pieces of this: categorical state is causally reused, and learned generator dynamics compose.

The complete route is therefore:

[
\text{language}
\rightarrow
\text{nominal quotient}
\rightarrow
\text{typed program}
\rightarrow
\text{causal state quotient}.
]

Many neural reasoning failures arise because these quotients are blended into one continuous residual stream. Shohin’s experiments strongly suggest they should remain explicit.

---

# 8. Exactness requirements become severe with depth

Component-level accuracy must be much higher than ordinary benchmark accuracy.

If a task requires ten dependent steps and you want 80% exact trajectories, independent per-step reliability must be approximately:

[
0.8^{1/10}\approx 97.8%.
]

At 32 steps it must be approximately:

[
0.8^{1/32}\approx 99.3%.
]

Similarly, a compiler at 95%, executor at 99%, halt at 98%, and serializer at 98% yield only:

[
0.95\times0.99\times0.98\times0.98
\approx90.3%
]

before accounting for multiple recurrent transitions.

That is why general-purpose Shohin should use:

* categorical or typed state;
* exact addressing and equality;
* forced primitive reuse;
* globally structured graph decoding;
* source deletion;
* strict causal interventions.

A fuzzy latent workspace with 90% local behavior will never produce reliable long computations.

---

# 9. A plausible parameter envelope

The current S9 system totals 134,580,264 parameters, leaving 15,419,735 parameters below a strict `<150M` ceiling. 

A reasonable post-S9.1 envelope is:

| Addition                                           | Target budget |
| -------------------------------------------------- | ------------: |
| byte-boundary and structured-assignment additions  |          0.3M |
| learned alias/nominal quotient                     |          0.8M |
| agenda planner and transactional controller        |          3.2M |
| typed rule-card and microprogram compiler          |          3.8M |
| graph-to-language attention bridge                 |          2.4M |
| route, uncertainty, and optional reweighting heads |          0.8M |
| **Total additions**                                |     **11.3M** |
| **Projected complete system**                      |  **≈145.88M** |
| **Remaining reserve**                              |    **≈4.12M** |

This is an engineering envelope, not an audited count. But it shows that the project does not need a major new backbone. The main resource should be **shared sequential computation**, not more unique weights.

---

# 10. The experiments I would run, in order

## Experiment 1: S9.1 Alpha-Closed Structured Compiler

Change only:

* byte-aligned candidate spans;
* anonymous operation atoms;
* operation-renaming orbit training;
* model-logit-only structured graph assignment.

Keep:

* S7/S8 runtime;
* parameter count as close as possible;
* all S9 controls and thresholds;
* fresh development and sealed confirmation.

Add controls:

* token-span S9 architecture;
* byte spans without orbit training;
* orbit training without structured assignment;
* structured assignment with uniform logits;
* structured assignment with shuffled logits;
* relation encoder with anonymous IDs disabled.

This is the highest-confidence next move.

## Experiment 2: First-Class Rule Cards

Build a CPU falsifier and neural board with three or four typed operation families:

* cyclic/permutation operations;
* Boolean circuits;
* finite-state or stack transducers;
* simple list rewrites.

Each episode gives unfamiliar names and either definitions, demonstrations, or both. The model must emit:

* type signature;
* family;
* determining fingerprint;
* discrete microprogram.

The executor receives no operation name or source text.

Controls:

* generic dense law embedding;
* law-ID memorizer;
* shuffled definitions;
* ambiguous evidence;
* deranged primitive semantics;
* program with state reset;
* host exact interpreter ceiling.

The key gate is held-out **program composition**, not merely held-out names.

## Experiment 3: Agenda-Graph Control

Use tasks with branching and subgoals:

* variable depths from 3 to 32;
* random graph storage order;
* shared-prefix terminal/nonterminal twins;
* independent branch order;
* distractor obligations;
* source deletion after initial graph compilation.

The model must build and update the work graph, select the next ready node, commit state changes, and emit HALT.

The runtime may reject malformed graph edits but may not repair them.

## Experiment 4: Causal Graph-to-Language Bridge

Freeze compiler, planner, and executor.

Train only the language bridge on:

* terminal state → concise answer;
* query + terminal state → requested format;
* terminal graph → explanation.

Require:

* donor-state following;
* zero/shuffled graph collapse;
* node-order invariance;
* source deletion;
* counterfactual valid-state verbalization;
* no answer recovery when the graph is wrong.

Only after this passes should broader conversational SFT touch the integrated stack.

---

# 11. Longer-term, higher-risk ideas

## 11.1 Verified skill crystallization

When a microprogram motif recurs often, distill it into a new learned generator or macro.

The macro must be behaviorally equivalent to its expanded program on a complete determining set or an explicitly bounded domain. The expanded program remains the control.

This would let Shohin develop a hierarchy:

[
\text{primitive}
\rightarrow
\text{microprogram}
\rightarrow
\text{macro}
\rightarrow
\text{high-level plan}.
]

It is analogous to compiler optimization and human skill chunking, while retaining exact causal accounting.

## 11.2 Gauge-equivariant reasoning everywhere

Extend invariance beyond operation names:

* entity renaming;
* variable alpha-renaming;
* graph node order;
* memory-slot permutation;
* primitive code relabeling;
* hidden coordinate recoding;
* output format recoding.

Every stage should expose its transformation group and be tested under it.

The model should operate on relational observables; a coordinate system should be chosen only when a pointer must be dereferenced or an answer serialized.

## 11.3 Version-space reasoning

Instead of always committing to one interpretation, maintain a small set of rule cards or plans consistent with the current evidence.

The model can:

* collapse the set when new evidence distinguishes candidates;
* execute candidates in parallel when inexpensive;
* return “underdetermined” when no distinction exists;
* ask a clarifying question in an explicitly interactive setting.

This would be a more profound reasoning capability than merely raising forced-choice accuracy: Shohin would distinguish **uncertainty from computational failure**.

## 11.4 Self-generated counterexample curricula

During training, use the exact task generator to find minimal source mutations that separate the model’s current wrong graph from the true graph:

* one changed argument edge;
* one order swap;
* one alias split;
* one operation-definition mutation;
* one terminal-status change.

This is training-time data engineering, not an inference-time reasoning mechanism. Oracle calls and generator work must be counted.

---

# 12. What not to do next

Do not:

* widen the S9 encoder before repairing proposal equivariance;
* reopen the S9 board or relax the 95% gate;
* add generic token-level recurrence and call it reasoning;
* train long chain-of-thought traces into the raw decoder;
* let a solver execute candidate graphs and select the one with the right answer;
* make carry, arithmetic, or scheduling a hidden host routine;
* merge compiler, executor, halt, and chat losses from the beginning;
* treat syntactic validity as semantic correctness;
* add redundancy after all semantic lanes have already chosen the same wrong program;
* assume a larger operation vocabulary is the same as learning new operations.

---

# Bottom line

The most promising route is a **Nominal Graph Rewrite Machine**:

1. **Alpha-invariant compiler:** language becomes an anonymous typed graph.
2. **First-class rule cards:** unfamiliar operations become executable data.
3. **Learned reasoning RISC:** complex behavior compiles to a small set of tied, verified primitive cells.
4. **Agenda controller:** the model owns decomposition, next-step choice, state commit, branching, and halt.
5. **Graph-conditioned realizer:** natural language is generated from the terminal state rather than used as the computational workspace.

This is not merely a collection of modules. It is a coherent theory of why S7 and S9 worked:

* S7 succeeded because it removed arbitrary coordinate dependence and forced generator reuse.
* S9 succeeded because it quotiented repeated identity before predicting semantics.
* General reasoning should continue the same pattern: **quotient away arbitrary names, reify definitions, and force every long computation through a small compositional basis.**

The immediate move is S9.1 with a tokenization-independent candidate lattice and structured relation assignment. The decisive move after that is to stop classifying operation names entirely and begin compiling operation **definitions** into discrete programs.

[1]: https://arxiv.org/html/2601.23169v2 "https://arxiv.org/html/2601.23169v2"
[2]: https://arxiv.org/pdf/2509.00360 "https://arxiv.org/pdf/2509.00360"
[3]: https://arxiv.org/abs/2506.21734 "https://arxiv.org/abs/2506.21734"
[4]: https://arxiv.org/html/2603.05234v1 "Recursive Inference Machines for Neural Reasoning"
[5]: https://arxiv.org/html/2605.10247v1 "Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning"
