Shohin is in the rare, honest place. The raw 125.08M chat anchor is weak. The isolated 134.58M stack is strong. S7 proved 100% exact recurrent execution of unseen cyclic laws. S9 proved 94.775% exact graph and 94.873% exact state/answer from natural language with a +48.389 point lead over the no-class-message control, and 0% for shuffled relations.

That is not imitation. That is model-owned compilation: source -> occurrence spans -> quotient by exact bytes -> identity classes -> relation/control graph -> predicted event order + nil halt -> frozen cyclic executor -> state.

The failure mode is also honest: it works only under exact repeated references, finite graph grammar, and a specialized executor. It missed the 95% class gate by 5 examples and failed 18 operation-nonce renames. It is not yet a general reasoner because it is still a decompiler for a toy language, not a computer.

To get to general purpose, you do not scale the chat model. You make the reasoning stack the unavoidable bottleneck.

### 1. What S9 Actually Invented

S9 is not chain-of-thought. It is quotient typing.

In type theory and in HoTT, a quotient type says: take a set of occurrences, define an equivalence relation, collapse equivalent ones. S9 does that literally. It enumerates bounded contiguous spans, scores them, groups by exact byte equality, then builds relations over groups, not tokens. That is why shuffling relations kills it and reindexing classes does not. It is variable binding done right, under a tiny equivalence relation.

General reasoning is the same operation with a learned equivalence relation.

Human language is: "Alice... she... the engineer... she" -> one class. Math is: "let x = 5 ... x + x" -> one class. Code is: "for i... i". All are occurrence-quotient problems. S9 solved the trivial case where equivalence = string equality. S10 needs to solve equivalence = model-predicted coreference.

### 2. The S9.1 fix is not a patch, it is a symmetry

Those 18 rename failures are the tell. The model still treats operation names as content.

Solution: **learned renaming equivariance**.

Treat operation and entity nonce IDs as an exchangeable set. Do not embed the string "OP_3". Embed its occurrence cluster. Architecture change:

* Replace absolute span position embeddings in the relation decoder with relative occurrence-position inside the class.
* Train with full permutation augmentation of operation nonces in every batch, plus a DeepSets-style class aggregator: class representation = sum over occurrence embeddings, not mean of token embeddings. This makes the decoder permutation-equivariant by construction.
* Add a hard gate: if two valid parses differ only by nonce renaming, force bit-identical graphs via canonical ordering after grouping. You already proved this works on 1,925 cases. Enforce it architecturally.

Second S9.1 fix: **constrained logit relation assignment**. Right now relation kinds are predicted freely. Switch to type-checked decoding: model emits logits over relation kinds, but a tiny validator rejects graphs that violate arity or that split a class. Logits are trained with a straight-through estimator. This keeps it model-owned but removes impossible graphs from the search space. That should clear the 5-example class gap.

### 3. From exact equality to learned alias: the real S10

Exact byte equality was your honest free-word boundary. Your control where every occurrence is a unique word correctly makes S9 abstain 2048/2048. That is good.

Now you need a learned equivalence that can still abstain.

Proposal: **Two-stage quotient**

Stage A: occurrence proposer from S9, unchanged.
Stage B: alias scorer: for every pair of spans, predict P(same entity). Train this with contrastive loss on synthetic data where you have similar-but-nonidentical distractors: "Alice" vs "Alicia", "x" vs "y", "the cup" vs "a cup". This is the negative set you noted you need.

Then take transitive closure of P > threshold to form classes. Exact equality becomes a strong prior kernel, not a rule. Start training with exact equality weight 1.0, anneal to 0.3. This is how you get "she = Alice" without losing verifiability. The CPU theorem still holds: corrupt relation kind, swap slots, merge classes = rejected. Now add a new CPU arm: similar-but-distinct alias = rejected.

This connects to linguistics, databases, and compilers simultaneously. In compilers this is SSA register allocation. In databases this is entity resolution. In linguistics this is anaphora.

### 4. From cyclic executor to universal executor

S7's cyclic executor is a finite state transducer. It loops over successor cells. To be general, you need three minimal additions that keep verifiability:

1.  **Stack:** push/pop for call/return. Gives you recursion and nested subproblems. Math proofs need it.
2.  **Conditional branch on state predicate:** not just nil halt, but `if state.field == X goto`. Gives you if/else.
3.  **Typed value heap:** your current state is cards and entries. Extend to int, string, list, dict with a small frozen ALU.

All three can be implemented with the same pop-insert transition you already have. Keep the executor frozen and tiny, < 5K lines. The model does not learn to execute, it learns to compile to it. This is the WASM / eBPF insight: keep the runtime dumb and auditable, make the compiler smart.

Now you have a Turing-complete IR that is still exactly scorable.

### 5. How to connect to natural language without collapse

This is where every reasoning project dies: post-training makes the model bypass the stack and imitate reasoning-shaped text.

You need a **causal bottleneck**.

After S9 passes all 17 gates on fresh boards, freeze it completely. Then:

* For any natural prompt, force generation to go through: prompt -> S9 graph -> recurrent execution -> terminal state object.
* The language decoder may ONLY attend to the terminal state object + its execution trace, NOT to the original prompt. Implement this as hard cross-attention masking. Original source deleted after compilation, as in your required evidence.
* Train the decoder with a causal necessity test: if you shuffle or zero the terminal state, answer accuracy must drop to near 0. If you shuffle the prompt but keep state, accuracy stays. This is your existing state-swap intervention, but applied to NLG.

That prevents collapse. The chat model cannot cheat because it has no other information.

This is analogous to proof-carrying code and to neuroscience: hippocampus indexes, cortex cannot recall without it. The stack is your hippocampal index.

### 6. Curriculum to general purpose

Don't train on MATH yet. MATH is too noisy to prove ownership.

**Phase A: Open-world quotient**
Same boards, but names, renderers, and laws are sampled from a much larger LLM paraphraser that never sees the board structure. Your split-disjoint guarantee remains. This forces true language grounding.

**Phase B: Compositional closure**
Take two valid S9 graphs, concatenate their sources with "After that," and require the model to emit the composed graph where the output state of graph1 is input to graph2. This teaches graph composition and breaks fixed depth assumptions. Depth 1-8 -> depth 1-16 via composition only.

**Phase C: Self-play with verification**
Use the frozen executor as an oracle. Let Shohin generate its own board descriptions, compile them, execute, and check that description and execution match. Keep only the ones that pass CPU gates. This is AlphaProof-style self-distillation but with 100% verification, not reward model.

**Phase D: Tool as memory**
Give the executor a foreign function interface: `python.exec`, `calc`. The model must compile to a graph that CALLs those tools with correct args from state. Now you have math and code reasoning that is still exactly checkable: the graph contains the call.

### 7. Why this can stay under 150M

You do not need more parameters. You need less leakage.

* Share the Shohin residual extractor between proposer and alias scorer.
* Make alias scorer a 2-layer MLP over pair differences, not a full transformer.
* Use LoRA adapters for the relation decoder instead of a separate encoder.

The universal executor adds zero parameters. The bottleneck decoder adds parameters but can be a small 30M cross-attention head.

The scaling law that matters here is not pretrain loss. It is **state bits retained per parameter under source deletion**. Measure that directly.

If you do this, Shohin stops being a small LLM that sometimes reasons. It becomes a small verified compiler with a natural language front end. The chat backbone becomes a paraphraser that feeds the compiler. General purpose reasoning emerges not from longer traces, but from a stronger equivalence relation and a more expressive, but still frozen, target machine.