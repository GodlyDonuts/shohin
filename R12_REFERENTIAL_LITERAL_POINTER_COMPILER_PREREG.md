# R12 Referential Literal-Pointer Compiler Preregistration

**Status:** **FROZEN 2026-07-18 before any neural fit, GPU job, confirmation
seed, or Shohin score.** This document authorizes only the deterministic CPU
semantic-compiler falsifier in `pipeline/semantic_compiler_falsifier.py`.

**Decision boundary:** the proposed compiler is not a new reasoning primitive.
It is a bounded text-to-program interface whose only admissible first claim is
improved exact compilation under controlled paraphrase, order, binding, and
distractor transfer. Execution, source-deleted state, autonomous recurrence,
serialization, and halt remain separate blocked stages.

## 1. Why this experiment exists

R4 established a real but incomplete result. A frozen Shohin base with dynamic
entity pointers improved exact programs from `469/896` to `624/896` and full
OOD answers from `2/192` to `51/192`. However, its evaluator did not require the
model to bind all operands:

```text
question text -> deterministic host lexer -> initial_values, operation_values
question text -> neural compiler          -> opcode, entity role, query
```

`safe_execute` then combined the neural opcode/query with the perfect lexical
values. R4's `program_exact` metric counted only opcode and query correctness.
This is an honest prior result, but it leaves a specific open interface: can a
model select every semantic operand from source text rather than receiving
numeric or referential values from a privileged lexer?

The frontier-plan proposal is therefore reduced to its smallest testable unit:
a **referential codon**

```text
[operation kind, entity-source pointer, literal-source pointer]
```

plus a query pointer. Pointers name token spans in the source. The host may
dereference a model-selected span exactly, but may not choose, repair, reorder,
or infer a span. Every dereference is charged as interface work.

## 2. Capability statement and no-go theorem

Let `x` be a source surface and let `P(x)` be its complete typed program,
including operation kinds, entity references, literal references, order, query,
and STOP. Let `E(P,q)` be an independently specified executor.

The compiler capability is

```text
C(x) = P(x)
```

under held-out renderer, entity, order, binding, and distractor combinations.
Exact-program accuracy is the primary metric. Answer accuracy is secondary and
must be reported both with the predicted compiler and an oracle compiler.

### Language-bridge collision no-go

If two surfaces are identical to the compiler but require distinct programs
whose complete future-query behavior differs, no deterministic compiler can be
exact on both. Under a balanced pair its maximum exact accuracy is `1/2`.

More generally, for an observable feature map `f`, the Bayes-optimal shortcut
ceiling on a finite board is

```text
sum_z max_p count(f(x)=z and P(x)=p) / number_of_examples.
```

The CPU falsifier computes this ceiling for token bags, operation bags,
entity/literal bags, absolute pointer positions, span widths, source length,
and renderer identity. A feature ceiling above `1/3` rejects the board before
training because the matched canonical/order/binding triples would be
shortcut-solvable.

This theorem does not make the compiler novel. It only certifies that the
frozen board does not contain the named direct leaks.

## 3. Exact CPU board

The development falsifier is a deterministic three-entity list machine. A
state is an ordered permutation of three nonce entities. An instruction is

```text
LEFT(entity_pointer, literal_pointer)
RIGHT(entity_pointer, literal_pointer)
```

where the literal is one or two positions. Movement is implemented by repeated
adjacent swaps with boundary clamping. A program contains exactly two
instructions followed by STOP. A late query asks for the entity at one of the
three positions.

The board contains 32 quartets, 128 surfaces total:

1. canonical rendering of a typed program;
2. independently rendered paraphrase of the same program;
3. token-multiset-matched operation-order twin with different behavior;
4. token-multiset-matched argument-binding twin with different behavior.

The canonical and both counterfactual twins have exactly equal Shohin-tokenizer
multisets. The query is selected mechanically so the canonical answer differs
from both twins. Each quartet uses three fresh nonce names with equal tokenizer
width. Distractor lines repeat a real entity and a real numeric literal but are
outside the instruction spans. Distractor position and introductory order vary
factorially.

Two independent executors must agree:

- executor A removes and reinserts the selected entity;
- executor B performs the specified number of adjacent swaps.

This finite board is deliberately favorable and bounded. Passing it says
nothing about arithmetic, free-form language, unbounded composition, or Shohin
trainability.

## 4. Frozen CPU gates

Every gate is mandatory:

1. exactly 32 quartets and 128 surfaces;
2. `128/128` typed-AST round trips;
3. `128/128` agreement between the two independent executors;
4. all canonical/paraphrase pairs have identical programs, terminal states,
   and complete three-query behavior;
5. every order twin and every binding twin has a distinct typed program,
   distinct terminal behavior, and the frozen query is a valid separator;
6. canonical/order/binding token multisets match in all 32 quartets after the
   immutable Shohin tokenizer;
7. every operation kind, entity argument, numeric argument, and query has a
   nonempty exact character span and nonempty tokenizer span;
8. nonce names are disjoint across quartets and have equal token width within a
   quartet;
9. every named shortcut feature has Bayes-optimal exact-program accuracy at or
   below `1/3` on the 96 matched surfaces;
10. no teacher model, answer model, remote service, model checkpoint, or
    production evaluation answer is read;
11. the report includes exact source-token, target-pointer, executor-call,
    oracle-call, and external-execution counts;
12. generator, test, preregistration, tokenizer hash, seed, and report hashes
    are recorded before any confirmation board exists.

One failed gate rejects the board. It is not repaired after viewing a model
score. The first confirmation seed is forbidden until the development report
and code are committed.

## 5. Neural pilot authorized only after a CPU pass

A CPU pass authorizes one isolated compiler pilot from the immutable 300k base.
It does not authorize executor integration. The pilot must freeze a fresh
train/development/confirmation corpus before fitting and compare:

1. **full referential codon:** operation, entity, literal, query pointers;
2. **R4 pointer baseline:** same base/budget, with privileged literal lexer;
3. **absolute-role control:** no dynamic entity matching;
4. **ordinary biaffine pointer network:** identical supervision and budget;
5. **text AST decoder:** canonical program tokens, same trainable parameters,
   examples, updates, and inference FLOPs;
6. **joint adapter control:** same total parameters without separated heads;
7. **shuffled-pointer sanity control;**
8. **oracle compiler ceiling.**

All arms receive identical source strings, AST labels, pointer labels, examples,
optimizer steps, and confirmation access. Parameter equality must be exact or
the larger control must be reported as favorable. No arm may receive structured
operations, entities, literals, state, answer, or renderer ID at inference.

Promotion requires, on the untouched confirmation board:

- full exact-program accuracy at least `60%`;
- at least `+10` percentage points over the strongest matched non-oracle
  control;
- order-twin and binding-twin exactness each at least `55%`;
- paraphrase consistency at least `90%` among correctly compiled anchors;
- answer accuracy with predicted programs at least `55%`;
- donor pointer interventions change the execution result in the predicted
  direction at least `90%` of eligible cases;
- all named shortcut-only classifiers remain below the preregistered ceiling.

These thresholds are intentionally demanding because R4 already produced a
large binding gain. A new pilot must improve the missing complete interface,
not merely repeat noun binding.

## 6. Resource vector and collapse dossier

The CPU falsifier's resource vector is recorded exactly in its report:

```text
(parameters=0,
 retained_bits=serialized typed AST and board rows,
 precision=exact integers and byte strings,
 source_bytes=all UTF-8 source bytes read,
 training_examples=0,
 oracle_calls=typed-program construction and separator selection,
 training_FLOPs=0,
 inference_work=tokenization plus two exact executors,
 sequential_depth=2 instructions,
 external_memory=report bytes,
 external_execution=two CPU list-machine evaluators)
```

Known collapses:

- the compiler is supervised semantic parsing, not representation discovery;
- pointer dereference is an external read interface;
- the bounded list machine is a finite transducer;
- phase separation can be unrolled into an acyclic evaluator;
- exact symbolic execution is a favorable oracle, not neural reasoning;
- fixed two-instruction length does not test learned halt;
- typed slots supply an ontology and do not solve hidden-coordinate
  identifiability.

The only open empirical question is learnability under controlled transfer.

## 7. Stage boundaries

Stage A is this compiler falsifier and, only after a pass, its isolated neural
pilot. Stage B is a separately preregistered learned source-deleted packet
executor with no source/KV path. Stage C is a separately trained controller and
halt policy with terminal/nonterminal twins. End-to-end native reasoning is not
claimed until one uninterrupted model-owned rollout compiles, updates, reuses,
queries, serializes, and halts without host semantic repair.
