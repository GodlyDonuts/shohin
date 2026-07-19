# R12 S8.1 Nil-Linked Law Graph Development Result

**Decision:** rejected as an end-to-end neural compiler; retained as a positive
conditional-execution result

**Development access:** 1

**Confirmation access:** 0; the sealed confirmation board remains unopened

## Custody

- source repair commit: `ce2a5e4`
- board commit: `5f1550d`
- board seed: `5943437777437228096`
- training seed: `8354164228219389085`
- bf16 preflight: Slurm `693527` on `evc39`, passed
- sole score-bearing development job: Slurm `693529` on `evc39`, completed
  cleanly in 4m25s
- board report SHA-256:
  `1dcd576d9706c011ff8164994f0424f4bdc96a16525cdda400559b255b3aa831`
- development SHA-256:
  `d16a1a8f773ff627b5f47ebd06b2344cad67a6473c431d78d79a5ef41f360d54`
- sealed confirmation SHA-256:
  `e951ac173135ee7791528ae78206cb48900865dbeade65085fcf948a2da2977d`
- checkpoint SHA-256:
  `44b3291555047085257cfb1c4ec03dd6e5485ce83e134a5200d8ea0055614585`
- evaluation SHA-256:
  `74a391f3fd3f123da13007ad19cad8bf9075aa0809df3561a122f65c04267600`
- assessment SHA-256:
  `d6aaa221c58387010e79ee65ccfc9087c3073ed488d86bf9b932599c7f6eb119`

The local mirror contains the checkpoint, evaluation, and assessment with the
same hashes. The checkpoint is not committed to Git.

## Frozen contract

The compiler saw 48,000 whole-source graph-field rows and no final state,
answer, recurrent trace, development law, or confirmation law. The 218-
parameter cyclic generator saw only 23 successor cells and three zero anchors.
The complete system has 133,692,848 parameters: 125,081,664 frozen base,
8,610,966 graph compiler, and 218 generator parameters.

The repaired nonce intervention was exercised before sealing over all 52,096
board rows. It rotates operation strings in source, repairs spans, retokenizes,
and recompiles; 9,018 rows change token count. S8 v1's invalid equal-width token
substitution is not reused.

## Development scores

| Arm | Exact state | Answer |
|---|---:|---:|
| Gold graph | 2,048/2,048 = 100.000% | 2,048/2,048 = 100.000% |
| **Treatment** | **514/2,048 = 25.098%** | **514/2,048 = 25.098%** |
| Favorable ordinary sequence parser | 205/2,048 = 10.010% | 209/2,048 = 10.205% |
| Storage-order shortcut | 82/2,048 = 4.004% | 209/2,048 = 10.205% |
| Reversed links | 40/2,048 = 1.953% | 152/2,048 = 7.422% |
| Deranged cards | 6/2,048 = 0.293% | 122/2,048 = 5.957% |
| One witness | 26/2,048 = 1.270% | 133/2,048 = 6.494% |
| State reset | 19/2,048 = 0.928% | 140/2,048 = 6.836% |
| Early nil | 22/2,048 = 1.074% | 140/2,048 = 6.836% |

Treatment exact state remains between 21.994% and 26.765% at every evaluated
depth from three through eight. The shuffled-label compiler emits zero exact
graphs. Graph-node reindexing is invariant on 514/514 eligible cases.

## Decisive decomposition

The treatment emits 514 valid graphs. Every one of those graphs is also the
exact semantic graph, has the exact node count and nil halt, and produces the
exact recurrent state and answer. There are **zero valid-but-wrong graphs**:

```text
valid graph       = 514
exact graph       = 514
exact state       = 514
exact answer      = 514
valid but wrong   = 0
```

Therefore S8.1 does not reveal an arithmetic, law-induction, link-traversal,
halt, or recurrent-state failure after successful compilation. Its entire
observed end-to-end deficit lies before execution: the token-role compiler
fails to extract a complete graph from unseen source renderers and nonce names.
Typical failures are roster/state cardinality, missing or duplicate card
witnesses, and non-unique repeated-name matching.

The pointer graph is nevertheless materially better than its favorable global-
rank parser: 25.098% versus 10.010% exact state. This supports model-owned linked
control as the retained execution interface, but it is far below the frozen
95% valid-graph and 90% exact-graph gates. The operation-nonce intervention is
bit-identical on 422/422 mutually valid cases, but 92 originally valid graphs
become invalid after recoding, so the preregistered all-valid eligibility gate
correctly fails.

## Claim boundary and next phase

S8.1 is not promoted and its confirmation board must not be opened. The result
supports only this bounded statement:

> When a learned whole-source compiler emits a valid S8 nil-linked graph, the
> confirmed cyclic substrate executes its model-owned order, halt, state
> transitions, and query exactly on this board.

S9 must not modify or widen that proven executor. It targets the isolated
grounding bottleneck with an occurrence-quotient relational compiler: learn
nonce-span boundaries and sentence-level relations, bind repeated occurrences
by exact emitted surface equality, then decode class-level relation tuples
instead of independently assigning a role to every subtoken. Exact grouping is
an architectural prior; boundaries and semantic relations remain model-owned.
No literature novelty claim is made. A fresh neural board is forbidden until
the quotient representation passes CPU sufficiency, permutation, negative-
control, information-flow, and host-resource audits.
