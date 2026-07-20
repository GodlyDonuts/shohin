# R12 S9.2 Global Anchor Closure Development Result

**Decision:** rejected on the sole fresh development read

**Confirmation:** unopened and permanently ineligible for S9.2 v1

**Slurm job:** `693890`, `evc25`, completed `0:0` in 59m39s

## Frozen identity

| Item | Identity |
|---|---|
| Scientific source commit | `38c934cf9f360e1fd13258c23be310e948cafba1` |
| Board seed | `3823077847356570601` |
| Training seed | `1277007704479652588` |
| Board report | `f22401e82690f8240abe89d6083e3a387619243bc68bb9d9e380540d90b1899e` |
| Base checkpoint | `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6` |
| S8.1 initializer | `44b3291555047085257cfb1c4ec03dd6e5485ce83e134a5200d8ea0055614585` |
| Tokenizer | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| Development split | `a186df62c9fe030d71f8c3734e6e6d570667c2c2ddc340d81718813258b96550` |
| Sealed confirmation | `84be3808f1f385740edf6acb8129985075e90506e31a2d087d8f34a79703c054` |
| Checkpoint | `590a3944c7af5bf7b27b4a27bddbe159f7e4f8b3fc7fac06bfa673d33c647918` |
| Evaluation | `0ef0523cd847bb32e0f5ba9096828e21d0ac22b095880595d1b6ac89e908685f` |
| Assessment | `58c3ca008764708d3f50301bee82bcd332aec939b3b6457f4abeeae7b5183bf9` |
| Development access ledger | `35effa66f661ca0ba7e407b0f05599ccaeabd23f3ce901c8e00ea2e71c7bc92a` |

All mirrored result artifacts hash-match Newton. Development/confirmation
access is exactly `1/0`. The confirmation split was not opened.

## Result

| Arm/decoder | Exact graph | Exact state | Exact answer |
|---|---:|---:|---:|
| S9.2 treatment | 340/2,048 = **16.602%** | 340/2,048 = **16.602%** | 340/2,048 = **16.602%** |
| Positive-orbit-only | 340/2,048 = 16.602% | 340/2,048 = 16.602% | 340/2,048 = 16.602% |
| No-class-message | 2,019/2,048 = **98.584%** | 2,022/2,048 = **98.730%** | 2,022/2,048 = **98.730%** |
| Layout-only | 544/2,048 = **26.562%** | 609/2,048 = 29.736% | 619/2,048 = 30.225% |
| Paired-shuffled | 0/2,048 | 0/2,048 | 0/2,048 |
| Same-logit local-root decoder | **2,038/2,048 = 99.512%** | n/a | n/a |
| Unconstrained decoder | **2,035/2,048 = 99.365%** | n/a | n/a |
| Uniform/source-free | 0/2,048 | n/a | n/a |

The assessment passes 21/43 gates: 14/31 inherited and 7/12 S9.2-specific.
This is not a threshold miss. It fails every primary exactness, depth, causal,
root, and treatment-over-control gate.

## Failure localization

Treatment state/answer accuracy by depth is:

| Depth | Correct |
|---|---:|
| 3 | 0/342 |
| 4 | 0/341 |
| 5 | 0/341 |
| 6 | 0/342 |
| 7 | 0/342 |
| 8 | **340/340** |

The treatment predicts card count almost perfectly (`2,047/2,048`) and modulus
perfectly, but predicts the complete `(m,c,d)` count tuple and root spans on only
`340/2,048`. Its success set is exactly the depth-eight stratum. The positive-
orbit-only arm has the same pattern. In contrast, removing occurrence-class
messages restores `2,019/2,048` exact graphs and uniformly high state accuracy
across every depth (`98.24%--99.71%`).

The evidence therefore rejects the S9.2 hypothesis. Global interval Viterbi is
mechanically correct, but the class-message logits it optimizes encode a
training-cardinality shortcut. The finite grammar turns that shortcut into an
irrevocable maximum-depth assignment. It destroys a same-logit local decision
that was already `99.512%` exact. The layout arm's `26.562%` exact graph score
also demonstrates substantial positional recoverability in this board.

The failure is not in the frozen S7/S8 executor, graph semantics, or recurrent
state transition. Every one of the 340 valid treatment graphs is exact, and all
340 execute to exact state and answer. Operation recoding, class reindexing,
relation-storage reindexing, state, and answer are bit-identical on all 340
eligible rows. The failure is selection before graph construction.

## Scientific disposition

1. Close S9.2 v1. Do not rescore, retune, or open confirmation.
2. Retain the no-class/local decoder only as a bounded compiler diagnostic. It
   does not become a promoted result because the preregistered treatment failed
   and the layout control is high.
3. Retire parser-only anchor optimization as the active reasoning frontier.
4. Use same-layout semantic counterfactuals and stronger grammar firewalls in
   any future language compiler claim.
5. Move the primary effort to a model-owned state experiment that deletes
   source access after compilation and causally tests recurrent state transport,
   tied update, query consumption, and halt.

This result narrows the problem: Shohin's frozen features can support nearly
exact bounded parsing when the harmful class-message path is removed, but S9.2
does not create reasoning and does not improve the compiler.
