# R12 S7 Learned Cayley Law Compiler: Board Receipt

**Date:** 2026-07-19  
**Decision:** `admit_s7_learned_cayley_board`  
**Source commit:** `b9a9414`  
**Board seed:** `4905719171551557987`  
**Frozen training seed:** `1314309421681697406`

## Contents

| File | Rows | SHA-256 |
|---|---:|---|
| `generator_train.jsonl` | 23 | `492fc927e8f172f05be8175b487c8e5b0ec91ede267034135bc6e80e36f0ad44` |
| `transformer_atomic_train.jsonl` | 984 | `f0a401db2df0cb641b22fea88d6b321d56612029385d0e239f8b0d654a64748c` |
| `atomic_development.jsonl` | 150 | `7263d3bfa2fa70ee35935f595b8eb9e496d77335ecdfd2945cb33f4861991af3` |
| `development.jsonl` | 2,048 | `19baa8c3e8b4cfb441dac24f40cb069f0d00600d49ec9a163ba7f020af47e70f` |
| `confirmation.sealed.jsonl` | 2,048 | `c2eb8d5c5dd285dfcb60389c3067c4842e47872d64b5233681c32c8542434bc5` |

Report SHA-256:
`2a471f3bc0da129a890878c802b587417fdbc756efa84d4d449f50374c92f306`.

## Law custody

S6 development laws are closed and do not score S7. For each modulus, the S7
train split is the old training pool plus identity/successor anchors. The
never-read S6 reserved-confirmation pool is split before row generation:

| Modulus | Train laws | S7 development laws | S7 confirmation laws | Closed S6 laws excluded |
|---:|---:|---:|---:|---:|
| 5 | 11 | 3 | 3 | 3 |
| 7 | 29 | 2 | 3 | 8 |
| 11 | 66 | 11 | 12 | 21 |

Train, development, and confirmation law sets are disjoint. Every development
and confirmation program uses at least two distinct laws. Development is
balanced over all 18 modulus/depth cells at 113 or 114 rows each.

## Hidden coordinate custody

Each modulus uses a post-commit random observed-symbol permutation. Raw
permutations are learnable only from the 23 successor cells and three zero
anchors; the report binds them without exposing them as model metadata:

- modulus 5 binding hash: `622b144a203299d37ac5ed5221218a3c01be61a36b87fd1260d42cb9fde42aad`
- modulus 7 binding hash: `5e5217244dedc14b5f3a34d6ccdd1a02e18ea8ef199b490fed81494d5c6c5eb1`
- modulus 11 binding hash: `1a015f68d74ee061c2a67ac9fb7b62851d06c73340615701c8214ffa81afabbc`

Generator and transformer training rows contain no slope, intercept, final
state, or answer fields. Treatment receives no train-law card except what is
implied by successor/zero anchors. The favorable transformer receives all 984
train-law atomic cells.

## Access state

- Development accesses: 0
- Confirmation accesses: 0
- Neural checkpoints: none
- Neural scores: none

Commit all board bytes and this receipt before synchronization or submission.
The development wrapper may read only `development.jsonl` and
`atomic_development.jsonl`. The confirmation file remains sealed unless the
immutable development assessor qualifies every gate.
