# R12 S3 Categorical Permutation Register Result

**Decision:** `reject_s3_categorical_register`

The exact categorical state representation is valid, but v1's neural update
cell is not permutation-equivariant and fails immediately outside the identity
state seen during atomic training.

## Custody

- Source/prereg commit `dc7c13e` preceded fit and every S3 score.
- Job `693127` completed once on H100 `evc29` in 5m39s, exit `0:0`.
- Training used 96,000 public rows / 192,000 independent atomic targets / 1,517
  updates / one epoch / seed `2026071903`.
- The 717,323-parameter executor kept the full system at 134,406,873
  parameters and recorded zero confirmation access.
- Training completed in 56.15 seconds; final executor state SHA-256:
  `8f043863138d32a89089ceaf029f89e2806d1f1c2df4c2735bf8bb1fb491b161`.
- Assessment SHA-256:
  `c9c7b545dfc85d614faaf1e943fb7932c6e54c80679bdb3ad735a09825418fdf`.

## Results

| Evaluation | Answers | Exact state | All transitions | Entity match |
|---|---:|---:|---:|---:|
| Two-step mean | 79.590% | 66.211% | 66.211% | 99.927% |
| Two-step ordered | 79.541% | 66.260% | 66.260% | 100.000% |
| Two-step gold | 79.541% | 66.260% | 66.260% | 100.000% |
| Lexical-OOD mean | 58.984% | 48.730% | 42.236% | 86.377% |
| Depth 3--8 mean | 53.906% | 41.260% | 17.627% | 71.879% |
| Depth 3--8 ordered | 54.834% | 42.285% | 18.750% | 73.551% |
| Depth 3--8 gold | 54.932% | 42.432% | 18.896% | 73.871% |

Depth-eight mean answers are 48.529%. Every substantive promotion gate fails.

## Mechanistic diagnosis

Atomic transition loss reaches effectively zero and the forward register is
always a valid S3 element. Identity is also not the problem: ordered and gold
are indistinguishable at two steps, while entity match is exact. The defect is
the update cell's input contract. It receives the complete assignment matrix
and immutable identity vector. Atomic training presents only the identity
assignment; the second recurrent call presents a non-identity assignment that
is out of distribution. The unrestricted MLP therefore memorizes the initial
coordinate frame rather than a local group action.

The only bounded repair is **equivariant local action**: compute current
location by multiplying register and categorical identity, then let the tied
cell depend only on location, operation kind, and amount. Those variables
fully determine the relative move permutation and have complete atomic support.
The raw assignment and immutable identity must not enter the transition MLP.
This is a separately preregistered architecture version, not a retry or width/
epoch/data change.

No confirmation, autonomous planning, learned halt, language reasoning, or
novelty claim is authorized.
