# R12 SD-CST Joint Renderer-Memory Program Decoder Preregistration

**Status:** implemented and locally audited before source freeze, seed, H100
fit, or scored access

**Parent:** exact rejected Renderer-Orbit v1.2 checkpoint

**Claim class:** favorable joint conventional compiler control on consumed
training rows; no primitive novelty or reasoning claim

## 1. Fixed diagnosis

The head-only renderer-native decoder in job `694073` preserves v1.2 exactly
but learns 0% fit line/event pointers and packets. Its line loss stays near
uniform and kind loss stays near chance. This means the frozen orbit memory does
not expose program structure in a form the new decoder can use. More decoder
epochs are forbidden.

The smallest next control co-adapts shared renderer memory and the decoder. It
does not alter the categorical tape or executor and adds no parameters beyond
the rejected head-only model.

## 2. Trainability and resources

Initialization is the exact v1.2 checkpoint SHA `2e019b81...`; native decoder
parameters are freshly initialized under the post-commit seed. Trainable names
are exactly:

- all 35 `native_*` decoder parameters;
- orbit byte and position embeddings;
- all eight orbit encoder layers; and
- orbit final normalization.

The residual projection and scalar gate, ordinal query/pointer/value motors,
all binding machinery, exact packet heads, categorical executor, motor, reader,
and Shohin trunk remain frozen. Query, binding, initial, and packet losses still
backpropagate through shared memory, providing preservation constraints.

| Quantity | Count |
|---|---:|
| complete compiler | 54,724,859 |
| trainable parameters | 32,782,853 |
| trainable tensor names | 135 |
| complete deployed system | 179,826,564 |
| strict-200M headroom | 20,173,436 |

An exact loaded-parent consumed-row backward pass has finite gradients in the
shared encoder and native decoder, no gradient in the frozen ordinal motor, and
a frozen-state digest over every excluded tensor.

## 3. Data and optimization

The pilot reuses the already-consumed 12,000-even / 2,000-odd renderer orbit.
This is adaptive mechanism development, not a fresh generalization result. No
development, confirmation, answer, state, or trajectory is reachable.

Optimization remains two epochs / 3,000 updates, family batch eight, AdamW lr
`2e-4`, betas `(0.9, 0.95)`, weight decay `0.01`, 100-step warmup, cosine decay,
clip `1.0`, and renderer consistency weight `1.0`. Event address/identity loss
uses the already-frozen support-safe curriculum; final fit event pointers must
reach 99%.

## 4. Gates

Every held-out renderer must reach:

1. initial state at least 95%;
2. complete kind at least 95%;
3. complete active identity at least 90%;
4. complete active amount at least 95%;
5. query and query pointer each at least 99%;
6. declaration and initial-occurrence pointers each at least 99%;
7. source-line pointers at least 95%;
8. event-occurrence pointers at least 90%;
9. complete packet at least 80%; and
10. every fit renderer event pointer at least 99%.

The frozen excluded-state digest must match, complete parameters must remain
strictly below 200M, and scored access must be `0/0`. No threshold or epoch may
change after output.

## 5. Honest boundary

Joint encoder/decoder training is ordinary representation learning. It is a
favorable conventional parser control under the R12 invention charter, not a
new reasoning primitive. A pass only establishes that the finite renderer
orbit is learnable under the parameter cap and permits this exact architecture
as a baseline in a separately committed fresh-board experiment. A failure
closes joint co-adaptation without a larger/longer retry.
