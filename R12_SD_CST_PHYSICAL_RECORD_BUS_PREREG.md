# R12 SD-CST Physical-Record Write-Bus Preregistration

**Status:** exact scientific source frozen before seed and H100 execution;
consumed-training mechanics only; no scored split may be opened

**Source contract:** exact architecture, pilot, test, job, and preregistration
commit `5c9a2855a202692996e6e4100c927e9d8842bf48`; the following documentation-only
commits do not change the scientific contract. After source push, raw 64-bit
beacon `18183044536483492966` was reduced modulo `2^63` to sole scientific seed
`8959672499628717158`.

**Parent:** rejected joint renderer-memory/native-decoder checkpoint SHA-256
`4b842e4c2d0d608c32f0fd113b404866be7269676084cdac9b1a00d43cdd298d`

**Claim class:** favorable conventional compiler-mechanics control; no native
reasoning, primitive novelty, broad generalization, or Shohin promotion claim

## 1. Fixed diagnosis

The joint global-query compiler did learn, but its moderate local errors
multiplied into zero complete packets. The exact post-hoc audit over all 48,000
fit and 8,000 held-out consumed rows found the following held-out
initialization-to-endpoint changes:

| Local quantity | Initialization | Endpoint |
|---|---:|---:|
| physical source line, per slot | 10.896% | 42.029% |
| event address, active slot | 6.555% | 25.466% |
| event kind, per slot | 40.719% | 55.731% |
| amount, active slot | 49.836% | 68.000% |
| identity, active slot | 41.168% | 50.325% |

Fit and held-out rates differ by at most 0.25 percentage points, ruling out a
renderer-parity overfit diagnosis. Gold source-line pooling raises kind to
73.641% but leaves amount at 68.134%. Gold event-span pooling raises identity
to exactly 100% over all 56,000 active slots. Therefore:

1. batching, gradients, and objectives are functioning;
2. the frozen declaration fingerprint matcher is sufficient after correct
   event localization;
3. the dominant unresolved mechanism is physical record/address
   factorization plus local field extraction; and
4. widening, adding epochs, or adding layers to the failed global-query
   contract is forbidden.

## 2. Distinct falsifier

The source has exactly nine newline-delimited physical records: one declaration
record and eight event records. The falsifier applies the following fixed
mechanism:

1. segment records only at observed newline bytes;
2. encode each record independently with shared relative positions and shared
   weights;
3. contextualize the resulting unordered nine-record set;
4. emit model logits assigning physical records to the nine semantic slots;
5. use local field motors to emit kind and amount from the assigned record;
6. use a local entity pointer inside each physical record, then reuse the
   frozen exact declaration fingerprint matcher; and
7. leave the already-successful declaration binding, initial-state transport,
   late query, categorical tape, executor, motor, reader, and Shohin trunk
   frozen.

The treatment normalizes assignment logits with 8 Sinkhorn row/column passes
during training and uses one deterministic greedy one-to-one assignment during
evaluation. The optimizer cannot inspect labels, target spans, tape validity,
executor state, answers, or retry feedback when choosing that assignment.
Greedy assignment is not represented as Hungarian or globally optimal MAP.

The delimiter and fixed record cardinality are explicit finite-grammar priors.
This is a conventional structured parser control, not a proposed reasoning
primitive.

## 3. Matched control

The sole run trains two arms serially:

| Arm | Assignment |
|---|---|
| `constrained` | Sinkhorn soft assignment in training; greedy one-to-one at evaluation |
| `independent` | independent physical-record softmax per semantic slot in training; independent argmax at evaluation |

Both arms:

- reconstruct the same exact parent checkpoint;
- receive byte-identical initial values for every new parameter;
- have identical parameter names, count, optimizer, data order, epochs,
  updates, losses, and random seed;
- differ only in assignment normalization and hard decoding; and
- retain a byte digest over every excluded parent tensor.

If the treatment reaches the absolute compiler gates but fails the frozen
five-point differential gates, the physical-record architecture may be
retained only as a conventional baseline and one-to-one attribution is
rejected.

## 4. Frozen architecture and parameter certificate

New trainable modules are exactly the 88 names beginning `record_`:

- byte and relative-position embeddings at width 384;
- four shared local-record Transformer layers, six heads, FFN 1,536;
- two record-set Transformer layers, six heads, FFN 1,536;
- record/set normalizations, a nine-role head and role embeddings;
- three-way kind and two-way amount motors; and
- local entity query/key projections.

Maximum local record width is 144 bytes. The audit over fit and held-out
renderer orbits covers all 56,000 rendered rows and observed zero cardinality
violations, a maximum 132-byte payload, and a maximum 133-byte compiler record
after retaining the newline delimiter. Overlength, empty, or non-nine-record
inputs fail closed.

| Quantity | Exact count |
|---|---:|
| immutable Shohin trunk | 125,081,664 |
| complete compiler, including frozen parent | 65,831,689 |
| new trainable parameters | 11,106,830 |
| categorical motor | 2,781 |
| categorical reader | 17,260 |
| **complete deployed system** | **190,933,394** |
| **strict-200M headroom** | **9,066,606** |

The complete deployed system must remain strictly below 200,000,000. No
parameter is added after this freeze. Historical sub-150M contracts remain
unchanged; this is a new user-authorized sub-200M experiment.

## 5. Data and optimization

The experiment reuses only the already-consumed projected-v2 training JSONL,
SHA-256
`b7756dbf8d4401dbc5fb897dee53f68758e27200b1ce0d2387631f2f0205ec25`.
It partitions 12,000 latent programs for fit and 2,000 disjoint latent programs
for held-out renderer-orbit mechanics using the existing deterministic ID hash.
Fit uses the even-parity renderer orbit; heldout uses the odd-parity renderer
orbit. This is adaptive training-only development, not fresh generalization.

Each arm uses exactly:

- two epochs / 3,000 optimizer updates;
- family batch size 8 and evaluation family batch size 16;
- AdamW, lr `2e-4`, betas `(0.9, 0.95)`, weight decay `0.01`;
- 100-update warmup and cosine decay;
- gradient clipping at `1.0`; and
- renderer consistency weight `1.0`.

No development, confirmation, answer, state, trajectory, or executor output is
reachable. The sole output directory must not preexist. No failed arm may be
restarted, extended, widened, or rescored under this contract.

## 6. Frozen absolute gates

For every held-out renderer, the constrained arm must reach:

1. initial state at least 95%;
2. query and query pointer at least 99%;
3. declaration and initial-occurrence pointers at least 99%;
4. all-nine physical-line pointers at least 95%;
5. active event-occurrence pointers at least 90%;
6. complete event kind at least 95%;
7. complete active identity at least 90%;
8. complete active amount at least 95%; and
9. complete packet at least 80%.

Every fit renderer must also reach line pointer at least 99%, event pointer at
least 99%, and complete packet at least 95%. Both excluded-parent digests must
remain byte-identical, arm parameter certificates must match, complete system
size must remain strictly below 200M, and scored access must be `0/0`.

## 7. Frozen attribution gates and decisions

On the minimum held-out-renderer rate, constrained must beat independent by at
least five percentage points independently for:

1. complete packets;
2. all-nine physical-line pointers; and
3. active event-occurrence pointers.

The decision is fixed:

- all absolute and attribution gates pass:
  `retain_physical_record_bus_and_one_to_one_assignment`;
- all absolute gates pass but any attribution gate fails:
  `retain_physical_record_bus_reject_one_to_one_attribution`;
- any absolute, preservation, parameter, or access gate fails:
  `reject_or_revise_physical_record_bus`.

No threshold may change after the seed or output exists.

## 8. Honest boundary

A pass establishes only that explicit delimiter-bounded physical
factorization, local extraction, and model-owned record assignment can compile
this finite renderer orbit under the sub-200M cap. It does not establish source
deletion by itself, general language parsing, self-selected programs, learned
halting, new algorithms, native reasoning, or state reuse. Advancement to a
fresh scored board would require a separate post-commit preregistration and new
data; current development and sealed confirmation remain unopened.
