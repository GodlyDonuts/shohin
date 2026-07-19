# R12 Referential Gather-Delete Permutation Executor Result

**Decision:** reject v1; do not generate confirmation

## Bottom line

The typed permutation state, source-deletion boundary, and recurrent call are
mechanically sound, but the v1 packet does not carry stable entity identity.
The primary tied treatment reaches only **48.340% answers**, **18.701% exact
final assignments**, and **17.236% both-transition exact**. A favorable untied
cell is tied at 48.438% answers, and a gold-pointer/kind training arm remains at
49.170%. No Stage-B promotion gate is close.

The failure is localized. Query and amount classification are 99.707% and
99.780%, respectively, while operation-entity matching is only 51.294%. The
compiler almost always points inside the correct multi-token span, but v1
softmax gathering collapses each span to one contextual subtoken. Across the
16,384 operation-entity references on the fresh qualification board:

- both selected positions lie inside their correct spans in 16,381/16,384 =
  **99.9817%**;
- selected operation/intro token IDs agree in only 9,720/16,384 = **59.3262%**;
- the complete operation/intro token-ID sequences agree in
  16,384/16,384 = **100%**.

Thus the next bounded repair is not more recurrence or capacity. It is a
set-valued, vocabulary-aligned identity packet that preserves the entire
selected span instead of sampling one contextual coordinate.

## Frozen execution

All four jobs used commit `d69250f`, raw-300k, the qualified ordinary compiler,
factorized train/development, seed `2026071901`, one epoch, and zero
confirmation access.

| Job | Arm | Node | Elapsed | Exit |
|---|---|---|---:|---:|
| `693111` | tied predicted packet | `evc28` | 2m17s | `0:0` |
| `693112` | untied predicted packet | `evc29` | 4m05s | `0:0` |
| `693113` | tied gold packet | `evc33` GPU 0 | 4m36s | `0:0` |
| `693114` | source-retained direct control | `evc33` GPU 1 | 3m50s | `0:0` |

The two `evc33` jobs used distinct H100 UUIDs. Base and compiler had zero
trainable parameters. Treatment training contained 192,000 atomic operation
targets and no full two-step state or answer target.

## Development scores

| Arm/intervention | Answer | Final assignment | Both transitions | Query | Entity match |
|---|---:|---:|---:|---:|---:|
| tied predicted | **48.340%** | **18.701%** | **17.236%** | 99.707% | 51.294% |
| untied predicted | 48.438% | 17.480% | 16.113% | 99.707% | 51.221% |
| tied, no-fit gold rescore | 45.508% | 19.727% | 15.625% | 99.707% | 46.582% |
| tied trained/evaluated gold | 49.170% | 20.166% | 19.385% | 99.902% | 53.760% |
| tied operation shuffle | 31.104% | 11.084% | 5.078% | 99.707% | 36.157% |
| tied query shuffle | 43.066% | 18.701% | 17.236% | 67.188% | 51.294% |
| source-retained direct | 37.988% | n/a | n/a | n/a | n/a |

Tied per-surface answer accuracy is binding twin 56.836%, canonical 46.289%,
order twin 41.602%, and paraphrase 48.633%. Only 24/512 quartets have all four
answers correct; one quartet has all four final assignments exact and none has
all four complete transition traces exact.

## Frozen-gate assessment

The source-retained 95% ceiling fails. Every gold/treatment capability floor
fails. Every per-surface/all-four floor fails. The operation and query shuffles
fall below their 45% ceilings, but treatment margins are only 17.236 and 5.273
percentage points, not the required 40. The 2,047/2,048 intervention coverage
passes. Tied remains within two points of untied and uses fewer parameters, but
both are weak. Identity/custody gates pass.

The failure is downstream of pointer accuracy because gold packet training and
rescoring do not rescue it. It is not caused by lack of a second cell because
untied does not improve it. The source-retained decoder also fails its ceiling,
so it is not a useful positive model for this representation.

## Consequence

V1 is preserved as a conventional negative baseline. Do not tune it, add
epochs, increase width, generate confirmation, or claim source-deleted
reasoning. The one admissible repair changes only the packet identity channel:

1. expose frozen vocabulary embedding states alongside contextual compiler
   memory;
2. use normalized sigmoid role masks, which preserve all tokens selected by
   the compiler's existing multi-token role supervision;
3. encode entity/literal/query identity from the complete vocabulary-aligned
   span, while retaining contextual operation-kind information separately;
4. repeat the same atomic-only training, two-step evaluation, favorable arms,
   and causal shuffles under a new preregistration.

This repair is inspired by the measured failure, not a post-hoc rescue of v1's
claim. It must freeze new source and gates before fitting.
