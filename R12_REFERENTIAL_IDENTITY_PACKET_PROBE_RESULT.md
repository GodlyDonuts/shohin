# R12 Referential Identity Packet Probe Result

**Decision:** admit the lexical set-valued packet as the only bounded RGDE v1.1
repair. This is a no-fit carrier result, not an executor or reasoning result.

## Frozen run

- Job `693117` completed `0:0` on H100 `evc29` in 14 seconds.
- It used the frozen raw-300k base, qualified ordinary compiler, and 2,048-row
  public compositional development split.
- It performed no optimizer update, state transition, answer prediction,
  confirmation access, retry, or arm selection.
- Result SHA-256:
  `dcc16fa3101e403a5cd2452171511fe9f4497c5c879ca1fa65da0e31ba615f60`.
- Log SHA-256:
  `9eefc04dfedfe65b9e4d221c85a4ae041bcede71dc8654041c293ffef3cccd55`.

## Result

The probe asks whether each operation entity is nearest to the matching one of
the three initial entities, using only the gathered vectors.

| Carrier | Correct | Accuracy |
|---|---:|---:|
| contextual state + pointer softmax (RGDE v1) | 1,312/4,096 | 32.031% |
| frozen token embedding + pointer softmax | 2,966/4,096 | 72.412% |
| **frozen token embedding + normalized sigmoid role span** | **4,090/4,096** | **99.854%** |
| frozen token embedding + gold span | 4,096/4,096 | 100.000% |

The admitted carrier scores 99.805% on canonical, binding-twin, and order-twin
surfaces and 100% on paraphrase. The v1 compiler had already selected a token
inside the correct span 99.982% of the time, but softmax chose the same single
subtoken across the two entity occurrences only 59.326% of the time. The new
carrier interprets a role as a set: sigmoid each role logit, mask invalid
tokens, normalize over the set, and average the frozen vocabulary embeddings.

## Interpretation

The failed RGDE v1 did not establish that source deletion destroys entity
identity. It established that a categorical one-token gather is the wrong
interface for multi-token referents. A zero-parameter set-valued lexical
channel nearly saturates the no-fit identity test while leaving operation
semantics in the contextual channel.

This authorizes one frozen v1.1 executor experiment. It does not authorize old
confirmation access, broader fitting, a natural-language claim, or a novelty
claim.
