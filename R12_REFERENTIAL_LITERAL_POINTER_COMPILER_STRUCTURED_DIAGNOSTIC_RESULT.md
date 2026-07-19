# R12 Structured Complete-Compiler Diagnostic Result

**Protocol:** `r12_referential_literal_pointer_compiler_v1_2_structured_development`
**Decision:** **REJECT AS AN INTEGRATED COMPILER; RETAIN STRUCTURAL EFFECT**

## Run identity

Job `692983` completed on a verified H100 PCIe on `evc25` with exit code zero in 13m28s. The fit
used the frozen v1.1 train/development bytes, one epoch, 1,514 updates, seed `2026071804`, and the
frozen optimizer schedule. Shohin remained frozen.

| Item | Value |
|---|---|
| adapter parameters | 6,402,701 |
| total parameters | 131,484,365 |
| fit elapsed | 555.566 seconds |
| initial adapter-state SHA-256 | `78add721586562cac4418fc539d8495737d772260354532a7f59f8b467ecfe15` |
| final adapter-state SHA-256 | `dd5d5b6a8c7d300c3fe4016098795a8caa9d671ed72a1a25a148e8f51de6be5c` |
| adapter file SHA-256 | `8d2278c369ead039a48bc39d8f9effab7198a5cecfabaf01e3724eec4ec3aa11` |
| development result SHA-256 | `0f237c052102955c26cc14340d16fe2139a020cac959ac27720f2faf22121d03` |
| log SHA-256 | `822c646fd55abaecbe80bb805bee8c6eb99dfcfeb309e15a23aa28e359985293` |

All three artifacts are hash-matched between Newton and the Mac. Confirmation access is zero.

## Frozen result

| Metric | v1.1 free slots | v1.2 structured | Change |
|---|---:|---:|---:|
| initial-state joint exact | 18.848% | **48.340%** | **+29.492pp** |
| full pointer exact | 2.197% | **0%** | -2.197pp |
| semantic-program exact | 15.283% | **0%** | -15.283pp |
| answer accuracy | 29.395% | **18.994%** | -10.401pp |
| operation-kind accuracy | 96.265% | **49.927%** | -46.338pp |
| canonical + paraphrase both exact | 0/512 | **0/512** | 0 |

The structural field improves ordered initial binding materially. On the paraphrase renderer,
initial-state joint rises to 60.547%, operation-0 joint to 40.625%, and operation-1 joint to 38.672%.
But the operation-kind loss remains at chance throughout training, so no semantic program is exact.

## Mechanism diagnosis

The role head can minimize pointer and role losses without forcing the free operation slot to read
the selected kind token. Structural location and semantic classification became disconnected
parameter paths. This is a wiring failure, not evidence against bidirectional parsing.

A post-hoc diagnostic composes v1.2 pointer predictions with the independently trained v1.1 kind
predictions, without fitting new weights. It reaches:

| Hybrid metric | Result |
|---|---:|
| answers | **987/2,048 = 48.193%** |
| semantic programs | **691/2,048 = 33.740%** |
| full pointer exact | **224/2,048 = 10.938%** |
| paraphrase answers | **270/512 = 52.734%** |
| paraphrase semantic programs | **224/512 = 43.750%** |
| all-four answers correct | **114/512** |

This hybrid is an external diagnostic, not an autonomous compiler. It supplies evidence that the
structural and semantic gains coexist when their parameter paths are isolated.

## Consequence

Reject v1.2 as an integrated compiler. A minimal successor may give structural roles and operation
semantics independent memory projections/decoders while retaining the same source-only inference
boundary. Passing exposed development can authorize only a fresh board and matched controls; v1.1
confirmation remains sealed.
