# R12 S6.2 Contextual Affine Law Neural Development Receipt

**Status:** frozen before development-board seed, board generation, fit, model
access, or score.

This receipt imports the theorem, claim boundary, law split, controls, and gates
from `R12_S6_CONTEXTUAL_AFFINE_LAW_INDUCTION_PREREG.md`, the sole scoreless
split repair from v1.1, and the passing CPU mechanics artifact at SHA-256
`a31a232c83a53d0b7aff87b4a495abd6740d98589059325951e2e4688e2bded6`.

## Frozen Development Data

After the source, tests, and this receipt are committed, draw exactly one board
seed. Build:

- 961 unique atomic training cells: every position of every admitted training
  law at moduli 5, 7, and 11;
- 2,048 balanced primary development programs over modulus x depth cells for
  moduli 5/7/11 and depths 3--8; and
- 512 modulus-13 scale-diagnostic programs.

Every program uses at least two held-out development laws, depth 3--8, a random
initial identity permutation, arbitrary nonce operation names, and a late
position query. Files must contain no confirmation programs. The treatment input
is exactly `(modulus, card_y0, card_y1, current_location)`; `control_law_id` is
visible only to the matched memorizer.

## Frozen Architecture

Treatment `ContextualAffineLawInducer`:

- four categorical tokens: `LAW`, `SUPPORT_0`, `SUPPORT_1`, `QUERY`;
- width 256, six pre-norm Transformer encoder layers;
- eight attention heads, feed-forward width 1,024, GELU, zero dropout;
- learned role, modulus, input-coordinate, and output-coordinate embeddings;
- one 13-way destination head with a hard modulus mask;
- **4,753,677** trainable parameters;
- **138,448,546** total parameters with the promoted bounded Shohin stack.

The favorable `LawIdMemorizer` uses the same transformer plus a 104-entry law
embedding and has **4,780,301** trainable parameters. Training law IDs are unique;
every development law receives the same OOV ID. The control has more parameters
than treatment and identical update count, batch stream, optimizer, and device.

## Frozen Optimization

Draw one training seed after the implementation commit. Both arms use:

- AdamW;
- 4,000 updates;
- batch size 256 sampled with replacement from the 961 atomic rows;
- learning rate `5e-4`;
- weight decay `0.01`;
- gradient-norm clip `1.0`; and
- the same sampled row-index stream.

Both arms must reach at least 99% exact atomic training accuracy before the sole
development read. Failure closes S6 without optimizer, width, epoch, seed, or
data repair.

## Sole Development Access

One serial H100 job fits both atomic arms, writes one immutable checkpoint, reads
the primary and diagnostic development boards exactly once, writes one
evaluation, and applies the already-frozen assessor. No retry may reuse the same
board after a model or score is produced. Infrastructure failure before a valid
checkpoint or development read may be documented and retired, but cannot alter
the mechanism or thresholds.

The evaluator reports host, treatment, deranged-card, one-witness, state-reset,
and OOV law-ID arms; every depth; the multi-law stratum; nonce-name invariance;
all held-out atomic law cells; and modulus-13 diagnostic accuracy. Confirmation
generation remains forbidden unless the assessor records
`qualify_s6_for_one_confirmation` with every gate true.

