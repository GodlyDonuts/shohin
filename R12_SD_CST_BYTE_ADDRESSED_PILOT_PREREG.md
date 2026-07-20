# R12 SD-CST Byte-Addressed Compiler Training Pilot

**Status:** frozen training-only architecture admission; no development or
confirmation access is authorized

## Diagnosis

SD-CST v1.1's generic residual-plus-slot compiler did not learn the program
fields. On the consumed 48,000-row training split it emitted zero STOPs in every
row, constrained STOP position remained at 16.6625%, initial-state exactness was
16.9458%, identity and amount cells were at chance, and only the isolated late
query learned. An exactly-one-STOP decoder would therefore conceal the failure.

The next falsifiable hypothesis is that the compiler needs an explicit learned
evidence address bus before categorical source deletion. It must first localize
the binding clause and each semantically numbered event clause, then emit the
same private categorical packet as SD-CST.

## Frozen pilot

The pilot reads only the already-consumed 48,000-row outcome-free training
split. Rows are ordered by `sha256(row_id)`; the first 40,000 are fit rows and
the remaining 8,000 are a held-out training partition. It cannot open the
development or confirmation filenames.

The compiler consumes UTF-8 bytes, not board metadata. It has:

- byte and absolute-position embeddings;
- six 384-wide, eight-head source self-attention layers with 1,536-wide MLPs;
- one binding/eight event learned address queries;
- model pointer logits over source bytes, supervised only to place probability
  inside the correct source line;
- two 384-wide slot-mixing layers with 1,024-wide MLPs;
- unchanged initial-state, event-kind, entity-role, amount, and late-query
  categorical heads; and
- no state, answer, trajectory, executor, repair, reward, development, or
  confirmation signal.

Pointer supervision is compiler localization, not host inference: at test time
the model's pointer-weighted source values alone form the slots. No target line,
span, entity, or operation is supplied to inference, and all source bytes and
pointer state are deleted when categorical outputs are emitted.

The byte compiler has 14,206,993 parameters. With the frozen 125,081,664 trunk,
19,206-parameter motor, and 835-parameter reader, the complete system is
139,308,698 parameters, 10,691,302 below the strict 150M cap.

## Frozen optimization

- seed drawn only after source commit;
- four epochs, batch 64, AdamW lr `3e-4`, betas `(0.9, 0.95)`, weight decay
  `0.01`, 100-update warmup, cosine decay to zero, gradient clip `1.0`;
- field losses are initial + kind + non-STOP identity + non-STOP amount + late
  query;
- kind class weights `[1, 1, 4]` counter the public one-STOP/seven-action
  grammar;
- line-address loss has weight `2.0`; and
- evaluation reports both raw independent kinds and exact global MAP under the
  public exactly-one-STOP grammar. The MAP decoder sees model kind logits only.

## Admission gates

The pilot advances the architecture, but makes no reasoning claim, only if all
held-out training-partition gates pass:

1. all nine pointer slots inside their correct source lines on at least 90%;
2. initial-state exactness at least 80%;
3. constrained event-kind exactness at least 90%;
4. entity-role exactness at least 80%;
5. amount exactness at least 90%;
6. late-query exactness at least 98%;
7. complete constrained tape exactness at least 60%;
8. complete system strictly below 150M; and
9. development/confirmation access exactly `0/0`.

Failure below every semantic-field gate rejects this compiler. A partial result
may justify one preregistered training-only ablation, but never a scored board.
Only a full gate pass may advance to independent mechanics/control tests and a
future post-commit fresh board.
