# R12 Projected SD-CST Fresh v2 Preregistration

**Status:** first unlaunched v2 board rejected by cross-generation audit;
successor source hardening in progress with no replacement seed or scored read

## Parent result and exact failure

Fresh v1 source `4a7fb4880c919735ae35bf1f33f4c7245a8bff73`, board seed
`3040523197183361035`, training seed `8787815392344128274`, and H100 job
`694008` produced a valid fixed epoch-four checkpoint. Treatment was exact on
48,000/48,000 training tapes. The equal-update row-shuffled-label arm was exact
on 870/48,000 tapes, 1,298/48,000 identities, and 185/48,000 binding pointers.
The checkpoint and gate configuration SHA-256 values are `91d4860b...` and
`a725fabe...`.

The sole development read then failed before an evaluation artifact existed:
independent per-slot kind argmax produced at least one row with other than
exactly one STOP, and `HardProgramTape` correctly rejected the malformed tape.
Development/confirmation access is `1/0`; that board is closed. No v1
development accuracy is claimed and no same-board diagnostic, rescore, or
alternate decode is permitted.

## v2 hypothesis

Event slots have a disclosed global grammar: exactly one of eight slots is
STOP. Independent categorical argmax ignores this dependency. v2 replaces only
kind discretization with the exact maximum-a-posteriori assignment under that
public grammar:

1. for every slot, choose the higher-logit non-STOP kind and record its score;
2. compute each slot's STOP gain: STOP logit minus best non-STOP score;
3. choose the first maximum-gain slot as STOP; and
4. retain the best non-STOP kind in every other slot.

This is the exact maximizer over all `8 * 2^7` legal kind tapes, not beam search,
retry, evaluator repair, or an oracle-selected STOP. It consumes only the
model's raw eight-by-three kind logits and public grammar. It adds zero learned
parameters and is applied identically to treatment, row-shuffled-label,
consumed-parent, and binding-source-free compiler arms.

## Cross-generation exclusion amendment

Structured-decoder source `03c10d2dba5ce09a939c0e58617f73492a7df162`
was frozen before board/training seeds `3069712212437980146` /
`1406604500382831061`. The built board passed its inherited-parent and internal
audits and was never synced, submitted, trained, or opened. A new pre-launch
audit against the already consumed v1 development split found 13 abstract
operation-sequence overlaps in v2 train and one in v2 development. Exact prompts
and names were zero; v2 confirmation had zero sequence and 13-gram overlap.
Because train exposure to a consumed development sequence is avoidable, this
unlaunched board is rejected.

The successor builder must require the exact consumed v1 development file with
SHA-256 `b85ea65ed310554192d421c909c6519e4738b01a80647abe7f4ffd1b70079c4e`.
It reserves every operation sequence in that file in addition to all inherited
parent-training sequences. Its report binds the consumed hash and measures
prompt, name, sequence, and 13-gram overlap against every new split. Exact
prompt/name/sequence overlap must be zero everywhere; 13-gram overlap must be
zero for successor train and sealed confirmation. Development grammar-level
13-gram overlap is reported but may be nonzero because development is the
explicitly iterative split and uses the same task grammar. Old confirmation is
never opened. This amendment changes no model, decoder, threshold, optimizer,
parameter, or scored-access contract. Freeze new source and draw entirely new
seeds after the added tests pass.

## Frozen audit evidence

Every compiled arm must export its full raw float32 kind logits to the scorer.
The independent assessor must:

- reject non-finite values, numeric type coercion, wrong ranks/shapes, wrong
  decoder identity, or extra decoder-evidence keys;
- recompute raw per-slot argmax, raw exactly-one-STOP status, non-STOP choices,
  STOP gains, selected STOP, and the complete structured kind tape;
- reject any packet that differs from the recomputed exact MAP; and
- report raw one-STOP and raw exact-kind rates overall and treatment raw
  one-STOP rates by variant.

Only the 25+1 categorical packet reaches the separate recurrent executor. Raw
logits remain scorer evidence and are not executor input. Program tensors are
still poisoned and destroyed before late-query compilation. The structured
decoder cannot inspect row IDs, variants, depth, oracle fields, final state,
answer, trajectory, evaluator output, or confirmation authorization.

## Unchanged contracts

All fresh v1 contracts remain unchanged unless this document says otherwise:

- exact byte parent, execution core, and consumed diagnostic hashes;
- 48,000 train rows, 2,304 development rows, and 2,304 sealed confirmation rows;
- inherited-parent overlap audit and split-disjoint names, prompts, sequences,
  and scored 13-grams;
- hash-bound exclusion of every sequence from the consumed v1 development set;
- 6,748,897 trainable parameters, 20,955,890 compiler parameters, and
  146,057,595 nominal complete-system parameters;
- strict sub-150M comparison and strict sub-200M global gates;
- treatment and independent per-row shuffled-label arm with shared
  initialization, minibatch order, optimizer, and exactly 3,000 updates;
- epoch four as the sole checkpoint;
- every packet, pointer, execution, variant, depth, attribution, paired,
  intervention, negative-control, source-deletion, and custody threshold; and
- one development read followed by one confirmation read only if every frozen
  development gate passes.

Schemas and protocol IDs advance from v1 to v2 so no v1 board, checkpoint,
configuration, evaluation, or assessment can be mixed into v2.

## Required pre-seed tests

Before source freeze and any seed:

1. compare the decoder against exhaustive enumeration of all legal assignments;
2. prove that it preserves independent argmax whenever independent argmax is
   already legal;
3. prove exactly one STOP for arbitrary finite logits and deterministic tie
   behavior;
4. make the assessor reject a packet/logit mismatch and malformed float evidence;
5. pass the synthetic 2,304-row perfect-system evaluator/assessor contract;
6. pass all prior projected mechanics, board, source-deletion, artifact-binding,
   and parent-reconstruction tests; and
7. pass static, format, shell, and source-manifest checks.

## Claim boundary

A v2 pass would establish fresh-distribution source-deleted execution in this
bounded three-entity transport language with a disclosed structured kind
decoder. It would not establish unconstrained natural-language reasoning, a
learned halting grammar, self-generated plans, or active use of Shohin's nominal
125.08M trunk. Raw versus structured kind rates must be disclosed so the
decoder's contribution remains visible.
