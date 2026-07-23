# CTAA Independent Opcode Binding Amendment

## Status

**Draft source amendment. REJECT_SOURCE_FREEZE.**

The current CTAA assessor reports binding_exact, but the value is an alias for
action-card content equality. The compiler resolves declaration identity into
card addresses before the hard packet is committed, so no independent binding
prediction survives into evidence. Renaming that alias cannot satisfy the
signed statistical contract.

This amendment defines the smallest causal representation that makes card
semantics, declaration binding, and event sequencing separately falsifiable.
It authorizes implementation and source-only testing. It does not authorize a
board seed, training seed, scored access, H100 job, or reasoning claim.

## Required Representation

The compiler must commit three distinct outputs:

1. action_cards[card_address]: the four predicted semantic state maps.
2. opcode_to_card[local_opcode]: a permutation mapping the four
   declaration-local opcode ordinals to card storage addresses.
3. opcode_schedule[step]: the event tape in declaration-local opcode ordinals
   plus the interior STOP symbol.

The packet derives the execution schedule instead of accepting a separately
predicted resolved tape:

    resolved_schedule[t] =
        STOP                               if opcode_schedule[t] == STOP
        opcode_to_card[opcode_schedule[t]] otherwise

local_opcode is the rendered declaration-line ordinal, not a semantic
operation label. A declaration order [W3, W1, W4, W2] therefore has the
binding permutation [2, 0, 3, 1] when cards are stored in canonical semantic
order.

The recurrent core remains source-deleted. It receives only the selected card
and current state after deterministic resolution.

## Packet And Evidence Contract

- Expand the hard packet from 56 to 60 bytes by adding four binding bytes.
- Require opcode_to_card to be exactly a permutation of 0..3.
- Reject duplicate, missing, out-of-range, or STOP binding values.
- Preserve opcode_schedule, opcode_to_card, and the deterministically resolved
  schedule in committed raw evidence.
- Recompute the resolved schedule from committed bytes and reject any mismatch.
- Commit all binding bytes before query disclosure.

The board oracle and train-only labels add opcode_to_card and opcode_schedule.
Development and confirmation program sources remain source-only; they must not
disclose those labels.

## Independent Metrics

    cards_exact =
        predicted_action_cards == oracle.action_cards

    independent_binding_exact =
        predicted_opcode_to_card == oracle.opcode_to_card

    opcode_schedule_exact =
        predicted_opcode_schedule == oracle.opcode_schedule

    schedule_exact =
        resolve(predicted_opcode_to_card, predicted_opcode_schedule)
        == oracle.schedule

    program_exact =
        packet_valid
        AND cards_exact
        AND independent_binding_exact
        AND opcode_schedule_exact
        AND initial_exact

The scorer must reject any attempt to recover binding by matching card
contents, card hashes, resolved schedules, outcome equality, or oracle-assisted
assignment.

## Decisive Controls

1. **Card-only mutation:** change one card coordinate while preserving binding
   and opcode schedule. Cards fail; binding remains exact.
2. **Binding-only mutation:** transpose two binding entries while preserving
   card bytes and opcode schedule. Cards remain exact; binding fails.
3. **Compensated relabeling:** apply the same opcode permutation to binding and
   opcode schedule so the resolved execution, state trace, terminal state, and
   answer remain unchanged. Binding and opcode-schedule identity must still
   fail against the committed source labels.
4. **Declaration-order shuffle:** reorder rule declarations without changing
   their semantics. Cards and resolved schedule remain unchanged while binding
   and opcode schedule transform consistently.
5. **Opcode alpha recode:** rename opaque opcode strings consistently. Every
   hard output remains invariant.
6. **Card-storage reindex:** permute physical card storage and change only the
   binding values. The opcode schedule remains byte-identical.

## Balance And Coverage

All 24 binding permutations must appear. Every sufficiently large scored
stratum must balance binding permutations and card-address marginals. The board
writer must prove that renderer, factorial cell, program class, depth, and
binding permutation are not confounded.

## Mandatory Tests

- Packet round-trip preserves all 60 bytes.
- A one-byte binding mutation changes packet and receipt hashes.
- Binding loss reaches the new decoder slots and shared four-class head.
- Invalid permutations fail before execution.
- Resolved-schedule inconsistency fails before scoring.
- Query disclosure cannot alter binding evidence.
- Card-only, binding-only, and compensated-relabeling controls produce the
  distinct metric patterns specified above.
- Runtime card reindexing changes binding only and preserves the local opcode
  tape.
- The assessor exposes independent_binding_exact; the obsolete aliased
  binding_exact field is rejected.

## Claim Boundary

Passing these source and mutation gates proves only that the binding capability
is independently measurable and causally used. Capability advancement still
requires the signed five-seed statistical specification, complete runtime
intervention/resource receipts, independent raw rescoring, and the unmocked
Linux custody smoke.
