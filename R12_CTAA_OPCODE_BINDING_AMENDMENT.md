# CTAA Independent Opcode Binding Amendment

## Status

**Implemented source amendment under test. REJECT_SOURCE_FREEZE.**

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
operation label. Physical card storage is fixed by literal addresses W1..W4;
there is no privileged "canonical semantic" ordering. A declaration order
[W3, W1, W4, W2] therefore has binding [2, 0, 3, 1]. This explicit address
gauge is mandatory: otherwise a simultaneous permutation of card storage and
binding is observationally equivalent and `binding_exact` is not identified.

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
        AND schedule_exact
        AND initial_exact

The scorer must reject any attempt to recover binding by matching card
contents, card hashes, resolved schedules, outcome equality, or oracle-assisted
assignment.

## Decisive Controls

1. **Card-only mutation:** change one card coordinate while preserving binding
   and opcode schedule. Cards fail; binding/local/resolved schedule remain
   exact; a separating trace must change.
2. **Binding-only mutation:** apply a nonidentity permutation to binding entries
   while preserving card bytes and opcode schedule. Cards/local tape remain
   exact; binding and resolved schedule fail; a separating trace must change.
3. **Compensated relabeling:** use a non-involutive three-cycle `pi`, with
   `T' = pi(T)` and `B' = B o pi^-1`, so `B'[pi(o)] = B[o]`. Binding and local
   tape identity fail, while resolved execution, state trace, terminal state,
   and answer remain byte-identical. A transposition alone is insufficient
   because it cannot expose an inverse-direction bug.
4. **Declaration-order shuffle:** reorder rule declarations without changing
   their semantics. Cards and resolved schedule remain unchanged while binding
   and opcode schedule transform consistently.
5. **Opcode alpha recode:** rename opaque opcode strings consistently. Every
   hard output remains invariant.
6. **Card-storage reindex:** for `cards'[new] = cards[old]`, set
   `binding'[opcode] = inverse_storage[binding[opcode]]`. The local opcode tape
   remains byte-identical; physical card and binding identities fail against
   the original oracle while the trace remains invariant.

## Balance And Coverage

All 24 binding permutations must appear. Every 288-row scored block uses a
deterministic `Z_24` coset construction: the 18 query/initial cells use residues
not divisible by four, the 16 renderers use residues not divisible by three,
and their modular sum selects one element of `S_4`. This yields exactly 12
occurrences of each binding and exactly 72 occurrences of every
local-opcode/card-address pair. Within each block every fixed renderer sees 18
distinct bindings and every fixed query/initial cell sees 16 distinct bindings.
The writer and an independent seedless audit must prove these equalities.

Exact balance in every smaller crossed subcell is arithmetically impossible
when its row count is not divisible by 24. Such cells must use the declared
block design and report the exact attainable marginal/separation bounds rather
than claiming impossible balance.

## Identification Completion

Packet identifiability is necessary but does not by itself establish transient
working memory or compositional reuse. Before source freeze the neural board
must additionally include:

- **Persistent excitation:** every opcode is invoked from separating states,
  so no binding entry is behaviorally dormant.
- **Alternating-group completion:** an `A4`-only training slice with held-out
  odd permutations, preventing a 24-case table from masquerading as an
  equivariant binding rule.
- **Write-delete-delay-read:** the source is deleted after compilation, at
  least one unrelated transition intervenes, and only then may a late query
  read the bound result.
- **Multi-epoch rebinding:** adjacent transpositions change declaration
  bindings across episodes while physical cards remain fixed, testing overwrite
  rather than static lookup. The adjacent-transposition Cayley graph of `S4`
  has diameter six, so confirmation must include cue paths through length six;
  a five-cue design cannot reach the reversal permutation.

These are predeclared falsifiers, not current capability claims.

## Mandatory Tests

- Packet round-trip preserves all 60 bytes.
- A one-byte binding mutation changes packet and receipt hashes.
- Binding loss reaches the new decoder slots and shared four-class head.
- Invalid permutations fail before execution.
- Resolved-schedule inconsistency fails before scoring.
- Query disclosure cannot alter binding evidence.
- Card-only, binding-only, and non-involutive compensated-relabeling controls
  produce the distinct metric and trace patterns specified above.
- Runtime card reindexing changes binding only and preserves the local opcode
  tape.
- The assessor exposes independent_binding_exact; the obsolete aliased
  binding_exact field is rejected.

## Implemented Source Evidence

The version-4 runtime plan now treats the three algebraic controls as mandatory
operations rather than optional unit tests. Every one of 864 source-blind
anchors receives:

- a card-only mutation of the physical card used at the first transition;
- a binding-only three-cycle that moves the first local opcode while preserving
  cards and local tape;
- a compensated non-involutive three-cycle that changes binding and local tape
  while preserving every resolved event.

The first two controls begin from a distinct initial state, so their first
transition must separate the original and mutated packet. The compensated arm
must preserve the complete state trace byte-for-byte. Plan, operation,
commitment, concrete-mutation, and runtime-implementation schemas were advanced
to prevent old 56-byte or pre-binding artifacts from entering the new
25,056-attempt evidence lattice.

This closes the source-level gauge/separability gate only. It does not close
the alternating-group completion, write-delete-delay-read, dynamic rebinding,
independent dual-scorer, capability-time resource, or Linux custody gates.

The source-free mechanics report
`artifacts/r12/ctaa_v2_preflight/binding_identification_mechanics_v1.json`
verifies the future experiment geometry: exact `12/12` `A4`/odd splits with
`3/3/3/3` local marginals, 72 delay cases at `0/32/128`, 24/24 donor-register
following, 23/24 identity-reset differences, all 24 reachable bindings across
1,092 adjacent-generator sequences, and 6,015 committed-prefix causality
checks. Its file SHA-256 is
`21cbcafb4d8adc49cebe978bd2a2b1d482a54f526a52f32553a7efa3e22960b9`.
These are finite mechanics, not learned results.

The neural completion source now instantiates the previously abstract A4 gate.
`pipeline/build_ctaa_binding_completion_board.py` renders a complete
24-permutation declaration orbit for every fixed semantic scaffold and seals
the odd half. `train/ctaa_binding_completion.py` defines a shared slot-local
treatment and a globally connected same-target control. Four opcode and four
physical-card slots are decoded independently. The treatment scores every
opcode/card pair with one shared `3840 -> 156 -> 1` network and zero global
context; the control uses the identical scorer and calls with all eight slots
in context. Both have 599,353 parameters and 9,587,136 dense analytic MACs.
The treatment is exactly bi-equivariant to opcode and card-slot permutations.
The 24-way classifier is retained only as a support-starved negative, not
misreported as a matched control.
`train/train_ctaa_binding_completion.py` independently decodes the four slots,
qualifies one common compiler on A4 only, freezes and stores one shared feature
cache, and has no confirmation input. `predict_ctaa_binding_completion.py`
requires an independently validated five-seed freeze manifest and has no oracle
input. `assess_ctaa_binding_completion.py` has no source input, revalidates all
five source-free artifacts, atomically spends the unique oracle-access ledger,
and opens one committed oracle blob once.
`capacity_ctaa_binding_completion.py` runs disposable all-S4 fits from
committed assessment labels without reopening the oracle. The chimera
diagnostic imports no odd representation: it composes every slot from a
distinct A4 donor and derives an odd target. The assessor also materializes the
actual 60-byte packet and separately reports card, binding, state, local tape,
resolved schedule, excitation, counterfactual effect, and whole-program
exactness. A measured-resource job and source-free finalizer apply immutable
five-seed attribution gates.

The immutable admission binds a canonical digest over every tracked protocol
source and direct runtime dependency, rejects dirty or untracked protocol
files, fixes one absolute custody directory, and preregisters every decision
threshold. Cross-stage tensor artifacts are hashed before restricted
`weights_only=True` loading. Odd source and oracle rows share unique opaque row
IDs, and the assessor claims a single `O_CREAT|O_EXCL` oracle ledger before its
one read.

This is implemented source under test, not a neural result. No production board
seed, training seed, sealed odd access, GPU allocation, or binding advancement
claim has been created. Source freeze still requires the final complete
regression, a fresh independent review of this hardened protocol, the measured
resource receipt, source-level symmetry checks, and the unmocked Linux custody
smoke.

## Claim Boundary

Passing these source and mutation gates proves only that the binding capability
is independently measurable and causally used. Capability advancement still
requires the signed five-seed statistical specification, complete runtime
intervention/resource receipts, independent raw rescoring, and the unmocked
Linux custody smoke.
