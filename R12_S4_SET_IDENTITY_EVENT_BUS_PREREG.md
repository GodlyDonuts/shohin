# R12 S4 Set-Identity Event Bus Preregistration

## Status

**Frozen before production seed selection, fresh-board generation, training, or score access.**
Three set-bus tests, one assessor test, one fresh-board mechanics test, `py_compile`, Ruff, Slurm
syntax, and an actual raw-300k/v1 finite-backward construction pass. No production board, H100 fit,
development score, or confirmation access exists at freeze.

Post-freeze custody: seeds `14970823073944690832`, `939143060519850990`, and
`15848092346808854751` were retired before board creation for recorded remote dependency/invocation/
audit failures. Replacement seed `11437896185638727043` is the sole production board. It has 2,048
rows / 512 matched groups, passes every frozen gate, and is read-only before model access. Data,
report, and safe-archive SHA-256 values are respectively
`b49ddbbfad3da04181d6ec5401f8412b2953185e5e91e344208c8b6b0c5ba1e8`,
`808b0e0287e53576ffb234a5ea855943552ef3e60b2d3d20847b79f7254d692c`, and
`28302861b383fbdc8e5056e25bbd98b188487e87b241d25d2ef5ac82cebd43ae`.

## Causal diagnosis

On a wholly fresh board the frozen S4 v1 parser recovers 2,048/2,048 event counts and 1,914/2,048
exact programs. S4 v2 trains independent absolute start/end pointers to low loss but collapses to
254/2,048 exact programs, with 1,179 crossed/invalid event boundaries. The missing invariant is not
another coordinate decoder. It is lexical equality between a roster mention and an event mention.

## Representation

For token IDs `x_t` and a normalized model-owned soft membership `a_t`, define the vocabulary-aligned
set carrier

`C(a, x)[v] = sum_t a_t * 1[x_t = v]`.

This is a sparse token-frequency distribution. Two occurrences of the same multi-token name produce
the same carrier when their memberships are correct, independent of absolute position or BPE width.
It is order-insensitive; the admitted name generator and corpus audit must exclude collisions where
that would identify two roster names. The carrier adds no learned lexical table.

## Treatment

Freeze the raw 300k model and every S4 v1 parser parameter. Use the frozen v1 role logits as soft
membership priors for the three roster slots and terminal query phrase. Add only four tied 384x384
linear maps:

- event-entity query and key;
- event-literal query and key.

Each discovered operation-kind anchor queries every source token. Its score is the tied contextual
query/key score plus the frozen v1 event-role logit. A masked softmax yields a complete soft token
set, never a start/end pair. Event identity is cosine matching between the event carrier and three
roster carriers. Literal membership weights the frozen v1 amount head. Query membership weights the
frozen v1 query head. The frozen training-only kind lexicon and locked S3 executor remain unchanged.

Training uses gold kind spans only to form supervised event queries, exactly as v2. Inference uses
only kind anchors discovered from source tokens and frozen v1 role evidence. No depth, event index,
program, answer, final state, gold count, or confirmation field enters inference.

The real assembly loads exactly 71 frozen v1 tensors. Four 384x384 tensors and 589,824 parameters
are trainable; the complete adapter has 9,198,095 parameters and the raw-base-plus-adapter system has
134,279,759 parameters, strictly below 150M.

## Controls

1. Frozen S4 v1 on the same fresh board.
2. Shuffled token-membership supervision with identical architecture, initialization, examples,
   updates, optimizer, and semantic-label inventory.
3. A roster-derangement intervention that cyclically permutes the three predicted roster carriers
   after parsing and before event identity matching.
4. Gold-program locked-S3 sanity.

No v1 fine-tuning, learned token embedding, extra epoch, width change, threshold, top-k token count,
decoder sweep, or result selection is allowed.

## Fresh-board rule

Commit this preregistration, generator, model, trainer, evaluator, assessor, tests, and jobs first.
Then draw one random production seed and generate a new 2,048-row development board. It must be
disjoint at exact prompt, word-13-gram, nonce/name, token-multiset identity, and factor levels from
all S4 train/old-development/v2-development and supplied public compiler/executor boards. Each row's
three roster token multisets must be unique. Each arm may read the board once. No post-score repair,
rescore, or confirmation access is admissible.

## Frozen gates

- event count at least 98% overall and 95% at every depth;
- exact program, state, and answer at least 95% overall;
- exact program at least 90% at each depth 5--8;
- roster-carrier recovery at least 95% overall;
- treatment exact programs at least one percentage point above frozen v1 on the same board;
- shuffled exact programs at most 40%;
- roster-deranged exact programs at most 40%;
- gold-program S3 sanity at least 99%;
- strict total parameters below 150,000,000;
- development access exactly one and confirmation access zero.

A pass authorizes one separately frozen confirmation board. It does not establish unseen operation
semantics, planning, learned halt, order-sensitive lexical identity, free-form reasoning, benchmark
improvement, or novelty.
