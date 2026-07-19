# R12 S9 Occurrence-Quotient CPU Result

**Decision:** admit theory/representation only; no neural capability claim

**Seed:** `792451398761220486`

**Rows:** 2,048 closed S8.1 development sources

**Report SHA-256:**
`f77dce825314cc38b0630cd574b450284c00fc8afa23dc0ab39cfc5be8ef2c94`

The oracle-emitted quotient reconstructs the exact graph, recurrent state, and
answer on 2,048/2,048 sources. Independent class-ID and relation-storage
permutations remain exact on 2,048/2,048. Swapping card witnesses leaves only
30/2,048 exact states; reversing links leaves 154/2,048. Splitting one repeated
operation, merging entity classes, giving every occurrence a unique free word,
corrupting a relation type, or swapping event argument slots causes strict
rejection on all 2,048 rows.

All 13 CPU gates pass. This shows that class-level emitted relations are a
lossless and causal interface to the retained S8 executor. It does not show
that a neural model can find the spans or relations. The CPU run uses frozen
board labels as oracle emissions and must not be reported as a reasoning score.
