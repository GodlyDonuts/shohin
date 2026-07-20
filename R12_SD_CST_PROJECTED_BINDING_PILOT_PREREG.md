# R12 SD-CST Dedicated-Projection Binding Pilot

**Status:** training-only pass; source-deletion mechanics gate preregistered; no scored access

## Parent results and diagnosis

The content fingerprint itself is viable: joint pilot `693974` reaches 100% on
declaration pointers, initial-occurrence pointers, and arbitrary initial state.
The exact frozen byte parent is also viable: hierarchical pilot `693977` keeps
line/kind/amount/query and one raw STOP at 100%, with every frozen parameter
byte-identical. Its model-line mask lifts event-occurrence span exactness from
0% to 32.625%.

`693977` is nevertheless rejected. With only new query vectors through the
parent's frozen key/query projections, declaration and initial-occurrence
all-slot exactness stay zero, event identity is 4.1625%, initial state 17.725%,
and whole tape 0.8125%. Report SHA-256 is
`2b27cd79ac35cb418c9b4f8f73db7378c9d3f4c0e70785291702d9383213d46a`.

The narrow hypothesis is that the frozen parent residual contains the needed
bytes/context but its line-task key projection is not a sufficient interface
for name-span binding. The next pilot changes that interface only.

## Frozen mechanism

The exact parent checkpoint and all inherited parameters remain frozen under the
same SHA and byte-digest gates as `693977`. The model-selected semantic line plus
newline mask is unchanged. The binding path adds:

- one independent bias-free 384-by-384 query projection; and
- one binding-only key adapter: `Linear(384,384) -> GELU -> Linear(384,384)`.

These layers read frozen source memory and are used only by declaration,
initial-occurrence, and event-occurrence pointers. They cannot change line
addresses, event kind, amount, query, the source encoder, or slot mixer. The
shared position-free byte-bigram fingerprint and permutation scorer are
unchanged.

The compiler has 20,955,890 parameters; 6,748,897 are trainable. Complete Shohin
is 146,057,595 parameters.

Training, parent hash, consumed 40k/8k partition, four epochs, optimizer, losses,
raw metrics, zero-access rule, and every gate are identical to the frozen-parent
hierarchical preregistration. Only the declared binding projection and its
parameters differ. Runtime accounting must reproduce the counts above and
remain below the stricter 150M pilot cap, despite global sub-200M authority.

## Decision

All 14 inherited hierarchical gates remain immutable, including 100% prefit
parent fields, frozen-parent byte identity, at least 90% event/name pointers,
at least 80% identity/initial, at least 60% raw whole tape, exactly one raw STOP,
and access `0/0`.

A pass advances only to shuffled/swap/no-address/hard-negative causal controls
and fresh-board end-to-end source deletion. It is not a reasoning score. A
failure closes this projected interface before any scored split.

## Result

Source commit `9bd2e04ea93406eb50a6fd112cd844892b72a7c4` preceded
seed `6715972906370623241`. Job `693979` completed on H100 `evc22` in
4m04s. Epoch one already reached 7,979/8,000 exact whole tapes; epochs two
through four reached 8,000/8,000. The final held-out consumed-training result is:

- 8,000/8,000 initial state, kind, identity, amount, query, and whole tape;
- 8,000/8,000 initial-occurrence pointers;
- 7,999/8,000 declaration pointers;
- 7,998/8,000 event-occurrence pointers;
- exactly one raw STOP on all 8,000 rows; and
- frozen-parent digest unchanged.

All 14 frozen gates pass. The untrained projected prefit has only 1/8,000 whole
tapes, so the result is not inherited from the frozen parent. Checkpoint and
report SHA-256 values are
`f347d1aea90dd3c60f7500167c7c22884451b365880259698306c6fce8ab10f3`
and `5d6be14798af3a75781898c6405e956fe9eb040e861ee63e669e7b87e7fa6f32`.
Development and confirmation access remain `0/0`.

This passes compiler mechanics only. The next authorized experiment is
`R12_SD_CST_PROJECTED_MECHANICS_PREREG.md`; no fresh scored board is authorized
until its separate-process source-deletion and causal controls pass.
