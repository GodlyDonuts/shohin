# R12 SD-CST Dedicated-Projection Binding Pilot

**Status:** frozen training-only interface falsifier; no scored access authorized

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
