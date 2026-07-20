# R12 ER-CST Fresh Board Preregistration Amendment v1.2

**Status:** pre-training identifiability repair. No training seed, H100 job, neural
output, development access, or confirmation access exists.

The v1.1 board from exact source `fba34cdc9bfab75882dee8093b07ab96042d4a07`
and seed `1686667709479653771` passed every integrity gate and remains unopened, but
a post-admission architecture/data audit found that its three rule-card records had
arbitrary latent slot IDs with no source-visible address. Exact ordered card scoring
would therefore reward accidental compact-name correlations or an unidentifiable
permutation choice.

That board is closed before training. V1.2 adds only a one-character rule-slot address
to each determining witness record: `W1`/`W2`/`W3` or the matched renderer's
`L1`/`L2`/`L3`. The address identifies where to store a learned card; it reveals no
permutation, opcode binding, execution result, query, state, or answer. Physical line
storage remains independently shuffled, so the compiler must still parse the address,
infer operation meaning from before/after symbol equality, and bind later opaque opcode
uses.

The independent production parser now requires exactly one of every rule slot and
checks each slot against the target. Row counts, family construction, name bijection,
renderer cosets, model architecture, parameter count, supervision, controls, gates,
and custody remain unchanged. The old board seed may not be reused. Commit and push
this exact repair before drawing a new board seed.
