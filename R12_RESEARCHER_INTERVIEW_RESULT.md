# R12 Researcher Interview Result

**Status:** canonical matched interview completed; all three checkpoints fail the
locked state-transition board. Transcript reading localizes distinct computation,
serialization, and iterative-consumption failures.

## Frozen execution

| Role | Job | Checkpoint SHA-256 | Result artifact SHA-256 |
|---|---:|---|---|
| Raw 200k | `692078` | `675af7cffdc87ccd43c56a15f0616d368442aad56deb0df3fe11b5a5064aac2a` | `aae9cef341ae76ae151706cbcaa96d6c606c6692802fd3d436440aa27b20028a` |
| DRS r3 from 200k | `692079` | `d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459` | `1dff3df56e07d323030a50a4a7ab3b02655b20b5c670756f4834689465ebfb24` |
| Raw 300k | `692080` | `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6` | `4a4b195de7191b385dbd935f693757831f8db0bf1ab8dee369e357a4875dda58` |

All jobs completed `0:0` on CUDA under source commit
`e72c28770c2fb776e90673f1a9f580339c81a1a6`. The three artifacts bind the same
interview SHA-256
`a72387a0a72418f119bf35791032bb889266b3f9ba8a3b728fc7f6978c0d4f8d`,
comparison-manifest SHA-256
`032ebbf6980d23afcb01ed321a9890e792e725dcb72441d915694ab1b40998c0`,
and tokenizer SHA-256
`87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4`.

## Locked scores

| Checkpoint | Exact syntax | Semantic state | RIV20 writer | RIV20 gold reader | RIV20 end to end |
|---|---:|---:|---:|---:|---:|
| Raw 200k | 0/20 | 0/20 | fail | fail | fail |
| DRS r3 200k | 0/20 | 0/20 | fail | fail | fail |
| Raw 300k | 0/20 | 0/20 | fail | fail | fail |

These are the official results. No post-hoc parser or prompt repair changes them.

## Direct transcript diagnosis

The raw 200k checkpoint sometimes computes a useful local quantity while ignoring
the requested interface. On RIV01 it emits `58+27=85`; on RIV09 it emits the
correct full sum `4786 + 5967 = 10753`; and the RIV20 writer computes
`31+14=45, 45*5=225`. None is serialized into the requested state packet, and the
gold-capsule reader answers `225` instead of applying `-37,+6` to reach `194`.
This is local arithmetic evidence, not a board pass.

DRS changes the failure mode. Six of the first nineteen responses are parseable in
the requested state grammar, versus zero for either raw checkpoint, but their values
are wrong. Examples include `ember=42` instead of 938,
`m1=40;m2=9;m3=4` instead of `7,63,67`, `a=7;b=6;c=0` instead of
`7,11,4`, and `r=152` instead of 194. DRS therefore improved response-mode control
without establishing the update rule. It also corrupted an unchanged field once.

Raw 300k does not improve this interview over raw 200k. It occasionally performs a
local subexpression, such as `14*14-39=157`, but drops the initial register value,
repeats indefinitely, or emits templates. In RIV20 it first reaches 225 and then
continues an uncontrolled update loop. Additional pretraining did not solve state
transport or halting on this board.

## Causal boundary and next test

The matched result supports a narrower controller/executor diagnosis:

1. Raw 200k contains some local arithmetic competence but lacks reliable packet
   serialization and instruction-conditioned halting.
2. DRS can impose a packet-like output grammar, consistent with its late residual
   digit channel, but does not reliably compute or consume the carried state.
3. Raw 300k provides no evidence that scale in pretraining steps alone repairs the
   missing state-update cycle.

The next descriptive interaction separates plain computation, gold-state
serialization, model-state serialization, packet copying, one-step consumption,
two-step consumption, and self-review. It may localize the failure more precisely,
but it cannot promote a model or architecture. Promotion still requires frozen
held-out autonomous multi-step gates with no host arithmetic, oracle schedule,
residual patch, or result tape.
