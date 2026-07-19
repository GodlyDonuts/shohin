# R12 RGDE Recurrent-Depth Confirmation Result

**Decision:** `reject_rgde_depth_confirmation`

**Strong diagnostic:** the frozen recurrent executor itself confirms through
depth eight under gold packets; the primary predicted packet fails on unseen
paired-name grounding. Do not refit or rerun this confirmation board.

## Custody

- Source/preregistration commit `85ead2e` preceded production seed
  `11772835344958352982`.
- The 2,048-row / 6,136-card / 798,346-token board passed every CPU/data gate.
- Board SHA-256:
  `742899905a39c0afc4575e94ff533d489aaf42992c248ddf4668f44609eab2d0`.
- Job `693124` ran once on H100 `evc29`. It wrote primary, gold, and operation
  control outputs, then exited `1:0` because the depth-stratified query labels
  made a full within-depth derangement impossible. There was no fit, retry,
  alternate seed, or old-confirmation access.
- Safe evidence archive SHA-256:
  `ec795b89683451e1f495e902a9db3fc8b736f23f4d4076ee7cbdcc1db7a20663`.

The missing query control and failed Slurm receipt independently fail gate 10.
The primary scores already fail gates 2--7, so no replacement control can
change the rejection decision.

## Primary predicted packet

| Depth | Answers | Exact final state | All transitions | Entity match |
|---:|---:|---:|---:|---:|
| 3 | 79.651% | 76.453% | 61.337% | 82.752% |
| 4 | 78.779% | 73.256% | 56.395% | 82.922% |
| 5 | 81.471% | 72.059% | 46.471% | 79.118% |
| 6 | 72.059% | 65.294% | 35.294% | 76.765% |
| 7 | 78.824% | 72.941% | 41.765% | 77.647% |
| 8 | 67.059% | 55.294% | 23.824% | 74.044% |
| **overall** | **76.318%** | **69.238%** | **44.238%** | **77.952%** |

Amount is 100% and query is 99.707%. Every surface lies between 75.000% and
78.516% answers; only 243/512 quartets have all four answers correct. The error
accumulates with depth because operation-to-initial entity identity is already
wrong on roughly 22% of atomic packets.

## Gold-packet localization

| Depth | Answers | Exact final state | All transitions | Entity match |
|---:|---:|---:|---:|---:|
| 3 | 99.419% | 100.000% | 99.419% | 100.000% |
| 4 | 99.709% | 100.000% | 97.965% | 100.000% |
| 5 | 100.000% | 100.000% | 99.412% | 100.000% |
| 6 | 100.000% | 100.000% | 99.706% | 100.000% |
| 7 | 100.000% | 100.000% | 99.118% | 100.000% |
| 8 | 99.118% | 100.000% | 98.824% | 100.000% |
| **overall** | **99.707%** | **100.000%** | **99.072%** | **100.000%** |

This is the main scientific result. The same 1,491,279-parameter cell was
trained only on independent atomic updates and never on depth 3--8. With the
compiler interface repaired by gold role spans/kinds, it maintains exact state
through eight recurrent calls. There is no intrinsic recurrent-state collapse
at this depth.

The 23.389-point answer gap and 30.762-point final-state gap between predicted
and gold localize the failure to Stage A packet grounding on the new hyphenated
paired names. Those names are compositionally novel relative to the compiler's
single nonce-name training distribution.

## Causal operation control

Replacing every operation stream with a different same-depth program reduces
answers to 32.275%, exact final state to 13.428%, and all-transition exactness
to 0.244%, while query remains 99.707%. The executor is consuming operation
packets causally; its positive gold depth score is not an inert-state shortcut.

## Next admissible move

Preserve this board as a sealed negative and never fit on it. The next
development experiment must target compositional referent grounding on a new,
disjoint public paired-name board. A no-fit relational carrier should compare
role-weighted token sets before source deletion rather than compressing each
referent independently to one mean vector. Only after that interface passes a
new development board may a second, independently seeded depth confirmation be
designed.

The confirmed current capability is therefore precise: **the native tied
executor generalizes its atomic update rule through depth eight when supplied
correct bounded packets; the ordinary compiler does not yet ground novel
composed entity names reliably enough for end-to-end confirmation.**
