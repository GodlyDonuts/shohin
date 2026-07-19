# R12 S6 Contextual Affine Law CPU Mechanics Result

**Decision:** `pass_s6_cpu_mechanics`

S6 defines a new operation law by two categorical demonstrations over an affine
position action and asks a learned unit to infer and recurrently apply laws absent
from training. This result establishes the exact mechanics and data split only;
no neural model, board, fit, development score, or confirmation artifact exists.

## Custody

- Original preregistration SHA-256:
  `0790584421afe6e27eb892173ca41c15c6931691434ba8f1205272c63b52b5ff`.
- The first CPU falsifier failed before writing a report because the raw hash
  split omitted value `1` from the modulus-5 training `card_y1` coordinate.
- V1.1 documents the sole split repair at SHA-256
  `a57d4108c0ae6e44555a30bae2dd59b13d479d2bbb3a406eee202113b572c013`.
  It moves the lexicographically first held-out law that supplies a missing
  coordinate into training, without changing the theorem, architecture,
  controls, thresholds, or claim boundary.
- The repaired falsifier report is
  `artifacts/r12/s6_contextual_affine_law_cpu_falsifier.json`, SHA-256
  `a31a232c83a53d0b7aff87b4a495abd6740d98589059325951e2e4688e2bded6`.

## Exact Results

All frozen gates pass over moduli 5, 7, 11, and diagnostic modulus 13:

- 328/328 affine laws have unique two-witness cards;
- every one-witness class contains exactly `m-1` possible laws;
- 3,748/3,748 law-position destinations reconstruct exactly;
- 3,748/3,748 categorical pop-insert cells close exactly;
- every modulus has a noncommutative order twin and separating late query;
- train, development, and reserved-confirmation law sets are disjoint;
- all admitted training splits cover every card coordinate and destination;
- treatment input contains only `modulus`, `card_y0`, `card_y1`, and
  `current_location`.

The repaired split counts are:

| Modulus | Train | Development | Reserved confirmation |
|---:|---:|---:|---:|
| 5 | 10 | 4 | 6 |
| 7 | 28 | 8 | 6 |
| 11 | 65 | 22 | 23 |
| 13 diagnostic | 82 | 39 | 35 |

Only modulus 5 required a promotion: `m5_a4_b2` moved from confirmation to
training to supply missing `card_y1=1`. No row or score existed when this repair
was frozen.

## Boundary And Authorization

This is an identifiability and mechanics pass, not learned reasoning. An exact
host affine decoder remains a favorable ceiling, and any fixed finite board can
be tabled. The permitted next action is to commit these bytes, draw one
post-commit development seed, build atomic training cells plus a disjoint
recurrent development board, and fit the preregistered card-conditioned module.
Confirmation generation remains forbidden until every development gate passes.
