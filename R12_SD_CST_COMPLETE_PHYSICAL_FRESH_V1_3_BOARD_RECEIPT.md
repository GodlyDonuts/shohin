# R12 SD-CST Complete Physical Fresh-Board v1.3 Receipt

**Status:** admitted and sealed before training-seed draw, model fitting, or
scored access.

## Custody ordering

1. Scientific source commit
   `eed66757c47e126b6566ee269bc73b0c0cef4fab` was pushed before randomness.
2. Raw 64-bit board beacon `18144246429379773690` was drawn afterward.
3. The beacon was reduced modulo `2^63` to board seed
   `8920874392524997882`.
4. The exact committed source generated and audited the board.
5. No training seed exists at this receipt point.

## Board certificate

| Split | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| Training | 48,000 | 180,571,520 | `bb870ac36ad4c78376dae08fe6f605e987a2a2883c5deb18bde06a82eb0b4ce2` |
| Development | 2,048 | 8,121,776 | `5dc5035cd825066be444ce33f6e30eb7aabad16e6ca0b4fe3b47fcc0060e9dfa` |
| Sealed confirmation | 2,048 | 8,128,688 | `6186fb8c83c9863db2844f5eb537194a713c5ab16d2a41f1f88f6e3742f02165` |

Board report SHA-256:
`fd487cdf7c30cf945ace152e389aaf5c354b8b6a55555c2acc6f046e8ed00b24`

The report identifies schema
`r12_sd_cst_complete_physical_fresh_board_report_v1_3`, protocol
`r12_sd_cst_complete_physical_fresh_v1_3`, exact source commit, and exact board
seed. All 17 admission gates pass. They include independent semantic/oracle
agreement, exactly nine bounded records and one HALT, production-parser
acceptance over all 52,096 rows, renderer parity, globally unique split/prior-
disjoint names, zero exact/13-gram/name/operation-sequence leakage, expected
counts, sealed confirmation, and zero scored access.

Confirmation is mode `0600`. Development/confirmation access is `0/0`; no
board-local access directory exists. A complete second build from the same
source, seed, and prior inputs is byte-identical for report, training,
development, and confirmation files.

## Frozen neural contract

Treatment and the matched family-deranged arm each train 102 tensor names / 
12,152,855 parameters for 3,000 updates. Complete deployed size remains
192,129,179 with 7,870,821 parameters of strict-200M headroom. Every setting,
gate, threshold, control, and claim boundary is frozen in
`R12_SD_CST_COMPLETE_PHYSICAL_FRESH_V1_3_PREREG.md`.

The next allowed action is to commit/push this receipt and only then draw one
training seed. No development or confirmation bytes may be opened before both
matched endpoints and the immutable gate config exist.
