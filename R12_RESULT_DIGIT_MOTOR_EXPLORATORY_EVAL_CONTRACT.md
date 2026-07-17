# R12 Result-Digit Motor Exploratory Evaluation Contract

**Protocol:** `R12-RDM-DEV-v1-disclosed-patch`

**Frozen:** 2026-07-17 19:45 EDT, while Newton job `692235` was still in
feature extraction. No teacher-forced development metric, autonomous score, or
cycle score had been printed or read. This contract does not change the running
source bytes.

**Status:** exploratory rejection/advancement gate only. It can never promote
an architecture or support a broad reasoning claim.

## Bound execution

| Item | SHA-256 |
|---|---|
| Slurm batch script spooled by `692235` | `7f6d973fdde5fa41638f01ea236dfdd4fb84841846d5ed2b35125000fcbaa17c` |
| Evaluator source | `a9e12df6fd1d2a78bb3fa3eb4435b8023c7602108652db573bf049f8dddbaa3a` |
| Evaluator tests | `7975329e86640f38c53caf3eaa234b0ea174459a3aedd409878a3130d53162fc` |
| Wide motor | `5b277e2797b9b4dee6bc0578e7891c5d0ae72d2217da74bf0fa1ab39df3b844a` |
| Base checkpoint | `d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459` |
| Tokenizer | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| Held-out episodes | `89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646` |
| Frozen cycle artifact | `0b927fee009de5e5cf87971ecaf390c716d6d9acb5644cabe3c176f6da9d4e7a` |

The motor's disclosed training-source manifest is
`dd94a00be3984d2f43d6342459ba9749e3164624bb9b96977bd19c11f23845f0`.
The evaluator records this separately from its own source contract and requires
the explicit noncanonical `--allow-evaluator-patch` authorization. The output
must be newly created at
`artifacts/r12/result_digit_motor_r2_eval/eval.json`.

## Parameter contract

The frozen base has 125,081,664 unique parameters. The
`576 -> 4096 -> 4096 -> 10` motor has 19,185,674, for a system total of
144,267,338 and 5,732,661 parameters of strict headroom below 150,000,000.

## Locked interpretation

Pool rates over all reported regimes by summing numerators and denominators.
All percentage-point differences below are absolute.

The exploratory treatment advances to a new hardened confirmation evaluator
only if every gate passes:

1. **Mechanical:** job exit `0:0`; all bound hashes match; output is complete,
   canonical JSON; `dead` teacher-forced metrics equal `base`; non-DWS checks
   report no motor firing and decoded identity on every listed prompt.
2. **Held-out actuator:** treatment teacher-forced digit top-1 is at least 99%,
   at least 3 points above base, and at least 10 points above shuffled.
3. **First autonomous transition:** treatment exceeds both base and shuffled by
   at least 5 points.
4. **Full autonomous state loop:** treatment exceeds both base and shuffled by
   at least 2 points.
5. **Frozen-cycle first transition:** treatment exceeds both base and shuffled
   by at least 5 points.
6. **Routing:** treatment and shuffled fire exactly once for every reported
   grammar-site opportunity; dead is behaviorally equal to base on every
   aggregate autonomous and cycle count.

Failure of any gate rejects this wide motor as an autonomous actuator despite
its training-board fit. There is no threshold retuning after the report opens.

## Mandatory confirmation after an exploratory pass

Even a complete pass above is not promotion evidence because this evaluator:

- rolls only the normal autonomous branch rather than a paired normal and
  counterfactual intervention;
- retains only bounded transcript samples rather than complete raw rollout
  evidence; and
- checks off-site preservation by decoded strings, not exact token IDs and
  per-boundary full-vocabulary logits.

Any successor must freeze a new secret confirmation board before extraction,
score both branches and paired intervention, retain complete row-level evidence,
require exact token and logit trajectory identity off-site, include
parameter-matched generic and shuffled controls, and remain host-ALU-free. A
positive result would demonstrate a grammar-gated local serialization actuator,
not general reasoning or a new computational primitive.
