# R12 Wide Result-Digit Motor Result

**Protocol:** `R12-RDM-DEV-v1-disclosed-patch`

**Decision:** `REJECT_WIDE_RESULT_DIGIT_MOTOR_AS_AUTONOMOUS_ACTUATOR`

**Claim boundary:** This is an exploratory development-board rejection. It is
not a broad reasoning result and does not authorize a hardened confirmation
evaluation, architecture promotion, or further result-digit-only scaling.

## Custody

The interpretation contract was frozen and pushed in commit `b2f5acb` before
job `692235` printed any teacher-forced, autonomous, or cycle score. The job
completed on `evc37` in `02:00:54` with Slurm state `COMPLETED` and exit
`0:0`.

| Artifact | SHA-256 |
|---|---|
| Wide motor | `5b277e2797b9b4dee6bc0578e7891c5d0ae72d2217da74bf0fa1ab39df3b844a` |
| Autonomous evaluation report | `a308d707cf9890aeb8f6a7706a104b6cb451a344afb24403e5f518ba1abd01d0` |
| Base checkpoint | `d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459` |
| Tokenizer | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| Held-out episodes | `89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646` |
| Frozen cycle board | `0b927fee009de5e5cf87971ecaf390c716d6d9acb5644cabe3c176f6da9d4e7a` |

The report is mirrored locally at
`artifacts/r12/result_digit_motor_r2_eval/eval.json`, size 279,192 bytes, mode
`0444`, with the same hash as Newton. The disclosed motor-source manifest is
`dd94a00be3984d2f43d6342459ba9749e3164624bb9b96977bd19c11f23845f0`.
The report records evaluator-source manifest
`6775d73f1d2dd0ea1260a1a82a6fa1af3aced143df5165aac0378eb7c9496cb6`
and explicit noncanonical patch authorization. The exact evaluator, test, and
spooled batch hashes remain bound in the pre-outcome contract.

## Locked gate result

Rates pool all five 50-episode regimes by summing numerators and denominators.
Percentage-point margins are absolute.

| Gate | Base | Treatment | Shuffled | Required treatment margin | Result |
|---|---:|---:|---:|---:|---|
| Held-out digit top-1, 2,800 rows | 90.8571% | 91.3571% | 49.7500% | >=99%; >=3pp over base; >=10pp over shuffled | **FAIL**: +0.5000pp over base |
| First autonomous transition | 203/250 = 81.2% | 203/250 = 81.2% | 127/250 = 50.8% | >=5pp over both | **FAIL**: +0.0pp over base |
| Full autonomous state loop | 61/250 = 24.4% | 63/250 = 25.2% | 4/250 = 1.6% | >=2pp over both | **FAIL**: +0.8pp over base |
| Frozen-cycle first transition | 14/50 = 28.0% | 14/50 = 28.0% | 2/50 = 4.0% | >=5pp over both | **FAIL**: +0.0pp over base |

Mechanical and routing gates pass. The dead motor is behaviorally identical to
base on every pooled and per-regime count, treatment and shuffled fire exactly
once at each reported grammar site, non-DWS prompts have zero sites/fires and
decoded identity, and all bound input hashes match. Passing these checks does
not override the four failed capability gates.

The training-board result was 100% treatment digit top-1 versus 94.2375% base
and 61.65% shuffled. Its collapse to 91.3571% on the held-out feature board
shows that the 19,185,674-parameter motor fit a strong digit readout without
learning a robust new transition rule. The small closed-loop change is not a
hidden first-step gain: first-transition exactness is identical to base.

## Direct transcript diagnosis

The retained autonomous transcripts contain the same first 15 episode IDs for
base and treatment. Across their aligned rows, treatment changes zero decoded
responses, rescues zero rows, and harms zero rows. Every one of the seven
incorrect retained states differs from the target only in `c`; digit and all
other fields are already correct.

The frozen-cycle transcript for `fit_w4-00062` is the cleanest intervention
example. The expected next state is:

```text
dws:op=sub;w=4;p=3;c=1;a=5552;b=7452;r=8090;z=0
```

Base emits `r=8000;c=0`. Treatment changes the result register to the correct
`r=8090` but still emits `c=0`, so the complete state remains wrong. This is
exactly the predicted separation: the fitted motor can serialize a residual
digit, but it neither computes nor commits the carry/borrow state required for
composition.

A separate direct local interaction on 50 held-out first transitions reached
50/50 result digits but only 39/50 exact complete states for both base and
treatment. All 11 failures were carry/borrow errors, with zero treatment
rescues or harms. The autonomous report independently confirms that diagnosis
at larger scale.

## Consequence

Close the wide result-digit-only branch. Do not increase motor width or run the
mandatory hardened confirmation evaluator: the frozen advancement contract
failed before confirmation eligibility. The actionable target is a coupled
digit-plus-carry state transition with private pre-emission state commit,
host-ALU-free autonomous rollout, and an exactly matched ordinary recurrent
control. A digit actuator may remain a diagnostic component, but it is not the
missing reasoning mechanism.
