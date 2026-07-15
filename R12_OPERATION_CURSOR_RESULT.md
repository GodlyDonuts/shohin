# R12 Raw-260k Operation-Cursor Result

**Status:** complete, immutable, interface-confounded negative diagnostic.

## Bottom line

Newton job `689717` completed all 528 frozen greedy calls against immutable raw
260k. Strict whole-response JSON parsing succeeded in `0/528` calls. Every
call consumed the full 32-token cap and none stopped at EOS. The preserved
semantic counters are therefore all zero by contract.

This is strong evidence of instruction-format and termination failure on this
interface. It does **not** establish that the model lacks a next-operation
preference, because operation selection was never observed independently of
JSON compliance, operand emission, free decoding, and stopping. No semantic
score may be salvaged post hoc from the malformed transcripts.

## Custody and accounting

| Object | Value |
|---|---|
| Newton job | `689717` on `evc42` |
| Slurm state | `COMPLETED`, exit `0:0`, elapsed `00:11:17` |
| Cases / transitions | `64 / 176` |
| Model calls | `528` |
| Prompt tokens | `53,928` |
| Sampled / decoded tokens | `16,896 / 16,896` |
| Repairs / retries / searches / verifier calls | `0 / 0 / 0 / 0` |
| Result SHA-256 | `5ba772ec68aaa445d1252022f00285fa83b3403f3376437d4386d143619da681` |
| Local result | `artifacts/eval_history/raw260k_operation_cursor_20260715.json`, mode `0444` |
| Preserved log SHA-256 | `5696cac7e447450f115a7f3910fe97904dacb43def23c91f6b633a6b026260c6` |

## Exact arm results

| Arm | Parse success | Parse errors | Semantic score |
|---|---:|---|---:|
| Full source plus cursor | `0/176` | 175 invalid JSON, 1 wrong key set | `0/176` selection |
| Residual suffix selector | `0/176` | 176 invalid JSON | `0/176` selection |
| Residual suffix plus oracle state | `0/176` | 176 invalid JSON | `0/176` joint selection and update |

All `528/528` responses ended at `max_new`; EOS stops were `0/528`. Typical
outputs expanded into operation lists, table-like continuations, extra keys,
numeric strings, or prose instead of the requested exact object. These raw
responses remain in the immutable result.

## Decision

The result closes this strict free-decode interface as a controller gate. It
does not authorize a controller fit and must not be converted into a latent
operation-selection claim. The next admissible measurement is a separately
preregistered one-forward, four-candidate operation-likelihood diagnostic that
removes JSON, operand, generation, and EOS confounds. Its draft implementation
is incomplete; it was not run or submitted during this result.
