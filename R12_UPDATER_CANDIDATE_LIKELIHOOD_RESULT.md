# R12 Raw-260k Updater Candidate-Likelihood Result

**Status:** complete, independently replayed, negative under the frozen candidate set.

## Bottom line

Raw 260k does **not** prefer the exact correct residual state-and-tail update in
any of the six frozen prompts. The correct candidate is the unique winner in
`0/6` prompts by normalized candidate likelihood, total candidate likelihood,
or candidate-plus-EOS likelihood. EOS is unique top-1 immediately after the
correct candidate in `0/6` prompts.

This rejects the narrow hypothesis that free decoding merely hides an already
preferred updater behind sequence-length or termination pressure. It does not
prove that no other rendering or hidden updater representation exists.

## Custody

| Object | SHA-256 |
|---|---|
| Result | `4ca100029806c933ba1d3137044c040b468d380ae9bb9f5efeadcbc949374525` |
| Prompt source | `4505602994a0e337b99359e580a6f2f04fad4d365b2dac59f4c339fac13a7593` |
| Raw-260k checkpoint | `91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d` |
| Tokenizer | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| Candidate manifest | `0e01fc54abfe63dcfd063fa6d5a1e4ed46b57aef617580d2cc286db839b3ba98` |
| Tokenized manifest | `13528bacb21d8bca006b434283830d9a0790225dd9a7ddcdcbc8f02e7bac8a99` |

Canonical local artifact:
`artifacts/eval_history/raw260k_updater_candidate_likelihood_20260715_mps.json`.
It is mode `0444`.

The executable and preregistration were committed as `2ad9127` before the
artifact was created. The v1 result schema does not embed those two source
hashes; this is a custody limitation to fix in any successor protocol, not a
reason to alter this result.

## Exact results

The gap is winner minus correct normalized log likelihood in nats per token.

| Prompt | Frozen winner | Correct rank | Gap | EOS rank after correct |
|---|---|---:|---:|---:|
| `joint_a` | arithmetic execution continuation | 4/5 | 0.879489 | 5 |
| `joint_b` | arithmetic execution continuation | 2/5 | 0.832254 | 5 |
| `joint_c` | arithmetic execution continuation | 4/5 | 0.941860 | 9 |
| `packet_a` | unchanged source packet | 3/5 | 0.516865 | 10 |
| `packet_b` | unchanged source packet | 3/5 | 0.426724 | 6 |
| `packet_c` | unchanged source packet | 3/5 | 0.324967 | 13 |

Mean winning gap: `0.653693` nats/token. All six row diagnoses are
`correct_update_not_preferred`; the aggregate is
`correct_update_not_consistently_preferred`.

The format split is informative but not causal proof. Natural `Work:` prompts
prefer continuing arithmetic, while explicit packet prompts prefer copying the
unchanged packet. Therefore a follow-up should exploit the arithmetic path or
measure operation selection directly; it should not train a packet-rewrite
mechanism on the assumption that one is already latent.

## Independent replay

An independent process replayed all 30 MPS forwards from the bound checkpoint.
The maximum candidate-token and EOS log-probability differences were both
exactly `0.0`. It recomputed every token alignment, total, normalized score,
EOS rank/margin, winner, diagnosis, summary count, and ledger field. Focused
tests passed `11/11`.

The frozen ledger is 12 source rows, six scored prompts, 30 model forwards,
1,080 replayed prompt tokens, 518 candidate tokens, 30 EOS targets, 548 total
teacher-forced targets, and 1,598 forward positions. There was no generation,
sampling, retry, candidate search, response-derived candidate construction, or
cross-candidate KV state.
