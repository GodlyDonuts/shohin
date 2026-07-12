# Shohin Reasoning Targets And Gaps

## Bottom Line

Shohin has a healthy 125.1M-parameter training run, not a demonstrated strong
reasoner. The stable loss curve and 99-100% H100 utilization establish
optimization health; they do not establish state tracking, instruction following,
or error correction. The live pretrain remains protected while diagnosis,
curriculum work, and SFT candidates stay pinned and isolated.

## Evidence

### Direct interaction

The direct transcript audit and 48-case prompt matrix show this is not primarily
a decoding or formatting problem:

- Raw 168k: 4/48 native Q/A, 4/48 direct-instruction, 0/48 chain-of-thought,
  and 5/48 one-shot cases.
- Raw 168.75k multi-turn audit: 1/6 initial, 1/6 after explicit review, and
  1/6 after receiving a verified intermediate fact. The only repeated success
  was a simple syllogism.
- When supplied `27 * 14 = 378`, the model did not reliably execute the
  remaining `+ 9` operation. Review requests repeated wrong calculations.

`train/deep_interaction_audit.py` extends this to the pinned 170k checkpoint. It
records initial answer, review, verified-state use, model-produced compact state,
and reuse of that state across arithmetic, base conversion, state updates, list
and string transformation, logic, counterexample correction, and a minimal
Python contract. The transcript is held-out diagnostic evidence, never training
data.

### Public and held-out results

- Raw 168.75k: GSM8K majority@4 5/100, greedy GSM8K 2/100, MATH-500 2/100,
  HumanEval 7/164, MBPP 0/100.
- V2 SFT: GSM8K majority@4 6/100, greedy GSM8K 14/100, MATH-500 6/100,
  HumanEval 6/164, MBPP 0/100; held-out procedural reasoning 90/800 = 11.25%.
- V4 r3: GSM8K majority@4 5/100, greedy GSM8K 14/100, MATH-500 1/100,
  HumanEval 2/164, MBPP 0/100. It is rejected for broad promotion, despite a
  useful 209/800 (26.125%) procedural held-out result. That score is above V2's
  90/800 on the same evaluator, but the two pilots start from different raw
  checkpoints, so it is evidence to investigate rather than a clean data-only
  attribution. Its matched direct matrix is still only 4/48 Q/A, 4/48 direct,
  4/48 CoT, and 10/48 one-shot; structured-looking derivations remain wrong on
  basic arithmetic and state tracking.

External scores require matching prompts, decoding, samples, and scorer. A custom
100-example board cannot support a claim to beat another model.

## Root Causes

1. **No language/instruction floor in pretraining.** The active equal-directory
   stream is about 75% math-oriented text and 25% raw code, with no material
   broad educational-English source. FineMath4 replayed roughly eleven times
   while FineMath3 had not completed one pass at 168k.
2. **Old resubmissions could repeat data prefixes.** Checkpoints did not record
   loader progress and restarted with the same seed. The forward-only stream
   generation fix prevents the next handoff from reusing the same ordering, but
   cannot make historic nominal tokens unique.
3. **SFT taught style more than algorithms.** The original mix was dominated by
   math derivations, had only hundreds of code rows, and covered too few
   procedural families.
4. **Code was scarce and prompt-misaligned.** Q/A code supervision did not match
   raw continuation evaluation. The prompt-boundary defect is fixed, but scale
   is still inadequate.
5. **Latent reasoning is untrained.** There is no compaction objective, trained
   recurrence, or test-time-loop policy in the flagship. KV caching is not
   self-summary. The short `n_loop=2` control was stable but 1.65x slower with
   no demonstrated capability gain.
6. **Scale is real.** At about 13.3B tokens/day, 2T tokens takes roughly 150
   days on the single H100. It only helps if data are unique, covered, audited,
   and followed by stronger post-training.

## Non-Negotiable Promotion Gates

1. **Data:** schema, deduplication, no held-out/eval overlap, answer or execution
   verification where applicable, source mix, replay report, and decoded-token
   quality scan.
2. **Atomic transfer:** new primitive data must materially improve on the
   disjoint 3,500-row primitive holdout, per family, without a broad regression.
3. **Broader reasoning:** balanced held-out procedural reasoning must improve on
   families absent from the candidate's training source.
4. **Public board:** GSM8K, MATH-500, HumanEval, and MBPP cannot materially
   regress; code must be evaluated with a represented prompt format.
5. **Direct interaction:** transcripts must show correct initial, review,
   verified-state, and compact-state-reuse behavior. Parser-only gains do not
   count.
6. **Generator/verifier:** report greedy pass@1, oracle@K, and verifier@K. Low
   oracle@K means improve generation; a large oracle/verifier gap means improve
   verification.

## Ordered Remediation

### Finish evidence before touching the flagship

- Complete V4's remaining held-out/direct/verifier work only as diagnosis; V4 is
  already rejected for broad promotion.
- Measure raw primitives, then run frozen V5. It must win the disjoint primitive
  gate before consuming a broader public board.
- Admit DCLM only after its manifest and full decoded-token scan both pass.
- After the 5B DCLM pilot passes, build and scan a 25B replacement before the
  long language-balanced phase. The replacement must be used instead of the
  pilot so the streamed prefix is not double-counted.

### Correct the next natural pretraining handoff

After every source passes its manifest gate, use the future-only curriculum:

- 50% audited educational English (FineWeb-Edu plus DCLM),
- 25% math/reasoning (FineMath, OpenWebMath, OpenMath),
- 25% code.

Start only from the newest numbered checkpoint with the distinct data stream,
the established BS32/ACC8 single-H100 configuration, and 524,288 tokens/update.
Never mutate the active job's shard list.

The loader reserves one sequence per active domain before applying weights. The
future script uses floor-aware weights verified by `test_domain_mix.py`; weights
that merely sum to 25/25/50 would silently produce the wrong batch mixture.

### Scale verified post-training by missing skill

- Use solver-generated primitives for atomic operations and their disjoint set
  for transfer measurement.
- Scale execution-verified code in the raw continuation format used at eval.
- Scale verified math, science, logic, and procedural traces to millions of rows
  with source caps so small groups are not replayed into memorization.
- Admit teacher outputs only after answer checking, decontamination, source
  labeling, and packing analysis.
- The official Apache-2.0 OpenR1-Math default subset is a promising verified
  math candidate. Its intake remains gated: retain only completed per-trace Math
  Verify results (with LLM-judge fallback), one trace per problem, concise output,
  and no evaluation-prompt overlap. A 10k pilot must pass before a full curation.

### Treat extended reasoning as separate research

First demonstrate atomic execution and useful oracle@K headroom. Then train a
small controlled recurrence or compact-state objective with an equally trained
non-recurrent control. A valid compaction feature needs state extraction,
validation, and continuation supervision; `<think>`, a KV cache, or an
inference-time loop does not supply that.

## External Target Reality

The target is intentionally strict. The official MobileLLM-R1 140M model card
reports roughly 6.2 MATH-500 and 4.1 GSM8K, while R1.5 reports 16.0 MATH-500 and
8.3 GSM8K under its own protocols. Their recipe includes far larger quality
pretraining, reasoning SFT, and later on-policy distillation. These are targets
to reproduce under matched protocols, not numbers to compare against a custom
100-example board.

Sources: [MobileLLM-R1 140M model card](https://huggingface.co/facebook/MobileLLM-R1-140M/blob/main/README.md),
[MobileLLM-R1.5 140M model card](https://huggingface.co/facebook/MobileLLM-R1.5-140M/blob/main/README.md),
and the [official MobileLLM-R1 repository](https://github.com/facebookresearch/MobileLLM-R1).

## Current Decision

Continue the protected 1-H100 pretrain because it is stable and remains useful
for the forthcoming language-balanced curriculum. Do not call it intelligent yet
and do not promote V2 or V4. The next claim of progress must be supported by the
pinned interaction transcript, primitive transfer, and protocol-correct benchmark
movement together.
