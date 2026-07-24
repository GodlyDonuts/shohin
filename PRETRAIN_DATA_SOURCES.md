# Shohin pretraining data admission plan

Research snapshot: **2026-07-21**. Planning artifact only: this file does not change the active data build or training mix. The machine-readable companion is [`pipeline/pretrain_sources.json`](pipeline/pretrain_sources.json).

## Executive decision

High-quality pretraining data is necessary, but it is not sufficient by itself to create reasoning. For a 130–140M model, the winning combination is:

1. a clean language and knowledge substrate;
2. unusually dense math, code, science, and procedural text;
3. verified reasoning traces during cooldown/SFT;
4. strict deduplication and evaluation decontamination; and
5. an architecture and token budget capable of learning the signal.

We should **not** build a giant undifferentiated pile. At this model size, weak or repetitive tokens displace useful tokens. The default admission plan below uses selected subsets, caps synthetic data, preserves source provenance, and globally deduplicates overlapping web/math/code families.

**Research-use policy:** licensing does not affect the quality ranking or block local, non-commercial training. Public and gated sources are both eligible. Redistribution is a separate operation, so gated rows stay in their original repository rather than being copied into `Godlydonuts/shohin`.

## Recommended stable-pretraining mix

This is the quality-first target for the next full pretraining tranche. Percentages are token-presentation shares, not raw download sizes.

| Domain | Share | Sources |
|---|---:|---|
| Educational/general web | 45% | FineWeb-Edu score 4–5 (22%), selected Essential-Web (10%), Nemotron-CC-v2.1 High-Quality organic (8%), DCLM residual (5%) |
| Math | 25% | UltraData-Math L2 English (7%), selected L3 QA/textbook English (7%), Nemotron-CC-Math 4plus (5%), FineMath 4+ residual (3%), MegaMath Web-Pro residual (2%), OpenWebMath residual (1%) |
| Code | 20% | StackV2-Edu (10%), Nemotron-CC-Code quality 3 (6%), selected Nemotron Code-v2 synthetic (2%), first-party unit-tested code (2%) |
| Science/procedural | 10% | first-party verified procedural data (4%), selected Nemotron Specialized-v1 (3%), StackExchange (1%), and a capped LibreTexts/arXiv/peS2o blend (2%) |

For another 300,000 steps at the current global batch/sequence contract, aim for at least **100–120B admitted unique tokens** and sample them into the required token-presentation budget. Do not manufacture the pool size by repeating weak sources. Track presentations, unique tokens, epochs per source, and duplicate rate separately.

The 45/25/20/10 split is a starting hypothesis. Freeze the validation suite first, then run small controlled source ablations before committing the full tranche.

## Source decisions

### P0 — acquire and admit first

| Source | Exact slice | Why it belongs | Admission conditions |
|---|---|---|---|
| [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | English, `int_score >= 4`; use score 3 only if an ablation earns it | 1.3T-token educational web corpus; model-based educational scoring and strong published ablations | HTML/boilerplate and exact/near dedup; decontam; cap dominant domains |
| [EssentialAI/essential-web-v1.0](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) | High technical correctness, conceptual/procedural cognitive type, nontrivial reasoning depth; use taxonomy math/code/STEM views as selectors, not extra independent corpora | 24T-token pool with unusually rich quality, subject, education, reasoning, math, and code metadata; globally deduplicated upstream | Dedup against FineWeb/DCLM because all derive from Common Crawl; manually audit each selection rule |
| [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | Baseline pool, domain-capped | Strong open web baseline produced by model-based filtering; useful linguistic and topic breadth | Lowest retention priority among the three web pools; retain only cross-source residual |
| [openbmb/UltraData-Math](https://huggingface.co/datasets/openbmb/UltraData-Math) | `UltraData-Math-L2-preview` plus English records from L3 QA and Textbook-Exercise; optionally a small multi-style slice | Current high-density math corpus. Its card reports stronger 1.2B ablations than Nemotron-CC-Math 4plus at comparable scale | English/language filtering; format validation; cap synthetic templates; globally dedup against Nemotron, MegaMath, FineMath, and OpenWebMath because L3 uses some of them as seeds |
| [common-pile/stackv2_edu_filtered](https://huggingface.co/datasets/common-pile/stackv2_edu_filtered) | Educational files in Python, C/C++, JavaScript/TypeScript, Rust, Java, Go, and shell | Roughly 67.8B tokens in the Comma mix; educational filtering and per-file provenance metadata make it a better default than ingesting all of The Stack | Parseability, secret/PII scan, repository split, near-dedup, generated/vendor code removal, per-record provenance retention |
| [nvidia/Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1) | `High-Quality` organic records; evaluate `High-Quality-DQA` separately | Adds 26B recent high-quality organic web tokens and 8B STEM DQA tokens; useful freshness and complementary filtering | Use directly from upstream; dedup against Essential-Web because DQA is derived from it; do not flood the mix with the 2.1T medium-high synthetic rephrases |
| [nvidia/Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) | `4plus` | Strong 133B-token math family and an important complementary ablation against UltraData | Direct upstream access; global math dedup; keep only if the mixed pilot beats UltraData-only |
| [nvidia/Nemotron-CC-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1) | Quality-3 records | Approximately 428B tokens of processed Common Crawl code pages; broader natural-language/code coverage than repository code alone | Direct upstream access; syntax/quality checks; dedup against code web pages and generated explanations |
| First-party verified procedural data | Reasoning Gym and Shohin-native tasks whose answers can be programmatically checked | Exact difficulty control, clean provenance, and signal aligned to the tiny model | Keep generators, seeds, verifier version, and pass/fail evidence; remove any eval-equivalent task instances |
| First-party verified code | Unit-tested programs plus concise problem/explanation pairs | High information density without relying on uncontrolled web code | Execute in a sandbox; retain test evidence; repository/problem split; reject copied benchmark tests and solutions |

### P1 — admit only the named residual or specialized slice

| Source | Decision |
|---|---|
| [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | Keep `finemath-4plus` residual after UltraData dedup. Do not sample 3+ and 4+ as independent streams: 4+ is nested inside the broader family. |
| [LLM360/MegaMath](https://huggingface.co/datasets/LLM360/MegaMath) | Prefer `megamath-web-pro`; admit only residual high-quality documents. Do not ingest all 371.6B tokens merely for scale. |
| [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) | Preserve a small residual-diversity slice. It remains useful, but its 14.7B tokens substantially overlap newer math mixtures. |
| [common-pile/stackexchange_filtered](https://huggingface.co/datasets/common-pile/stackexchange_filtered) | Admit answer-rich technical/scientific threads, with attribution and source-specific CC BY-SA obligations preserved. Remove low-signal social/meta pages. |
| [common-pile/libretexts_filtered](https://huggingface.co/datasets/common-pile/libretexts_filtered) | Admit educational textbook chapters after formatting and attribution checks. |
| [common-pile/arxiv_papers_filtered](https://huggingface.co/datasets/common-pile/arxiv_papers_filtered) and peS2o from [Common Pile/Comma](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset) | Use a capped science-method slice. Do not let equation-dense papers crowd out explanatory science. |
| [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) | Do **not** ingest the full legacy code pool. Consider notebooks, issues, and commits for natural-language/code interaction only; dedup them against StackV2 and remove low-signal repository exhaust. |
| [nvidia/Nemotron-Pretraining-Code-v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2) | Admit selected QA, code-review, student-teacher, rewriting, and transpilation subsets. The raw-GitHub portion is metadata, not usable code text. Cap this synthetic slice at 2% initially. |
| [nvidia/Nemotron-Pretraining-Specialized-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1) | Admit selected InfiniByte reasoning, scientific coding, math-textbook, and sampled RQA records during late pretraining/cooldown. Start at 3%; long-form RQA must earn more weight in an ablation. |

### NVIDIA access and storage boundary

The gated NVIDIA cards explicitly intend the corpora for model training, and the collection card describes the release as ready for commercial use. We will therefore use the best slices in local research instead of letting access terms lower their quality ranking. Their data agreement still distinguishes training from redistributing the corpus: do not copy gated records or reconstructive tokenized shards into `Godlydonuts/shohin`. Store upstream IDs, revisions, selection manifests, non-reconstructive hashes, and audit results there.

| Source | Selected use | Important correction |
|---|---|---|
| [nvidia/Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1) | High-Quality organic and a separate DQA pilot | Avoid the enormous medium-high synthetic-rephrase pool until it proves value at 140M |
| [nvidia/Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) | `4plus`, direct from upstream | Globally dedup because newer math corpora reuse overlapping seeds |
| [nvidia/Nemotron-CC-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1) | Quality-3 records, direct from upstream | This is the processed text corpus; approximately 428B tokens |
| [nvidia/Nemotron-Pretraining-Code-v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2) | Selected synthetic text configurations | Raw-GitHub records are metadata; QA/review/student-teacher/rewriting/transpilation contain usable text |
| [nvidia/Nemotron-Pretraining-Specialized-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1) | InfiniByte, scientific coding, math textbooks, and sampled RQA | Prefer v1's named reasoning/STEM slices over indiscriminate use of v1.2 |

These are included in the proposed mix, not merely held in reserve. Each still has to pass an equal-token quality ablation; public availability is not evidence that every subset is useful.

### Reject or hold by default

| Source | Decision and reason |
|---|---|
| [togethercomputer/RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | **Reject for the new build.** It was an important 2023 LLaMA reproduction corpus, but newer filtered pools are stronger and cleaner. Its mixed source licenses and overlap add work without a clear residual-quality case. |
| [nvidia/Nemotron-Pretraining-Code-v3](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v3) | **Reject as direct training data.** The Hugging Face release contains metadata/index records for GitHub files, not the source-code text. It can be an acquisition index only, subject to repository licenses. |
| [nvidia/Nemotron-Pretraining-Specialized-v1.2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1.2) | **Hold.** It is primarily factual-recall, moral-scenario, generative, and multiple-choice synthetic data—not the reasoning substrate we need. A small fact-seeking slice may be reconsidered only after an ablation. |
| Full StarCoderData or The Stack v2 | **Reject as an undifferentiated stream.** Volume, duplicates, generated/vendor files, credentials, and heterogeneous licenses are all costly at 140M. Use selected educational or NL-code subsets instead. |

## Mandatory admission pipeline

No dataset receives a training weight merely because it downloaded successfully.

1. **Pin and inventory:** exact upstream repo, config, split, immutable revision, row count, byte count, and upstream license/terms snapshot.
2. **Normalize with provenance:** every document keeps source, upstream ID, URL/repository where permitted, license, selection score, and transformation history.
3. **Structural quality gates:** language ID, decoding/Unicode checks, repetition, boilerplate, document length, equation/code parseability, PII/secrets, and source-specific rules.
4. **Within-source dedup:** exact hash, paragraph dedup, and MinHash/LSH near-dedup; repository-level splits for code.
5. **Cross-source priority dedup:** verified first-party > UltraData L3 > UltraData L2 > Nemotron Math 4plus > FineMath 4+ > MegaMath Web-Pro > OpenWebMath; verified code > StackV2-Edu > Nemotron CC Code > Nemotron Code-v2 > StarCoder ancillary; Essential selected > Nemotron-CC-v2.1 HQ > FineWeb-Edu 4–5 > DCLM residual. Dedup the whole admitted pool, not each source in isolation.
6. **Evaluation decontamination:** match against all benchmark prompts, reference answers, solutions, tests, common paraphrases, and held-out generator families before tokenization. Include GSM8K, MATH/MATH-500, ARC, HellaSwag, PIQA, HumanEval, MBPP, TACO/CodeContests holdouts, and Shohin-native evaluations.
7. **Shard audit:** random human inspection plus quantitative reports by source and domain. Fail closed on missing provenance, corrupted records, secrets, or failed quality checks; license labels are retained but do not gate non-commercial research admission.
8. **Tokenize and account:** record unique pre-tokenization documents, unique Shohin tokens, presentation tokens, epochs, packing waste, and rejection reasons.
9. **Ablate:** compare equal-token pilots using frozen validation sets. Promote a source only if its domain gain is not paid for by unacceptable general-language or contamination regressions.

## Hugging Face storage policy

`Godlydonuts/shohin` should be the control plane, not automatically a mirror of every upstream corpus.

Safe default layout:

```text
registry/pretrain_sources.json
manifests/<source>/<upstream_revision>.jsonl
audits/<source>/<build_id>/{license,quality,dedup,decontam,shards}.json
generators/<first_party_corpus>/<version>/...
data/<source>/<build_id>/...          # only when redistribution was explicitly cleared
```

- Publicly store the source registry, immutable revisions, selection recipes, aggregate statistics, and non-reconstructive audits.
- Store first-party generated records when every generator/input license permits it and verification evidence accompanies the release.
- Store third-party data shards only after a source-specific redistribution review and all attribution/share-alike/removal requirements are implemented.
- Never mirror gated NVIDIA records. Do not assume a private Hugging Face repository overrides the upstream agreement.
- For code, preserve the per-file license and provenance through tokenization; an aggregate dataset-level label is not enough.

## Immediate build order

1. Freeze the benchmark and native-evaluation contamination sets.
2. Acquire small audit samples from FineWeb-Edu, Essential-Web, DCLM, Nemotron-CC-v2.1, UltraData L2/L3, Nemotron Math 4plus, StackV2-Edu, Nemotron CC Code, and the selected specialized slices.
3. Implement one shared normalized document schema and cross-source dedup keys.
4. Produce 0.5–1B-token pilot mixes and run equal-token ablations from the same checkpoint.
5. Scale only admitted sources to the 100–120B unique-token target.
6. Keep the active 300k checkpoint and live relaunch configuration unchanged until these pilots pass.
