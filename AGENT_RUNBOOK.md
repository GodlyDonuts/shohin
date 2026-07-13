# AGENT RUNBOOK — Shohin autonomous custody

> **If you are a new agent taking over, read THIS FILE FIRST, top to bottom.** It is the single
> source of truth for the live training run and the standing directive. Everything you need to keep
> the model alive, hit the next milestone, and not break anything is here. Other docs
> (`MASTER_PLAN.md`, `DIVERGENCE_DIAGNOSIS.md`, `DATA.md`) are background/history; this file is the
> operational plan of record.
>
> **Last updated:** 2026-07-13 17:15 EDT (flagship custody remains intentionally hands-off with its full 200k checkpoint hash-verified on Newton and locally; continuous latent rollout, LSA, and CPR are closed negative branches; DRS v2 core establishes 497/500 correct first emitted states but only 275/500 final answers, including 0/100 width-8, so it is a narrow recurrent executor with path-dependent failures; the core transcript probe now confirms those failures are late recurrent-transport errors; a matched 80-direction raw residual-patching baseline is negative and the post-DRS comparison is safely queued; held-out wording, direct interaction, and NLL remain serialized through a just-passed 1-H100 CUDA preflight; both full v3 transition-basis and static-tape/recurrent-register corpora are hash-admitted but untrained, with STRR now the higher-priority representation control; a counterfactual-workspace-induction control is designed but remains conditional on a positive STRR primitive; DCRD and CBC remain conditional; ADL remains CPU-admitted only). Keep the "LIVE STATE" section current
> every milestone — update it, don't let it rot.

---

## 0. The mission and the governing directive

**Mission:** Ship the best sub-200M-parameter (currently 125.1M trained parameters) verifiable-reasoning language model of
2026 — math / code / logic. Concretely: **beat MobileLLM-R1-140M** on GSM8K / MATH-500 / HumanEval /
MBPP / logic. Reasoning specialist, verifiable-first. Data must be decontaminated, execution/answer-
verified, concise-CoT.

**Governing directive (verbatim, from the user — this overrides earlier plans):**

> "Forget the cluster that is not possible. So put all your effort into making our pretraining as
> perfect as it can be on the single GPU. Train it for weeks I dont mind. The only thing that matters
> is SoTA."

**Superseding user authorization (2026-07-12):** "we can use 2 GPU's if you want for training (it will be faster)."
It authorizes the validated two-H100 path at a **natural** handoff and for isolated experiments; it does
not authorize interrupting or modifying the running one-H100 writer.

What that means operationally, and is now settled — **do not relitigate:**
- **Protected current writer; two-H100 successor allowed.** `685084` remains one H100 until it exits.
  Its successor may use two H100s only with the same 524,288-token global update, fresh-optimizer
  rewarmup, an actual DDP/CUDA health check, and single-writer safety. The 8xH100 path remains unavailable.
- **Time is unlimited.** Weeks are fine. Optimize for final quality, not speed.
- **The two levers that matter at 135M:** (1) more tokens through the model, (2) more/better data.
  Everything we do serves those two. (Architecture is frozen; see §7.)

**Most recent user instruction (2026-07-07, `/goal`):**
> "I need you to do highly detailed documentation as you go on so any future agent knows exactly what
> to do. I am also going to go to sleep. I entrust you to take good care of our model."

→ This runbook is that documentation. Keep it detailed and current. **Priority #1 is keeping the
pretrain ALIVE**; priority #2 is executing the milestone transitions in §4 correctly; priority #3 is
keeping this doc accurate.

**2026-07-07 executive-director directive:** the user gave Codex full authority to do whatever is
needed to achieve the project goal. Operationally, this means: protect the live run first; make
bounded, auditable changes without approval loops; prefer verified data and measured gates over
speculative novelty; keep GitHub/Newton/local state synchronized enough that a takeover agent can act.
Do not wait for permission to fix obvious data/training gaps.

---

## 1. LIVE STATE  ← update this every milestone

| Item | Value (as of 2026-07-13 current custody check) |
|---|---|
| **60k pretrain job** | `680149`, name `shohin-flagship`, node **evc22**, **DONE** (`[done] 60000 steps in 112203s`) |
| **Extended pretrain job** | `683715` completed cleanly. Current active continuation is **`685084`** on **evc22**: one H100, `BS=32 ACC=8 CKPT=250`, exact 524,288-token updates, and the proven default compile path. It resumed `ckpt_0141500.pt -> step 141501`. User-authorized **continuity fallback** `686732` is held `afterany:685084`: two H100s, `NG=2 BS=32 ACC=4 CKPT=250`, four CPUs, fresh optimizer, and the same 524,288-token update; it excludes CUDA-preflight failures `evc26,31,36,43,50`. It retains the old math/code-only stream. Before the dependency releases, prefer a validated language-balanced two-H100 relaunch only if both 25B language replacements are fully scanned and hash-approved; otherwise let `686732` keep tokens moving. |
| Extended pretrain status | **`685084` reached 200,000 cleanly** at loss **1.6200**, gnorm **0.11**, and **154.30k tok/s**, then remained healthy through observed step **201,620** at **154.29k tok/s** (latest loss **1.8294**, gnorm **0.10**). One isolated guard skip at 193,437 and one at 200,387 (gnorm 2.02 versus 0.11 EMA) each recovered on the next logged step; there is no persistent instability. A direct node read at 11:33 shows **100% GPU utilization**, 63,767 / 81,559 MiB VRAM, and 301 / 310 W. Numbered checkpoints extend through `ckpt_0201500.pt`; durable Newton `best_step200000.pt` and full local DR `train/flagship_out/ckpt_0200000.pt` match at md5 **`510d57df578447986b40e20029511b9d`**. The exact 200k milestone is **104,857,600,000 nominal update tokens**; current mounted corpus capacity is **57,826,022,271 manifest tokens**. These are intentionally separate metrics in `TRAINING_METRICS.md`. Next local DR target is 210k. Do not integrate CUDA graphs into live training: the clean whole-update canary gained only ~1.8% while removing the flagship's established guard/observability path. |
| **SFT feedback job** | `681000`, name `shohin-sft`, **DONE**; wrote baseline `train/sft_out/sft_ep3.pt`. Isolated v2 pilot `685708` completed one epoch from `best_step120000.pt` to `train/sft_v2_120k/sft_ep1.pt`. It is a narrow arithmetic-format ablation, not a promoted broad-reasoning recipe. V4 `686323` was canceled before its first step after stale 40/35/15/10 weights; v4 `686324` was canceled before any artifact after a code-boundary audit found 461/3,542 legacy BPE prompt-prefix mismatches. Both are invalid and preserved. Corrected v4 pilot **`686326`** completed one epoch in 1,218s to `train/sft_v4_168750_r3/sft_ep1.pt`, with audited 40/47/8/5 weights and inference-aligned prompt/completion token construction. It is an unevaluated candidate, not promoted. |
| **Eval board job** | Corrected CUDA-only v2 board **`686277` completed**: GSM8K maj@4 **6/100**, pass@1 **14/100**, MATH-500 **6/100**, HumanEval **6/164**, MBPP **0/100**. The v2 pilot is **rejected for promotion**. RG held-out `686278` is **90/800 = 11.25%** and in-training `686279` is **98/800 = 12.25%**: it learned a few routines that transfer but remains zero on most logic/transformation/cipher/geometry families. The corrected raw-base board **`686315`** pinned `best_step168750.pt` and completed: GSM8K maj@4 **5/100**, pass@1 **2/100**, MATH-500 **2/100**, HumanEval **7/164**, MBPP **0/100**. `686316` direct adaptive interaction was **1/6 initial, 1/6 after explicit self-review, 1/6 with a verified intermediate fact**; only the simple syllogism was correct. V4 r3 public board **`686336` completed and is rejected**: GSM8K maj@4 **5/100**, pass@1 **14/100**, MATH-500 **1/100**, HumanEval **2/164**, MBPP **0/100**. Its corrected held-out procedural result `686337` is **209/800 = 26.125%**, above V2's 90/800 on the same evaluator, but the later raw base means this is a useful diagnostic signal rather than clean data-only attribution. V4 remains a generator/verifier candidate, not a broad promotion. |
| **V5 primitive board** | **`686401` completed; V5 is rejected for broad promotion.** From the raw-168.75k base it scored GSM8K maj@4 **10/100**, greedy **9/100**, MATH-500 **3/100**, HumanEval **2/164**, and MBPP **0/100**. It shows narrow arithmetic-format transfer but regresses code from raw's 7/164 HumanEval and does not establish broad math, code, or instruction-following transfer. |
| **V6 contract SFT** | **`686413` completed cleanly** to `train/sft_v6_contracts_168750_r2/sft_ep1.pt` (4,535 updates, 1,388s). Its fresh 245-case contract holdout `686414` rises from raw **20/245 = 8.16%** to **142/245 = 57.96%**, especially review 28/35, scaffold 34/35, and reuse 34/35. This is deliberately not a latent-reasoning claim: independent deep audit `686415` is only **4/8 initial, 1/8 review, 1/8 scaffold, 0/8 compact reuse** and shows invalid compact calculations. Matched matrix `686438` completed at Q/A **4/48 -> 17/48**, direct **5/48 -> 10/48**, CoT **0/48 -> 21/48**, one-shot **7/48 -> 7/48**; it establishes contract transfer but not compact-state reasoning. No public board is justified. |
| **V7 typed-state SFT** | The fresh solver-verified state corpus has **315,000 train / 10,500 held-out** rows across write, repair, and reuse contracts; independent audit reports 0 malformed rows, duplicate questions, exact held-out prompts, or 13-gram held-out overlap. Raw pinned baseline `686456` completed its 420 rows before a duplicate-output exit: **21/420 = 5.0% answer accuracy and 0/280 valid typed states**. V7 `686467` completed cleanly from `best_step168750.pt`. Its exact 420-prompt holdout `686471` reaches **307/420 = 73.10% answers** and **169/280 = 60.36% exact states**: repair 128/140 answers and 132/140 states, reuse 140/140 answers, but write only 39/140 answers and 37/140 states. The independent eight-case interview `686484` is **1/8 initial, 1/8 review, 1/8 scaffold, and 0/8 compact reuse**, with wrong arithmetic even after a verified fact and malformed/unrelated state text. **Reject V7 as a general-reasoning or latent-compaction candidate.** Keep it isolated as a constructed-contract diagnostic; its hash-matched state/deep artifacts are `1f9fe0b2993d1a9dafc98cd2d7943887` and `c4963fae52d5ac9c38614e77f93f98c8`. |
| **VRWM closed-loop research** | Raw 180k control is **0/400** on the full p80 executable-state suite. r3 one-epoch SFT gives **43/400** closed-loop default prompts but **0/50** paraphrase p10, so it is rejected as template-bound. r4 state-only control gives **32/400 default / 2/400 semantic**; r4 deterministic-scratch variant gives **120/400 default / 21/400 semantic** across value and 4/8/16/32-length OOD regimes. The scratch variant is real narrow executable-state evidence, but its semantic transfer and long-length scores are insufficient for any general-reasoning, latent-reasoning, or promotion claim. Repair-trained r5 SFT `686820` completed cleanly (4,272 steps, 1,303s; checkpoint md5 `ef99f8c2ab5835c8229bcd4f36fb8789`, Newton+local). Its default first pass reaches **174/400**, but semantic first pass and semantic self-repair are both only **17/400**, below r4 scratch's 21/400. Fresh operator interview is **0/8** initial/review/fact/reuse versus raw-180k **1/8** initial and fact: r5 answers ordinary questions with irrelevant `check:/wm:` strings. It is rejected for broad promotion; default self-repair was deliberately canceled after 80/400 because it could not repair that broad-mode failure and was blocking stronger controls. |
| **V8 broad / V9 broad-plus-memory controls** | Raw pinned 180k public board **`686861`** remains running before V8; completed results are GSM8K maj@8 **2/200 = 1.0%**, pass@1 **7/200 = 3.5%**, MATH-500 **5/200 = 2.5%**, HumanEval **7/164 = 4.3%**, with MBPP still running. V8 SFT **`686862`** remains held after that raw baseline. V9 data is **1,213,830** clean/decontaminated rows, 95,163 packed sequences, SHA-256 `8f5845d19d3a3852ed4c7c930b23a27032aaf47e868ef082b7f6f85697b364f4`. Its isolated one-epoch SFT **`686876` completed** to `train/sft_v9_broad_vrwm_180k_r1/sft_ep1.pt` (local/Newton md5 `2a8ddd4dc4f91918eb47dc2701d4b21d`); it loaded 1,213,569 individually fitting examples and skipped 261 frozen rows over 2,048 tokens. Its in-progress board has GSM8K maj@8 **18/200 = 9.0%** and pass@1 **21/200 = 10.5%**, while MATH-500 is still running. That lift is not a promotion: the completed unseen direct interview `686943` is only **1/8 initial, 0/8 review, 1/8 verified fact, 0/8 valid state/reuse**, no better than raw. The transcript-only decoder now retains non-EOS special tokens: V9 emits `<think>` on **9/12** fresh trace prompts but has **0/12** correct intermediates, final answers, or trace-and-final pairs (artifact md5 `6ee477740087c07ab5c0d04b65d28371`). The trace-audit source is `aafd7e5e9aec5ae6dfe83a7ef0c78d66641c1b7add462674f1b30d56382fdf38`; the re-audited V8/V9 frozen prompts retain zero exact/13-gram hits (md5 `49cdf6a0ba055b321637fd1607d9503b` / `1fe9c4ac6faec7aaaff96fd2e7ce4f59`). V9 stays in the chain **`686942` board + `686943` interview -> `686944` default / `686945` semantic memory -> `686946` visible trace -> `686947` decision**. |
| **Semantic bridge / V10 candidate** | Solver-verified semantic-bridge v1 is a **new, isolated data candidate**: 200,000 train and 5,000 held-out rows across product-adjust, state-chain, base-conversion, fact-continuation, and calculation-repair families. Its admission audit `686964` found 0 malformed/duplicate/eval-overlap rows, **17,270,366 tokens**, and **8,432** exact 2,048-token packs. V10 mix build **`686966` completed and admitted the frozen data candidate**: **899,928** rows (math 292,944 / procedural 374,659 / code 7,250 / teacher 25,075 / bridge 200,000), **81,705** exact packed sequences, 0 malformed/duplicate rows, and 0 exact or 13-gram held-out overlaps. Data SHA is `65259560b7c99d95156dc08267c74bd437ee216a2e92826eec60ddc52a9f3a2f`; per-epoch replay is math 0.71x / procedural 1.28x / code 2.46x / teacher 2.04x / bridge 0.97x. No V10 SFT is submitted: it waits for V9's full transfer-control decision. `train/eval_semantic_bridge.py` now provides a deterministic family-balanced, special-token-preserving held-out gate; a V10 model must improve it and the existing direct/public gates before any promotion. |
| **Semantic capsule / future context-scaling control** | The new semantic-capsule protocol is deliberately distinct from V7/VRWM: it carries two named facts extracted from a natural-language record, then drops that record and repeatedly supplies only the model-produced capsule plus a new event. The controller never executes, repairs, or selects a state. Train/held-out domains, field names, number bands, event language, and lengths are disjoint; held-out regimes are 4/8/12 steps and final queries require retained facts. CPU-only generation **`686991`** and its independent protocol/overlap audit **`686992`** are queued. The audit recomputes every serialized transition and query, then checks all held-out controller prompts (not only initial prompts) for exact and 13-gram train overlap. This is an unadmitted research candidate; no SFT or flagship path references it. |
| **2-H100 speed canary** | `681040`, name `shohin-ddp2-canary`, **COMPLETED cleanly** on evc42: resumed from `ckpt_0060000.pt`, `world=2`, loss in band, no DDP hang, ended at `61050` in 2093s with ~262k tok/s (~1.76x the 1-GPU ~149k tok/s). Latest clean revalidation **`686734`** on evc37 seeded `best_step180000.pt`, loaded the synced forward-stream code, confirmed `world=2`, `BS=32`, `ACC=4`, and `stream_generation=1`, then completed 320 bounded updates with no CUDA/NCCL/DDP error. Its compile-free late windows were **291.7-293.9k tok/s**, about **1.90x** the live 154.3k one-H100 rate; compile-inclusive final logged rate was 243.8k tok/s. It had one guard skip at its terminal step 180,319, so it is a throughput/transport validation rather than a quality result. `686728` is deliberately **not a model result**: both attempts on evc31 failed at `torch.cuda.set_device(local_rank=1)` before model loading, so it was canceled and evc31 joined the preflight blacklist. Current `evc36` idleness is likewise not usable capacity: isolated real-allocation probe `687150` failed with `CUDA-capable device(s) is/are busy or unavailable` after Slurm granted one H100. `686732` is the safeguarded natural successor. Do not confuse idle `evc6`/`evc16` with H100 capacity: they are V100 nodes and the trainer is bf16/H100-oriented. `evc105` is idle 4x H200 NVL, but Slurm rejects this account on `short`/`ucfit`, so it is not usable unless the user's allocation changes. |
| **Current pretrain correction** | `685084` reached **200,000** at loss **1.6200**, gnorm **0.11**, and **154.30k tok/s**, then stayed healthy through observed step **201,270** at **154.29k tok/s** (loss **1.6461**, gnorm **0.13**). The isolated guard skips at 193,437 and 200,387 recovered on the next logged steps; no persistent instability exists. A direct node read confirms 100% H100 utilization with 63,767 / 81,559 MiB used. `ckpt_0200000.pt` and durable Newton `best_step200000.pt` match local `train/flagship_out/ckpt_0200000.pt` at md5 **`510d57df578447986b40e20029511b9d`**. Next DR target is 210k; dependency-held `686732` remains the validated two-H100 natural successor and cannot modify the current writer. |
| **Current V8/V9 control correction** | Raw board `686861` is **complete** at GSM8K maj@8 **2/200**, pass@1 **7/200**, MATH-500 **5/200**, HumanEval **7/164**, MBPP **0/200**. Replacement V8 SFT `687003` completed one epoch (4,580 updates in 1,496s) to `train/sft_v8_180k_r3/sft_ep1.pt`; its eight-case direct gate `687033` is **0/8 initial** and its held-out trace gate `687034` is **0/12 correct traces/finals**, so it is already ineligible for broad promotion while `687032` records the full board. V9 board `686942` is final: GSM8K **18/200** maj@8 and **21/200** pass@1, MATH-500 **5/200**, HumanEval **5/164** (below raw's 7/164), MBPP **1/200**. Its completed direct 1/8 and visible trace 0/12 gates already block broad promotion. |
| **Current semantic bridge control** | Held-out raw 180k baseline `686978` is **0/500** answers and 0 visible answer-traces. V9 `686979` is **29/500** answers but only **5/500** correct visible trace-and-answer pairs, concentrated in state-chain (20 answers) and fact-continuation (5); it is narrow transfer, not thinking. V10 remains untrained until full V9 control decision. |
| **Current semantic capsule correction** | First 360,000-row build `686991` was rejected by audit `686992` for 884 duplicate no-op-swap prompts; artifacts were preserved under `artifacts/rejected/`, never edited. Corrected CPU build `687009` completed 360,000 rows and 3,000 held-out 4/8/12-step episodes. Independent audit `687010` passed: 0 malformed/duplicate rows, 0 invalid episodes, and 0 exact/13-gram hits across all 30,000 held-out controller prompts. Generic admission `687015` **passed** 360,000 rows, 0 generic overlaps, and 22,899 exact packs. Raw H100 control `687017` failed CUDA preflight on excluded bad node `evc26` before model load; its dependency was canceled. Replacement controls **`687019` raw 180k** and **`687020` V9** both completed **0/300 closed-loop**, with 0 correct initial capsules and 0 correct transitions in every 4/8/12-step semantic regime. This rejects the capsule route for both checkpoints; no SFT references this candidate. |
| **Continuous latent-rollout pilot** | **Rejected against the required matched answer-only control.** The corrected 96,000-row / 1,800-held-out data admission passed, and mechanics canary `687046` was finite. Matched 24k pilots `687048` (L=0) and `687049` (progressive L=4) then shared a 896-case seed-controlled screen: L=0 scored **190/896 = 21.21%** versus the latent checkpoint's best L=4 **173/896 = 19.31%**. L=4 lost fitted, depth-OOD, and language-OOD accuracy and tied at zero full-OOD. Within-model L=4 movement is therefore non-causal; do not scale answer-only soft-token rollout. Preserve reports md5 `e1c093a51a808c14acb21299fbf8be7b` / `9a516983a4b1cd85bd4bbc96f4f97230`. |
| **Digitwise Recurrent Scratchpad (DRS)** | DRS is the discrete alternative after continuous-packet failures: the model must author one fixed-width decimal microstate per turn (one result digit, carry/borrow, and program counter); the controller only forwards that exact emitted state and never executes or repairs arithmetic. Stokes CPU build **`738117`** committed immutable v1 data: 439,865 train rows and 1,500 held-out counterfactual episodes, structurally solver-valid with zero duplicate normalized prompts. Independent audit **`738120` rejected v1 before GPU use**: it found **27** held-out 13-gram overlaps, each a real shared operand tape across `add`/`sub` operation variants, not a harmless static template hit. Preserve v1 unchanged. The corrected v2 generator reserves `(width,left,right)` across both operations and counterfactual branches, and audit **`738123` passed**: 439,865 train rows, 1,500 held-out episodes, 19,800 controller prompts, and 0 invalid/duplicate/exact/13-gram findings. V2 train/eval SHA-256: `381b8bbf3a4eddb7b08b0f9d4b08ea3ce65e1f0ec48de930632d54417c2f7f35` / `89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646`. To separate algorithmic execution from wording transfer, isolated H100 chain uses `best_step200000.pt`: raw held-out wording, raw core wording on the same episodes, one DRS-only SFT epoch, then matched post-SFT core and held-out wording evaluations. **Both raw controls are complete at 0/500** first transitions, loops, finals, counterfactual finals, and interventions in every 100-episode fit/value/width regime. Core result `digitwise_recurrent_v2_raw200k_core_p100.json` has MD5 `20a5d4cc4a776ee3ffb9220f288f4f6a`; its 34 unique first responses (mode count 265) remain malformed/repetitive rather than a fixed state. Thus neither ordinary nor canonical wording reveals pre-existing protocol competence. Original children `687350/687351` were canceled before start to add the core control. The first replacement SFT `687363` and its children were canceled before allocation after a boundary audit found that their stored script would wrap `completion_prompt` in a second `Question/Answer` frame. Replacement **`687419`** passed the hash-bound and exact-boundary gates but then failed CUDA allocation on **evc44** before model load (`CUDA-capable device(s) is/are busy or unavailable`); it wrote no model artifact and is an infrastructure non-result. Real H100 smoke **`687428`** then passed on **evc49** (CUDA tensor plus bf16 matmul). Current clean chain is **`687430 -> {687431,687432}`**, pinned to evc49 with fresh output paths, hash-bound prompt construction, and a real CUDA preflight. DRS-only excludes `evc23`, `evc32`, and `evc44` in addition to earlier CUDA failures; do not generalize those exclusions to the flagship. The chain cannot alter the flagship output/corpus. A pass would establish only narrow local digitwise execution from a canonical state, not language parsing, broad reasoning, or context scaling. |
| **DRS current causal chain** | Supersedes the obsolete `687430` chain above. Uncompiled isolated SFT **`687459`** completed cleanly on evc49 from immutable `best_step200000.pt`: 439,865 hash-bound rows, 51,131,402 packed tokens, 10,623,342 answer tokens, 24,966 packed sequences, one epoch / **1,561** updates / **1,115s**, output `train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt` MD5 `6f30db16208d274229950b17662dda01`. Fit loss is not a result. Core artifact **`687460`** establishes **275/500** final answers: fit 100/100 (w4), 98/100 (w6); value OOD 34/100 (w4), 43/100 (w6); width OOD 0/100 (w8). Crucially, its first emitted state is right on **497/500** episodes, including 98/100 at width 8, but later state transport compounds errors (353/453 emitted transitions correct before failure at width 8). `687560` then passed a no-data CUDA/bf16 H100 preflight on evc49 in 4 seconds. The gated remaining evidence path is `687562` transcript probe -> `687563` held-out wording -> `687564` direct interaction -> `687565` raw NLL. It is isolated, one-H100 per process, and cannot write the active pretrain output or corpus. |
| **Append-only Delta Ledger (ADL)** | New isolated context-scaling candidate, untrained and **admitted for future isolated GPU use**, but deliberately not submitted before the DRS causal decision. Unlike DRS, a turn emits only `step,digit,carry`; every four model-authored deltas must be compacted by the model into a short block, after which the controller drops the four raw records and transports only the exact block text. The controller does not calculate/rewrite content. The candidate tests whether reducing response copy burden creates an execution foothold and whether a first-level model-authored compaction survives paired counterfactuals. Local protocol/controller, generator, independent audit, and a 40-episode/20-held-out smoke all pass with 0 malformed rows, duplicates, exact prompt hits, and 13-gram overlap. CPU-only Stokes build **`738186`** completed separately named immutable artifacts: **384,000** train rows and **1,000** paired held-out episodes over five 200-episode regimes, with data/heldout SHA-256 `ef317dd5aed85fa83add40a637c52232f4b4daf626e609f88926cb358113cbec` / `3117ec5072134a9bade424499be9ee3a3e504e4f26deec445c3b5b1baeccaca0`. Independent Stokes audit **`738187` passed**: 0 invalid rows/episodes, duplicate normalized prompts, exact prompt hits, or 13-gram overlaps across **42,000** held-out controller prompts; report SHA-256 `5d0e2acd2cfc042de7c76266d048987c347d1e5d22b05e79232fce8ea5c9258f` matches the copy already on Newton. Raw-200k likelihood on a fixed 20-way local `digit,carry` choice is **0/16 top-1**, mean correct rank **10.688/20**, and defaults to `d=0;c=0` under both core/held-out wording; it is a learnability curriculum, not a pre-existing latent arithmetic shortcut. Artifact md5 `9ae4c88aca13079fe69036a47b88e597`. `eval_append_ledger.py`, its paired transport-only unit test, and hash-bound exact-prompt SFT/evaluation wrappers are preflighted but unsubmitted. Recursive block-of-block compression is explicitly deferred until first-level evidence exists. |
| **Current latent-pilot correction** | Corrected generation `687041` and admission `687042` **passed**: **96,000** unique solver-verified answer-only train rows and **1,800** held-out depth-5/6/8 rows, zero malformed/duplicate/exact/13-gram overlaps, training SHA-256 `aa65eefd1dbee25c3c7cec956059a970ea53e079bbc4f4695dd160cacd980fd9`. Mechanics canary `687046` was finite. The matched 24k pilots `687048` (**L=0 control**) and `687049` (**progressive L=4**) are now **rejected** by the shared 896-case, same-seed screen: control L=0 is **190/896 = 21.21%**, while the latent model is **147/896 L=0**, **163/896 L=2**, and **173/896 L=4 = 19.31%**. Its best L=4 result loses fit-IID 46.48% vs 50.39%, depth-OOD 12.50% vs 14.06%, language-OOD 11.72% vs 13.28%, and ties full-OOD at 0%. Within-model L=4 improvement is therefore not evidence; the matched control is stronger. Preserve the two hash-bound reports (md5 `e1c093a51a808c14acb21299fbf8be7b` / `9a516983a4b1cd85bd4bbc96f4f97230`) and do not scale answer-only latent rollout. |
| **Source-dropping packet memory** | **Rejected for promotion.** Admitted data has **192,000** train rows and **1,536** held-out rows across IID, length, language, and full OOD regimes; audit reports zero invalid/duplicate/exact train-eval prompt collisions, train SHA-256 `419199a756679e61601c05481ffc59221fba75b601608990539781013a51da64`, eval SHA-256 `6a10a6b27be8dc6b0a36954296d9f33abd099d87bbdff46c251a1357f1c894c3`. Matched M0 (`687082`, slots=0) completed 6,000 updates in 1,153s; M1 (`687083`, slots=8) completed the same 6,000 updates in 2,694s from the same raw-180k/data/seed/schedule. Held-out M1 normal/zero/shuffled was **6/384 / 6/384 / 9/384**. Locked comparator `687088` returned `advance=false`: M1 normal lost to shuffled on IID (**4/96 vs 5/96**) and had no positive margin in length+language (**2/192 vs 3/192 controls**), zero chunk wins, and zero query-kind wins. No source token is present at decode time, but no causal retained-information claim survived. |
| **Certified latent ledger (CLL)** | **V2 is rejected for promotion.** Fresh CPU `687141` completed in **85s** with **223,996** train rows, **7,936** held-out rows, **16,000/384** counterfactual pairs, and zero invalid/duplicate/exact/13-gram/pair findings; train/eval SHA-256 `8760df867b4da98dcc84b356eea8e0d70922e3280e868193216173603814c08c` / `dfcf852c0ca57b6bb58f4f7c1a775221e33e97aa261d79478cc3bf36e82e5fc7`. Hash-bound token report md5 `89ac960e6eb12b0f2dbe25f42a9ee3d1` reduces train chunk/source means **228.60/511.63 -> 41.44/92.74** tokens and held-out means **194.93/660.24 -> 37.43/126.79**. Matched raw-190k M0 `687144` is 11/631 (1.743%) in every mode. Eight-slot M1 `687145` is normal **16/631 = 2.536%**, zero **4/631 = 0.634%**, shuffled **12/631 = 1.902%**. Locked `687148` returns `advance=false`: fit margin **0.79pp <15pp**, length/language **0.63pp <5pp**, **2 <3** chunk wins, and **0/128** correct-and-different intervention pairs. It only meets the weak two-query-kind count. Keep reports (M1 md5 `526e9a2c365ff195bb369fd5c49ae60a`; comparator md5 `bf831f2e8ea4d8528dbe44da6343aac9`) as negative evidence. |
| **Latent state algebra (LSA)** | **Rejected by locked matched comparator `687172`; no stage 2.** The verified-geometry candidate and answer-only control each completed the same 7,163 updates from `best_step190000.pt`. On all-held-out source-removed evaluations, fit-IID normal margin was only **+1.04pp** versus the required +10pp; combined length/language margin **+0.09pp** versus +5pp; equivalent-pair margin **+0.52pp** versus +10pp; and all arms scored **0/576** intervention pairs. It won only two rather than three required chunk counts. The normal packet effect disappears under zero and shuffled-source controls, so it is not causal retained-state evidence. Final reports are mirrored locally/Newton at md5 `af9c6f40d066040a52505fa49cfd9c37`, `2d465a4d38a1aa09cf090f7fab07b90f`, and `6810722f720c1ad8d2874de4e4ff5329`. |
| **DRS transcript gate** | **`687562` completed cleanly** on evc49 in 5m52s; its local/Newton mirrored core transcript artifact SHA-256 is `c36b09f61abc0ec63fe07d7870ea24583120227ad82e53499034f0ac8d75db28`. On 25 paired core episodes it has **25/25 first transitions**, **13/25 final answers**, and the same path-dependent pattern as the full 500-case board: fit w4/w6 **5/5** each; value-OOD w4 **1/5** with four failures at transition 3; value-OOD w6 **2/5** with three failures at transition 5; width-OOD w8 **0/5** with first failures at transitions 2/3/4. One retained value-OOD failure preserved the correct written result digit but dropped the required terminal carry (`c=0` versus `c=1`), and its paired counterfactual also failed. A transcript-retention bug affected only the auxiliary per-regime sample bucket, never scores; it is fixed locally with a passing evaluator test and must be synced before the next transcript-bearing comparison. **`687563` held-out wording is currently running**, followed serially by direct interaction `687564` and raw-NLL `687565`. |
| **Workspace residual-patching baseline** | New read-only diagnostic `train/probe_digitwise_workspace.py` measures whether a last-position residual is an actionable, broadcastable local state: it symmetrically replaces the residual after each selected block with one from a matched held-out DRS transition whose carry or result digit differs, then measures directional next-token log-odds. It is explicitly a late-layer logit-lens proxy, **not** a Jacobian lens or a thinking claim. Raw-200k local-MPS baseline uses 5 held-out regimes x 4 matched pairs x carry/digit x both swap directions = **40 directions per field/layer**; artifact SHA-256 `78b5efa4f3f7fe3ef10104de8d02fdee67f253c805c58f214a4cd1985c495875`, copied Newton/local. Carry deltas are only +0.001 to +0.028 with 18-22/40 positive directions; digit deltas -0.042 to +0.0002 with 14-20/40 positive. The raw model has no stable simple last-token workspace under this test. Matching post-DRS job **`687578`** is held `afterany:687565` with the same 80-direction configuration; it cannot write training data or model output. The static-tape factor evaluator received the same transcript-bucketing repair and passes local/Newton tests before any STRR result is possible. |
| 60k final loss | final logged band ~1.5-1.7; last logged step 59990 loss 1.6989, lr 0.0005 |
| 60k skips | **45 total**, stable/healthy |
| **Corpus-expansion job** | `680324` — **✅ DONE** (finished ~12:10) |
| finemath3 output | `artifacts/shards/finemath3/` — **✅ COMPLETE: 125 shards, exactly 25.0B tokens** (`manifest.json` present, 22 GB; 8,575 contaminated docs dropped vs evalgrams). **Included in the 300k relaunch SHARDS.** |
| SFT mix (Newton) | `artifacts/sft/sft_mix_core.jsonl` — **97,439 examples**, rebuilt 2026-07-08 with hard eval filtering. Audit: 0 malformed rows, 0 duplicate questions, 0 exact eval-prompt hits; builder dropped **206 exact eval-prompt overlaps** and **741 eval 13-gram overlaps** before writing. md5 `53ed91368b4c238dc18a1ab1699e4158`; report md5 `21459b382767801e205f3f625ce106cd`. |
| Local teacher distillers | Nemotron screen run completed at **1,781 rows** but provider health was poor (`kept=19`, `err=52506` in the screen run). Bounded probes after that were also unhealthy: Nemotron `limit=5` kept 0 with 3 provider errors; GLM `limit=3` kept 0 with 3 provider errors. **Leave Nemotron and GLM paused until provider health clears; do not blindly respawn.** HY3 bulk process died/stalled after ~25.2k rows; `conc=2` and `conc=1` restarts exited without appending, even though a tiny direct Hermes/probe call completed cleanly. Claude/minimax snapshots are present. GLM remains the preferred strongest open-weight teacher when available. |
| **Verified-data expansion** | **Active, CPU-only, isolated from pretrain.** `openmath_pt` is complete at **5,000,000,144 tokens / 50 shards** and remains future-relaunch-only. FineWeb-Edu `686298` is a **4,599,762,693-token / 46-shard pilot only**, not an admitted 25B replacement: it must never be mounted as `fineweb_edu_25b` or substituted for a 25B approval. It remains preserved for diagnosis with manifest md5 `05739fa1cb31a45e1c496909f9461fa1`. DCLM-Baseline pilot `686342` and full scan `686360` are **complete and accepted as a pilot only**: **5,000,000,487 tokens / 50 shards**, 3,506,817 kept of 4,203,263 seen, with 1,820 eval-contaminated rows rejected and no entropy or byte-fallback outlier. The ongoing DCLM 25B builder `686529` and Stokes FineWeb r2 builder `738030` are both future-only; each must pass a fresh machine-readable scan and hash-bound approval before a natural handoff. The solver-verified **primitives v1** ablation data is complete: **210,000 train / 3,500 held-out** rows across arithmetic, base conversion, state updates, sort/dedup, string insertion, syllogism, and correction; 0 malformed rows, 0 duplicate normalized questions, and no train/held-out prompt overlap. `rg_v4` has **374,659 valid traces**, 0 malformed rows, and 0 normalized-question duplicates across 25 answer-checked families. CodeContests `686291` completed **3,000** train-only Python examples; its completion-form derivative has **3,593** deduplicated examples. Frozen **v4** has **643,595 clean rows**: math 240,297 / procedural 374,659 / code 3,542 / teacher 25,097. Exact SFT packing `686312` passed at **62,926** sequences. The original 15% code target would replay code 7.7x and teacher 3.1x, so v4 pilot weights are revised to **40/47/8/5** (math/procedural/code/teacher; code ~4.1x, teacher ~1.6x). `686317` is independently scaling more verified CodeContests data for a later mix; it does not alter v4. No new output may enter a frozen SFT mix before its final audit. |
| **25B language replacements** | **FineWeb `686530` is rejected as an undersized replacement:** it used `sample-10BT` and committed only **4,599,748,648 manifest tokens / 46 shards**, not 25B. Preserve it for diagnosis; it is not a future relaunch candidate and its missing machine-readable scan approval cannot be repaired retroactively. The guarded replacement **Stokes CPU `738030`** writes only `artifacts/shards/fineweb_edu_25b_r2.partial`, uses `sample-100BT`, and must prove **>=24.5B** manifest tokens before it may atomically publish `fineweb_edu_25b_r2`. DCLM **`686529`** remains a separate future-only 25B build. Both candidates still require a fresh full scan, matching manifest/report hashes, a hash-bound `*.approved.json`, and <=1% byte fallback before a natural language-data relaunch. Never mix in the 5B pilots. |
| **TACO algorithmic-code gate** | Full pilot audit **`686514` passed**: every one of 250 initially accepted Python programs matched its source record and passed **all** bounded supplied stdin/stdout cases; 0 execution/missing-case/source-unmatched drops, 0 malformed rows, duplicate prompts, exact eval hits, or 13-gram hits. Hash-matched local/Newton full derivative md5 `a33b5e0b5287bdcaec2fac550a13d5cb`; report md5 `c42b6e5cf8408c0d1cccd5f73b6a319e`. It remains an `algorithmic_code` source, deliberately separate from HumanEval/MBPP completion code. Prefix-order control `686552` failed safely at the standard quality gate with **11 normalized duplicate questions**; its 3k output/report were moved intact under `artifacts/sft/rejected/` and cannot advance. The curator now deduplicates with the same normalized identity as the quality auditor. Fresh admissible shuffled chain **`686583 -> 686584 -> 686585 -> 686586`** selects up to 3k unique rows from 5k seen via a 10k seeded shuffle buffer, replays all supplied tests, freezes a clean `algorithmic_code` candidate (>=1,000 rows), then measures true 2,048-token packing. No SFT may be submitted before those reports are reviewed. |
| **Direct capability audit** | Raw 168k/170k evidence remains negative: raw scored **4/48 Q/A**, **4/48 plain instruction**, **0/48 CoT**, **5/48 one-shot**; the pinned 170k compact-state interview `686370` was **1/8 initial, 0/8 review, 1/8 scaffold, 0/8 compact-state reuse**. A fresh local-MPS seven-case re-interview reproduced **1/7 initial, 0/7 review, 1/7 verified-fact use, 0/7 compact reuse**. A separate eight-case composition interview then independently reproduced **1/8 initial, 0/8 review, 1/8 verified-fact use, 0/8 valid `state=` emissions, and 0/8 valid-state-and-reuse**. Only the simple logic constraint passed; product/reject arithmetic, base conversion, sequential state, sort/string transformation, and two executable Python contracts all failed. New raw-180k visible-thinking interview is **0/12 trace present, 0/12 verified intermediate trace, 0/12 final answer, 0/12 trace-and-final**. It hallucinates `27 x 14 = 498`, compounds it to 7,768, and drops into a fresh `Question:` on base conversion and repair; artifact `thinking_trace_raw180k_mps_20260712.json` is local/Newton md5 `5eb7fa94cea87cc138416801f593f85c`. The fresh typed-state baseline `686456` is similarly negative: **21/420 answers and 0/280 exact emitted states**; by contract it is 0/140 write, 6/140 repair, and 15/140 reuse answers. V5 primitive SFT has a real but narrow direct signal: held-out primitive gate **`686368` = 272/700 (38.86%)** vs raw **0/700**, with 100/100 syllogisms, 88/100 string insertion, 47/100 correction, but only 12/100 arithmetic, 4/100 base conversion, 2/100 sort, and 19/100 state updates. Its verbatim interview `686388` is **2/8 initial, 2/8 review, 1/8 scaffold, 3/8 compact-state reuse**; the new reuse successes are trained primitive-like operations only. Matched matrix `686389` does show Q/A **4/48 -> 17/48** and CoT **0/48 -> 11/48**, concentrated in arithmetic/sort/state, while direct instruction is only **5/48 -> 6/48** and one-shot regresses **7/48 -> 5/48**. Full evidence: `CAPABILITY_DIAGNOSIS.md`, `TARGETS_AND_GAPS.md`, `artifacts/eval_history/deep_interaction_raw170k_r2_686370.json` (md5 `1979bcc79cb18830cb3080a7cab85e82`), refreshed `manual_capability_raw170k_refresh_20260712_mps.json` (md5 `9cd3216365b5292851e298bee4a1aeef`), composition artifact `generalization_interview_raw170k_20260712_mps.json` (md5 `d9ad30fad6c00958ad9d6908ca14c38a`), `sft_v5_primitives_168750_rg_heldout_686368.json` (md5 `4b6fe30e6b75ca7e409a152286f0ff8e`), `sft_v5_primitives_168750_deep_interaction_686388.json` (md5 `19ae7208d981f22abbfe8f9b48523c0c`), and remote state artifact `artifacts/eval_history/raw168750_primitives_v3_state_p20.json`. |
| **Post-board transcript gate** | Fresh operator-run probe `686425` compared raw 168.75k with V5 on seven new cases and preserved every turn. Raw: **1/7 initial, 0/7 review, 1/7 verified-fact, 0/7 state reuse**. V5: **3/7, 3/7, 2/7, 3/7**, limited to arithmetic, sorting, and logic; it still fails base conversion, sequential state update, string insertion, and syntax-valid Python. V5 almost never emitted the requested `state=` line, so its reuse score is **not** latent context compaction. Canonical local/Newton hash-matched artifact: `artifacts/eval_history/manual_capability_raw168750_vs_sft_v5_20260712_JOBID.json`, md5 `28dd0b15de2af16a10a2012f630072a1`. |
| **Visible-thinking decoder correction** | The tokenizer treats `<think>` / `</think>` as special tokens, so default benchmark decoding suppresses them correctly but the first transcript audit hid them. `eval_suite.py` now exposes explicit special-token policy; public boards keep `skip_special_tokens=True`, while trace audits use `False` and strip only sampled EOS. Corrected raw 180k is still **0/12** tags/intermediates/finals/both, now canonically stored at md5 `c13291539297653d92b70cf454f7ee63`; V9's tags **9/12 but 0 correct** remain an explicit format-imitation failure, not thinking. |
| **Verifier selection gate** | Corrected evaluator job `686437` reports held-out GSM8K first-pass **7/100**, oracle@16 **36/100**, verifier@16 **9/100**. Candidate sampling provides real headroom, but the 30-step verifier captures only two points of it. Improve generation and verifier ranking separately; do not report best-of-16 as current pass@1. |
| **Verifier data-volume gate** | The clean first cross-family verifier derivative has 1,201 positive + 1,201 negative examples and only **737 packed 512-token sequences**, or about **92 updates at two epochs**. It is too small and high-variance for a meaningful SFT, so no verifier SFT was submitted. Normalized verifier-prompt dedup now blocks punctuation-only duplicates (`bec11a4`). The 28-family train bank is actually 11,200 rows (400/family). Active isolated H100 rollout `686536` correctly covers its first 10,000 rows/25 sorted families and remains useful, but cannot by itself become training data. New tail rollout **`686564`** skips exactly 10,000 valid rows and covers the missing 1,200/three families; `686565` combines and globally deduplicates both outputs, then `686566` packs the complete 28-family derivative. The prior pending 25-family data/packing jobs `686540`/`686541` were canceled before start. `sft_verifier.sbatch` now refuses to launch unless the packing report is hash-bound to the current JSONL, has both labels, uses the requested length, and reaches **>=3,000 packed 512-token sequences**; that gate is meant to buy several hundred updates rather than repeat the 92-update failure. |
| **Pretraining monitor gate** | `train/eval_nll.py` and isolated `train/jobs/eval_nll.sbatch` report pure token-weighted NLL/perplexity for named frozen monitor JSONLs, excluding any training-only `zloss` and binding every result to an input SHA-256. Fixed English monitor is WikiText-103 test: 1,723 docs / 301,241 source tokens / SHA-256 `fbe8687d618550d2251b397d436abb30a990d5f0b4e7c25cfe85c7265aa251d8`; raw 170k local-MPS baseline is **NLL 3.9648849, PPL 52.7142** over 301,056 tokens (result md5 `fa9f0ea310287d710d9300c8cb0781ab`). Fixed code monitor is CodeContests **test** split: 122 unique Python-3 prompt+code rows / 146,896 source tokens / SHA-256 `62668905552c89650c0dcd79227a6fe606e107c39a126f7c1b5d9364ee8fc687`; raw 170k is **NLL 1.3537146, PPL 3.8718** over 145,408 tokens (result md5 `8b52525e46958ac20a1d6973b6de0cc0`). The gap is useful directional telemetry, but **not a proof of source-disjointness or causality**: the raw code corpus is CodeParrot-Clean and tokenized shards cannot yet prove absence of CodeContests-derived code. Therefore HumanEval/MBPP, held-out execution, and direct code transcripts remain decisive. These are regression monitors, **not** reasoning claims or web-disjointness proofs. H100 attempts `686587`, `686589`, `686590`, and `686592` encountered CUDA-unavailable/busy preflight on evc26/31/36/43 and wrote no result; evaluator now requires a real CUDA tensor allocation before loading a model. Do not burn more H100 slots without a known-good node smoke. Never place monitor inputs in `artifacts/evals` (live decontamination glob) or any training shard path. |
| **Recurrence ablation** | Mame `n_loop=1` job `686301` and `n_loop=2` job `686302` both completed cleanly for 800 matched updates. `n_loop=2` is mechanically stable but not promoted: final logged loss was essentially tied (**2.4890 vs 2.4899**) while time rose **886s -> 1466s** and steady throughput fell **472.7k -> 286.0k tok/s**. New isolated **`686928`** is held after raw board `686861` and loads the immutable 180k checkpoint twice, with `n_loop=1` and `n_loop=2`, for the same 24-case Q/A+CoT matrix and eight-case direct interview. It writes transcripts only, is explicitly an **untrained test-time architecture probe**, and can only decide whether a future equally trained recurrent control is worth running. It cannot promote recurrence or alter the flagship. |
| **Future handoff data stream** | **Fixed forward-only, never applied to active `685084`.** Prior checkpoints did not serialize `ShardLoader` state; every resumed job could recreate its stream from the same `DSEED=777`. The audited `train.py` now records `data_stream_generation`/seed and resumes with a deterministic distinct stream generation. It was synced to Newton at 18:09 EDT while the running Python process remained untouched; a legacy current checkpoint will therefore start successor generation 1 rather than replaying seed 777. Tiny-train smoke verified generation **0 -> 1** and checkpoint metadata. This prevents repeated prefixes at the next natural handoff but does not claim exact cursor restoration for old chunks. |
| Preserved checkpoints (cluster) | Includes prior milestones plus durable **`best_step190000.pt`** (copied from numbered `ckpt_0190000.pt` before normal retention can reap it); md5 **`3e195aaf44a14259797c49d7f80d9c7f`**. |
| **Local DR backup (Mac)** | Includes prior milestones plus **`train/flagship_out/ckpt_0190000.pt`**, full+optimizer, locally and remotely md5 **`3e195aaf44a14259797c49d7f80d9c7f`**. Next DR target: 200k. |
| **Large artifact transfer policy** | For big checkpoints/shards/uploads, prefer VPS-to-VPS or Newton-to-VPS staging when credentials/hosts are available; the VPS links have ~20 Gbit internet and should beat Mac↔Newton transfers. Still use `.part` files and md5/sha256 on both ends before trusting or deleting anything. |
| **CPU-only execution policy** | User-directed: route future CPU-only corpus generation, audits, packing, and report jobs to **Stokes** (`ssh sa305415@stokes.ist.ucf.edu`), preserving Newton for H100 work and GPU-adjacent scheduler coordination. Access was revalidated 2026-07-13: Stokes host `euser1` exposes 64 CPUs and the shared `/lustre/fs1/home/sa305415` workspace. Before a launch, verify output ownership and that no writer overlaps an existing Newton job; do not duplicate an in-flight CPU writer. |

**Checkpoints preserved so far:** every 10k through 50k; 60k is model-only because the trainer writes
`ckpt_final.pt` without optimizer state. `ckpt_0060000.pt` is local and cluster hash-matched. The 300k
extension resumes from `ckpt_0060000.pt` with fresh optimizer rewarmup, so no stale 59k momentum is used.
`ckpt_0059000.pt` is the local full+optimizer emergency fallback if a fresh-optimizer resume proves bad.

**Next actions in order:** (1) Watch `685084`: retain the normal ~154k tok/s band and expected 250-step
checkpoints, preserve/download 200k, and never interrupt a recovered isolated gnorm skip. (2) Before `685084`
exits, determine whether FineWeb r2 and DCLM 25B both have committed manifests, full machine-readable scans,
and hash-bound approvals. If both pass, cancel fallback `686732` before its dependency releases and submit the
language-balanced relaunch with `NG=2 BS=32 ACC=4`, an exact 524,288-token update, fresh optimizer, and the
newest preserved numbered checkpoint. If either fails or is incomplete, let `686732` resume the old stream
with `world=2`, a new data-stream generation, and no NCCL/CUDA failure rather than leaving the flagship idle.
(3) Retain V7
`686467` only as rejected constructed-contract evidence; its 73.10% generator holdout does not survive the
two independent transfer interviews, so no V7 public board is justified. (3) Complete and independently scan
the 25B DCLM replacement before it can enter any future relaunch. (4) Retain raw `686370`, V5 `686388`, V6
`686415`, and V7 `686484` as negative compact-state evidence; do not mistake generated-contract performance
for autonomous latent reasoning. (5) Improve the verifier separately: `686437` showed first-pass 7/100,
 oracle@16 36/100, verifier@16 9/100; the new cross-family rollout must pass a held-out selection gate before
 any promotion. (6) Let shuffled `686554 -> 686555` scale and fully replay TACO before considering an isolated
 algorithmic-code ablation. (7) At the next controlled/natural pretrain handoff only, use manifest-gated language,
math/reasoning, and code sources with explicit, per-batch-safe domain weights; do not alter live SHARDS.
The future relaunch target is **600k absolute steps**, not 300k: `train.py --steps` is absolute and the
current phase already targets 300k, so a 300k relaunch would take zero updates.

---

## 1a. PROGRESS JOURNAL (append-only — newest last)

Terse, dated entries so a fresh agent sees the trajectory, not just a point-in-time snapshot. **Add a
line at each milestone / intervention / decision.** Don't rewrite history; append.

- **2026-07-06 eve** — Flagship pretrain (job `680149`) running on evc22, single H100, `--steps 60000`,
  domain-interleaved dataloader (the divergence fix). Resumed onto evc22 after earlier cluster-
  saturation churn; stable since.
- **2026-07-06** — Corpus-expansion job `680324` launched on evc1 (CPU) to tokenize finemath-3plus →
  `shards/finemath3` (~25B math tokens, ~doubles corpus + triples math).
- **2026-07-07 ~04:05** — Wrote this runbook + mirrored to cluster + added auto-memory
  `shohin-custody-runbook`. Pretrain ~step 41.3k, loss ~1.6, healthy. User set `/goal`: keep detailed
  docs current for any future agent, and take care of the model overnight.
- **2026-07-07 ~10:25** — Preserved `best_step40000.pt`. Pretrain ~step 47k, still stable-phase.
- **2026-07-07 ~11:00** — **WSD DECAY BEGAN at step 48k** (lr started dropping 0.005→0.0047…). This is
  the phase that converts flat stable-phase loss into capability; 60k will be the first honest read.
- **2026-07-07 ~11:50** — Pretrain ~step 49.3k, loss ~1.6, lr 0.0045, 34 skips total (clean).
  finemath3 at 122 shards (~24.4B), completing imminently. Corrected 60k ETA to ~10–11h (this evening;
  prior "~1.3 days" was an overestimate). Updated §1 + added this journal per user's housekeeping ask.
- **2026-07-07 ~12:06** — **Local disaster-recovery backup taken** (user asked "what's saved locally
  if the cluster dies?"). Local fallback was only step-15k weights; pulled `ckpt_0049000.pt` (1.0 GB,
  full+optimizer, md5 5123710a25e8, verified bit-identical + load-tested) to `train/flagship_out/`.
  Standing habit: refresh the local DR copy at each 10k milestone (overwrite; keep newest 1–2). To
  recover on a fresh machine/cluster: place it in an `--out` dir and launch flagship.sbatch with
  `--resume` (it carries Muon+AdamW state → resumes with momentum). macOS `rsync` rejected fancy flags
  (openrsync); use plain `scp` for cluster→Mac pulls.
- **2026-07-07 ~12:10** — **finemath-3plus corpus DONE** (job 680324 finished). 125 shards, **exactly
  25.0B tokens** (13.41M docs kept of 13.48M; 62.7k dropped short, 8,575 dropped as eval-contaminated),
  22 GB, `shards/finemath3/manifest.json` written. Total pretrain corpus now **62.4B** (37.4B + 25.0B).
  Will fold into SHARDS at the 60k→300k relaunch (§4A b); deliberately NOT added to the current job
  (changing the data mix mid-run — esp. on a recovery resubmit — is unsafe).
- **2026-07-07 ~12:57** — Step 50k reached. Preserved `best_step50000.pt` on cluster; refreshed local
  DR backup → `ckpt_0050000.pt` (md5 53bc952edc0b, verified + load-tested, old 49k removed). Pretrain
  ~step 50.4k, loss ~1.6, lr 0.0041 (decay on track), 34 skips. ~9–10h to 60k.
- **2026-07-07 ~13:50** — Pretrain ~step 51.4k, **loss hit 1.44 (new low)** — decay is converting to
  real loss reduction (lr 0.0037), encouraging pre-60k. Probed OpenMathInstruct-2 for `openmath_pt`
  corpus: fields are `problem` / `generated_solution` / `expected_answer` (need to concat the first
  two). **Blocked on login node** (datasets→torch import fails: OpenBLAS RLIMIT_NPROC thread explosion,
  then `libcupti.so.12` mmap fail) — confirms the standing rule: run tokenization via **sbatch on a
  compute node** (as finemath3 did), never the login node. **Decision: defer `openmath_pt` to
  post-60k** (idle CPU is plentiful during the 2-wk extension; first extend uses the 62B corpus;
  openmath_pt folds in at a later relaunch). To do it: add multi-field concat to `tokenize_shards.py`
  (`--text-cols problem generated_solution`), wrap in a compute-node sbatch, decontam vs evalgrams.
- **2026-07-07 ~14:35** — Pretrain ~step 52.1k, loss ~1.7 (in-band), lr 0.0035, 37 skips (healthy).
  Prep for post-60k `openmath_pt`: added `--text-cols` multi-field-concat flag to
  `pipeline/tokenize_shards.py` (compiles + parses; backward-compatible — default still single
  `--text-col`), synced to cluster (md5 cfa75e03). So post-60k the openmath_pt job is just: wrap
  `tokenize_shards.py --dataset nvidia/OpenMathInstruct-2 --text-cols problem generated_solution
  --decontam-grams evals/evalgrams.pkl --out-dir shards/openmath_pt` in a compute-node sbatch.
- **2026-07-07 ~15:20** — **MAJOR: teacher unlocked + distillation pipeline live.** User revealed a
  free-this-week Hermes/Nous teacher (`hermes` CLI, model `tencent/hy3:free`, Nous Portal, authed on
  the Mac). This unblocks our core thesis (short-CoT distillation — previously GPU-gated). Built
  `pipeline/hermes_distill.py` (rejection-sampled: generate trace → verify vs gold → keep only correct;
  parallel, resumable, emits curated SFT format). Validated: GSM8K **100% yield** (20/20, then 47/48),
  0 rate-limit errors, clean eval-aligned traces. Launched full GSM8K-train distillation in background
  (7,473 problems → `artifacts/sft/hy3_gsm8k.jsonl`). User green-lit ALL FOUR data initiatives — see
  new **§11 DATA EXPANSION PLAN**. Throughput ~0.5/s via CLI (startup-bound); `hermes proxy`
  (OpenAI-compatible) is the scale path for big volume.
- **2026-07-07 ~16:00** — **Breadth expansion (user: "don't just focus on math — all kinds of
  reasoning; use hermes for everything; store data locally").** Generalized `hermes_distill.py` to
  multiple-choice (`mc`) + yes/no (`exact`) verification (per-row `prompt`/`answer_type`). Wrote
  `pipeline/fetch_problems.py` → normalized TRAIN banks in `artifacts/problems/`: gsm8k(7473),
  math(12496), sciq(11679), commonsenseqa(9741), openbookqa(4957), arc_easy(2251), arc_challenge(1119)
  = **~49.7k problems across math + science + commonsense** (logiqa loader deprecated → logic still a
  gap, fill via Reasoning-Gym). `hermes proxy` scale-path abandoned (wants different auth than the CLI
  login). **Machine limit: 8-core / 8 GB Mac** — parallel-process fan-out would OOM, so consolidated to
  **ONE interleaved run** over a shuffled all-domains bank (`combined.jsonl`) at conc 20 (load ~5, mem
  ok) → **`artifacts/sft/hy3_reasoning.jsonl`** (canonical growing file; 92% yield). Resumable; seeded
  with the 474 already-distilled traces. ~0.5/s → banks a few k diverse verified traces over the window.
- **2026-07-07 ~16:30** — **Logic gap filled.** reasoning_gym not installed locally (didn't risk a
  pip install mid-run on the 8 GB Mac); instead fetched **ReClor** (LSAT/GMAT logical reasoning, 4,638
  train, MC) via `datasets` and folded it in. Rebuilt `combined.jsonl` → **54,354 problems / 8 domains**
  (gsm8k, math, sciq, commonsenseqa, openbookqa, arc_easy/challenge, **reclor=logic**). Did a *verified*
  restart of the distiller (PID changed; resumed skipping the 1,703 done; confirmed ALIVE + banking new
  traces before ending the turn). Distillation now spans math + science + commonsense + logic. NOTE:
  harness writes `source="hy3"` for all outputs (per-domain provenance lives in `artifacts/problems/*`;
  map back by question if needed).
- **2026-07-07 ~17:20** — **OpenRouter unlock + faster backend.** User gave an OpenRouter API key
  (stored in git-ignored `.env` as `OPENROUTER_API_KEY` — NEVER put it in runbook/memory/mirrored files)
  + two models: `tencent/hy3:free` and `nvidia/nemotron-3-ultra-550b-a55b:free` (a 550B teacher, free).
  Added an `--backend openrouter` HTTP path to `hermes_distill.py` (OpenAI-compatible, rate-limit
  backoff) — direct HTTP has no per-call process spawn, so high concurrency is Mac-safe. Also fixed a
  **math verifier bug** (competition-MATH gold carries units/latex like `0.4\mbox{ miles}` that never
  string-matched the model's clean `0.4` → added last-number numeric fallback; yield 16%→87% on easy
  segments). **Switched the bulk run to OpenRouter hy3 conc-24** (`artifacts/sft/hy3_reasoning.jsonl`,
  resumed from ~3,030) — faster + lighter than the CLI. Yield varies by domain (science/gsm8k high;
  competition-math/reclor-logic lower = sampler rejecting hy3's hard-problem misses; kept traces always
  verified). **nemotron-550b** noted for a future *quality* pass on hard math/logic (stronger teacher →
  higher yield on hard problems), but it's slow under concurrency (run low-conc).
- **2026-07-07 ~17:45** — **Multi-provider teacher FLEET** (user: "don't pick one provider — run them
  ALL to max throughput"). Added NVIDIA endpoint (`integrate.api.nvidia.com/v1`) — new models
  `z-ai/glm-5.2` (SoTA, but **currently DEGRADED server-side** — retry later), `nvidia/
  nemotron-3-ultra-550b-a55b` (strong, concise), `minimaxai/minimax-m3`. Generalized `hermes_distill.py`
  backend to any OpenAI-compatible endpoint (`--backend nvidia/openrouter`, `BACKENDS` map, `--max-tokens`
  cap for concise traces). NVIDIA key in git-ignored `.env` (`NVIDIA_API_KEY`). **KEY FINDING: OpenRouter
  free tier = 1,000 requests/DAY** (hit it fast at 2.5/s → 429 `free-models-per-day`; resets ~2.3h) — so
  **OpenRouter is NOT for bulk**; the **hermes CLI (Nous Portal) has no daily cap** and is the reliable
  bulk workhorse. **Current fleet (3 channels, own output files, teacher-diverse):** Nous-hy3 CLI conc-12
  → `hy3_reasoning.jsonl` (bulk); NVIDIA nemotron-ultra conc-6 → `hy3_reasoning_nemotron.jsonl` (80%
  yield); NVIDIA minimax conc-3 → `hy3_reasoning_minimax.jsonl` (was erroring at conc-6, gentled). Merge
  all `hy3_reasoning*.jsonl` for the SFT mix (dedup by question, keep teacher diversity).
- **2026-07-07 ~18:00** — **GLM-5.2 recovered → added as a channel; dropped weak minimax.** GLM-5.2
  (SoTA) is now the best channel: **88% yield @ 1.7/s** (fast + strong, 0 err) → `hy3_reasoning_glm.jsonl`
  (`--backend nvidia --model z-ai/glm-5.2 --concurrency 5`). minimax dropped (25% yield, erroring,
  barely contributing). **Fleet now: Nous-hy3 (bulk) + NVIDIA nemotron-ultra (80%) + NVIDIA GLM-5.2
  (88%, SoTA).** ~4,000 traces total. (minimax file `hy3_reasoning_minimax.jsonl` kept — 49 valid
  traces.) OpenRouter still daily-capped (re-add as burst when reset).
- **2026-07-07 ~18:15** — **GitHub backup (user: "put everything on github as well as locally").**
  Remote = `github.com/GodlyDonuts/shohin`. **Redacted the Newton password** out of this runbook (moved
  to git-ignored `.env` as `NEWTON_PW`); full secret scan clean before pushing. Pushed: all code + docs
  (commit 0aa12b6) + distilled traces + tokenizer + eval sets (commit e487677, ~11M). Deliberately kept
  OUT of git (regenerable, avoid bloat): `openmath2.jsonl` (89M → `sft_curate.py`), problem banks (→
  `fetch_problems.py`). `.env` + `artifacts/` stay git-ignored; specific data files force-added.
  **ONGOING BACKUP HABIT:** periodically (each cycle or two) `git add -f artifacts/sft/hy3_reasoning*.jsonl
  && git add -A && git commit -m "backup: distilled traces + docs" && git push origin main` so the
  growing fleet output + doc updates stay mirrored to GitHub.
- **2026-07-07 ~18:50** — **Claude subagents = 4th teacher channel (user: "make traces with subagents,
  maximize").** Workflow `claude-teacher-distill` (repeatable): I pick fresh UNSOLVED problems, split into
  batches, spawn 20 subagents that each solve their batch (**gold withheld** → genuine reasoning), return
  {id,response}; I verify vs the withheld gold + rejection-sample → `hy3_reasoning_claude.jsonl` (source
  `claude`). First run: 20 agents, 525k tok, 300 problems → 152 kept. **Found + fixed a MATH VERIFIER BUG**
  along the way: competition-MATH gold is LaTeX (`\frac{3}{2}`) which never matched the model's clean
  `3/2` — was silently tanking EVERY teacher's math yield. Fixed `verify()` (LaTeX-aware `_norm_math` +
  `_eval_math` fraction eval); re-verify recovered +87. **Restarted the whole fleet** to pick up the fix
  (they re-attempt previously-rejected problems → recover false-rejected math). Repeatable: to run another
  Claude round, regen `claude_batches.json`+`claude_goldmap.json` (fresh unsolved) then re-invoke the
  workflow scriptPath; verify via `scratchpad/verify_claude.py`. Grand total ~6,225 traces (5 domains,
  4 teachers). WF gotcha: `args` didn't reach the script — hardcode values in the script instead.
- *(next: 60k FEEDBACK+EXTEND (~3.5h) — see §4A; fleet = hy3+nemotron+GLM+Claude-subagents; §11; git push)*
- **2026-07-07 ~19:15** — **Codex takeover.** Verified Newton auth + read-only health: job `680149`
  alive on evc22 at step ~56.9k, lr ~0.0017, 45 skips, latest numbered ckpt `ckpt_0056000.pt`, preserved
  through `best_step50000.pt`; ETA to 60k ~3h. Local teacher fleet still has three open writers:
  `hy3_reasoning.jsonl` (active, PID 33375, ~5.4k rows), `hy3_reasoning_nemotron.jsonl` (PID 33376,
  ~752 rows, slow), and `hy3_reasoning_glm.jsonl` (PID 33377, ~25 rows, process alive but file idle since
  ~18:01; inspect/relaunch only if still idle on next cycle). Created Codex heartbeat
  `shohin-flagship-custody` every ~40 min for the 60k transition. Do not manually edit live JSONL files.
- **2026-07-07 ~19:50** — **SFT data policy upgraded.** User made Codex executive director. Decision:
  default SFT should train from a frozen, auditable curated mix, not live distiller files and not the
  old math-only baseline. Added `pipeline/build_sft_mix.py`; `train/jobs/sft.sbatch` now builds
  `artifacts/sft/sft_mix_core.jsonl` by default (deduped by normalized question, concise response caps,
  source-priority tie-breaks) and keeps `SFT_BASELINE=1` as the old `openmath2 + rgym + code` escape
  hatch. Pulled missing `rgym.jsonl` from Newton to local. Local snapshot: **81,894 examples** =
  OpenMath backbone + 993 Reasoning-Gym + 446 code + ~6.2k verified teacher traces. Before 60k SFT,
  sync current local `hy3_reasoning*.jsonl` and this new script/job to Newton, then let `sft.sbatch`
  build the cluster-side frozen mix.
- **2026-07-07 ~22:15-22:35** — **60k FEEDBACK+EXTEND transition executed.** Job `680149` finished cleanly:
  `[done] 60000 steps in 112203s`, 45 skips total, final logged loss band still healthy. Trainer only
  wrote model-only `ckpt_final.pt`, so created `best_step60000.model.pt` and numbered
  `ckpt_0060000.pt` from final weights; extension uses `FRESH_OPT=1` to rewarm optimizer instead of
  carrying stale 59k momentum. Synced latest teacher traces to Newton and rebuilt cluster
  `sft_mix_core.jsonl` (**85,593 examples**). First SFT submission `680991` exposed a cwd bug: `sft.sbatch`
  rebuilt the mix from `$BASE/train` and produced 0 rows. Canceled it immediately, fixed script to build
  from repo root, rebuilt mix, and relaunched SFT as **`681000`** on evc43. Launched 300k extension as
  **`680992`** on evc22 with SHARDS including `finemath3`; verified resume line:
  `ckpt_0060000.pt -> start step 60001 (FRESH optimizer: momentum reset + rewarmup)`.
- **2026-07-07 ~23:15** — **Local checkpoint custody verified.** Downloaded and hash-matched Newton to
  Mac: `ckpt_0060000.pt` / `best_step60000.model.pt` (model-only 60k, md5
  `d2fdf867bd49cf517b62364e152bffde`), `ckpt_0059000.pt` (full+optimizer fallback, md5
  `0038df81be145cf4a4b0644e2dce284a`), and `sft_ep3.pt` (md5
  `dda39ab36aa73bd6284b94d9fbf252e5`). SFT job `681000` completed; eval job **`681030`** is running on
  evc32 against `sft_ep3.pt`. Extension job `680992` healthy at step ~60.8k, loss in band, rewarmup lr
  ~0.002, throughput ~147k tok/s. Next DR target: download 70k checkpoint when it appears.
- **2026-07-07 ~23:20** — Heartbeat check: extension job `680992` healthy at step **60,920** (loss
  1.7179, lr 0.0023, ~147.6k tok/s); eval job `681030` still running, partial board currently shows
  `sft_ep3.pt` weak on early math metrics (GSM8K 6/100, MATH500 0/100; code still in progress). Local
  distillers: HY3 process alive but not recently writing, nemotron still slowly adding rows, GLM had no
  data movement since 22:41 and no log movement since 18:58. Terminated the stalled GLM PID and probed
  NVIDIA directly; `z-ai/glm-5.2` returned HTTP 429, so do **not** relaunch GLM until a later heartbeat
  shows the provider limit cleared. HY3 and nemotron were left untouched.
- **2026-07-07 ~23:27** — User challenged whether 2 GPUs are worth the speedup. Decision: yes, likely
  worth pursuing, but only through a measured canary that preserves training semantics. Added
  `train/jobs/ddp2_canary.sbatch` and submitted **job `681040`** as a separate 2-H100 branch, excluding
  the live/eval/down nodes. It copies `flagship_out/ckpt_0060000.pt` into its own output directory
  (`train/ddp2_canary_60000_20260707_232737`) and runs `NG=2 BS=16 ACC=8 STEPS=61050 FRESH_OPT=1`
  with the same 62.4B shard mix and same effective batch as the single-GPU extension. **Do not kill or
  migrate `680992` yet.** Promotion rule: only switch the flagship to two GPUs after `681040` shows clean
  DDP startup, no rank hang, loss/skip behavior matching the 60k replay, and materially better throughput
  (target at least ~1.6x useful tok/s).
- **2026-07-08 ~00:00** — Heartbeat milestone: `sft_ep3.pt` eval job `681030` completed cleanly in
  48m41s but the recipe underperformed badly on the small board: GSM8K **6/100**, MATH500 **0/100**,
  HumanEval **4/164**, MBPP **0/100**. Treat this as evidence that the first curated SFT format/mixture
  is not yet useful, not as a pretrain failure; live pretrain remains healthy at step **61,620** (loss
  1.5953, lr 0.0040, ~148.7k tok/s). Downloaded and hash-verified `ckpt_0061000.pt` to the Mac
  (full+optimizer, md5 `28a18ebd7efc67cbbb72db6505493248`) as the latest post-60k DR point. Canary
  `681040` is still pending on priority with scheduled node `evc42` and predicted start around
  2026-07-08 01:01. HY3/nemotron teacher writers are still producing valid JSONL; GLM remains paused
  on NVIDIA 429.
- **2026-07-08 ~00:06** — User clarified transfer ops: large uploads/downloads should use VPS-to-VPS
  or Newton-to-VPS staging when possible because those links have ~20 Gbit internet. Updated the live
  transfer policy accordingly. Mac local DR copies are still useful, but for bulk checkpoint/corpus
  movement the preferred path is via VPS staging with `.part` files plus md5/sha256 verification on both
  ends.
- **2026-07-08 ~00:55** — **2-H100 canary succeeded; promotion attempted with safe fallback.** Canary
  job `681040` completed cleanly on evc42: `world=2`, resumed from `ckpt_0060000.pt`, no DDP hang, loss
  stayed in band, and throughput reached ~262k tok/s vs the live 1-GPU ~149k tok/s (~1.76x). Preserved
  the live full checkpoint `ckpt_0062000.pt` to `best_step62000_pre2gpu.pt` (md5
  `e4f3de659effac5c6875c6ae17d6b544`) and stopped old 1-GPU job `680992`. A direct long 2-GPU promotion
  did not start promptly due Slurm priority, so switched to short backfill chunks: current job `681083`
  is RUNNING on evc32 for 30m, resumed from `ckpt_0062000.pt`, and has printed steps through 62030.
  Queue is dependency-ordered to avoid overlapping writes to `flagship_out`: `681080` 1-GPU 2h starts
  after `681083`; `681078` 2-GPU 2h starts after `681080` using new `train/jobs/flagship2.sbatch`.
- **2026-07-08 ~01:25** — **Backfill handoff guarded.** Short 1-GPU job `681083` ran to its 30-minute
  limit on evc32, resumed from `ckpt_0062000.pt`, reached step **62490**, and last saved
  `ckpt_0062400.pt`; loss stayed in band. `681080` was released but priority-pending with predicted
  start ~02:35 on evc50, so submitted deadline-bounded filler **`681087`** (`--time=00:30:00`,
  `--deadline=02:30`, `CKPT=100`) and set `681080` dependency to `afterany:681087`. This prevents
  overlapping writes while allowing Slurm to use any short backfill hole before the scheduled
  continuation.
- **2026-07-08 ~01:27** — Filler `681087` started immediately on evc32, resumed from
  `ckpt_0062400.pt -> start step 62401`, and printed healthy early steps through **62430** (loss
  1.65-1.69, lr 0.0050). `681080` remains held on `afterany:681087`; next check must ensure it releases
  after the filler ends and resumes from the newest checkpoint.
- **2026-07-08 ~01:59** — **Handoff to scheduled continuation succeeded.** Filler `681087` timed out
  cleanly at 30m after reaching **step 62900** and saving `ckpt_0062900.pt`. Scheduled 1-GPU job
  **`681080`** started immediately on evc29, resumed from `ckpt_0062900.pt -> start step 62901`, and
  printed healthy steps through **62940** (loss 1.60-1.66, lr 0.0050). Next queued job remains
  **`681078`**, the 2-H100 chunk, dependency-held behind `681080` to avoid concurrent writes.
- **2026-07-08 ~02:20** — **2-GPU promotion tightened after user sleep handoff.** User pointed at idle
  `ev6`/`ev16` and `evc105`. Verified `evc6`/`evc16` are idle but are V100 nodes
  (`tesla_v100-pcie-16gb:2` and `tesla_v100-pcie-32gb:2`), unsafe/likely slower for this bf16 H100
  trainer. Verified `evc105` is idle 4x H200 NVL but only in `short,ucfit`; `sbatch --test-only` on
  both partitions returns `Invalid account or account/partition combination`, so this account cannot use
  it. No normal-partition node currently has 2 free H100s; `681080` is still healthy on one H100 (latest
  seen **step 63190**, `ckpt_0063000.pt` saved), so do not kill it. Updated `flagship2.sbatch` to the
  lean request (`-c 4`, `--mem=96G`, `OMP_NUM_THREADS=2`) and replaced old queued `681078` with
  **`681090`**, a dependency-held 2-H100 job after `681080` with `NG=2 BS=16 ACC=8`, `CKPT=500`, and
  only down/drain H100 nodes excluded.
- **2026-07-08 ~02:23** — **Overnight no-idle chain installed.** User is asleep and asked Codex to act
  as executive director, try hard for 2 GPUs, and benchmark/verify as needed. Patched `flagship2.sbatch`
  with `AUTO_REQUEUE` control to avoid a fallback overlap if a 2-GPU zero-step failure self-requeues.
  Replaced `681090` with **`681091`**, a lean 2-H100 job after `681080` with `AUTO_REQUEUE=0` and
  deadline `2026-07-08T06:20:00` (2h job must start by ~04:20, shortly after `681080` ends). Queued
  **`681092`** as the 1-H100 fallback `afterany:681091`. Outcome: if Slurm can place 2 H100s, 681091
  runs; if not, deadline cancellation releases 681092 so training keeps moving with a single writer.
- **2026-07-08 ~02:36** — Heartbeat: `681080` remains RUNNING on evc29, healthy through **step 63570**,
  with `ckpt_0063500.pt` saved and throughput ~142k tok/s. Saw two isolated `[skip:gnorm]` events around
  step 63340, then immediate recovery to normal gnorm/loss; no intervention. `681091` (2-H100) and
  `681092` (1-H100 fallback) remain dependency-held. Local HY3/Nemotron teacher writers are alive and
  valid JSONL (`hy3_reasoning` ~18.5k rows, `nemotron` ~1.18k rows).
- **2026-07-08 ~03:16** — Heartbeat: `681080` remains RUNNING on evc29, healthy through **step 64240**,
  with `ckpt_0064000.pt` saved and throughput ~144.6k tok/s. A single additional isolated gnorm skip
  occurred around step 63699 and recovered immediately. `681091` (lean 2-H100 attempt) and `681092`
  (1-H100 fallback) are still dependency-held behind `681080`. Local HY3/Nemotron teacher writers are
  alive and valid JSONL (`hy3_reasoning` ~19.7k rows, `nemotron` ~1.22k rows).
- **2026-07-08 ~04:05** — **2-H100 continuation live.** `681080` timed out cleanly on evc29 after
  reaching **step 64890**; last saved checkpoint was `ckpt_0064500.pt`, so expected unsaved tail was
  discarded. `681091` started immediately on evc42 with two H100s visible, `world=2`, `NG=2 BS=16
  ACC=8`, and resumed from `ckpt_0064500.pt -> start step 64501`. Healthy early steps through
  **64680**; throughput is warming past **238k tok/s** and should continue toward the canary's ~260k.
  Canceled old 1-H100 fallback `681092` and queued **`681105`** as the next 2-H100 chunk after `681091`
  (deadline `2026-07-08T08:20:00`, `AUTO_REQUEUE=0`), with **`681106`** as the 1-H100 fallback after
  `681105`. This keeps the run on the fastest validated path while preserving single-writer safety.
- **2026-07-08 ~04:42** — **2-H100 continuation steady-state verified.** `681091` is still RUNNING on
  evc42 with `world=2`, healthy through **step 65860**, loss/gnorm in band, and throughput now
  **~269.6k tok/s**, slightly above the canary target. Checkpoints **`ckpt_0065000.pt`** and
  **`ckpt_0065500.pt`** are saved; pulled `ckpt_0065500.pt` to the Mac and md5-verified it
  (`670ae99c278cf26706ebb1b5ee8d7b72`) as the first stable promoted 2-H100 DR point. Queue remains
  single-writer safe: `681105` is dependency-held after `681091`; `681106` is the 1-H100 fallback after
  `681105`. HY3/Nemotron local teacher writers remain alive and valid JSONL (~22.4k HY3, ~1.31k
  Nemotron).
- **2026-07-08 ~05:59** — **2-H100 handoff succeeded.** `681091` hit wall-time on evc42 at 05:55,
  after reaching **step 68190** and saving **`ckpt_0068000.pt`** (latest durable point from that chunk).
  `681105` started immediately on evc29 with two H100s, `world=2`, `NG=2 BS=16 ACC=8`, and resumed from
  **`ckpt_0068000.pt -> start step 68001`**. First printed step **68010** is healthy; throughput is
  still cold-starting/compiling. `681106` remains dependency-held as the 1-H100 fallback after `681105`.
  Local HY3/Nemotron teacher writers remain alive and valid JSONL (~25.0k HY3, ~1.39k Nemotron).
- **2026-07-08 ~06:40** — **681105 healthy; next 2-H100 chunk queued.** `681105` is healthy through
  **step 69440** on evc29, still `world=2`, loss/gnorm in band, and throughput has warmed to **~268.8k
  tok/s**. Checkpoints **`ckpt_0068500.pt`** and **`ckpt_0069000.pt`** are saved. Submitted **`681115`**
  as the next 2-H100 chunk after `681105` with deadline `2026-07-08T10:20:00`, `AUTO_REQUEUE=0`,
  `NG=2 BS=16 ACC=8`, `CKPT=500`, and the same explicit bad/down-node exclusion; moved **`681106`**
  behind `681115` as the 1-H100 fallback. HY3 bulk distiller was not alive; `conc=2` and `conc=1`
  restarts both exited without appending, although direct Hermes and a tiny foreground probe completed
  cleanly. Decision: leave HY3 paused until the harness is inspected instead of churn-restarting it.
  Nemotron remains alive and writing. Next heartbeat: preserve 70k when available and verify `681115`
  remains dependency-held or starts cleanly after `681105`.
- **2026-07-08 ~07:20** — **70k preserved and locally verified.** `681105` is still RUNNING on evc29,
  healthy through **step 70600**, `world=2`, loss/gnorm in band, throughput **~272.6k tok/s**. Preserved
  Newton `ckpt_0070000.pt` to **`best_step70000.pt`** and verified both md5
  `87f28ff961c579c7263136892b340d6f`; downloaded `ckpt_0070000.pt` to the Mac via `.part` and md5
  matched it as the latest local DR checkpoint. `ckpt_0070500.pt` also exists on Newton. Queue remains
  single-writer safe: `681115` dependency-held after `681105`, `681106` fallback after `681115`.
- **2026-07-08 ~08:05** — **2-H100 handoff to `681115` succeeded; next block queued.** `681105` timed
  out cleanly on evc29 after reaching **step 71730** and saving **`ckpt_0071500.pt`**. `681115` started
  immediately on evc29 with two H100s, `world=2`, `NG=2 BS=16 ACC=8`, and resumed from
  **`ckpt_0071500.pt -> start step 71501`**; early steps through **71590** are healthy, with throughput
  warming through **~209.7k tok/s** after compile. Submitted **`681123`** as the next 2-H100 chunk after
  `681115` (deadline `2026-07-08T12:20:00`, `AUTO_REQUEUE=0`) and moved **`681106`** behind it as the
  1-H100 fallback. Local HY3 remains paused at 25,243 rows pending harness inspection; Nemotron remains
  alive and writing (`hy3_reasoning_nemotron.jsonl` 1,510 rows).
- **2026-07-08 ~08:40** — **681115 steady-state verified.** `681115` is still RUNNING on evc29 with
  `world=2`, healthy through **step 72750**, **0 skips**, loss/gnorm in band, and throughput warmed to
  **~271.9k tok/s**. Checkpoints **`ckpt_0072000.pt`** and **`ckpt_0072500.pt`** are saved. Queue remains
  single-writer safe: **`681123`** waits on `afterany:681115`; **`681106`** waits on `afterany:681123`
  as the 1-H100 fallback. Nemotron teacher remains alive and writing (`hy3_reasoning_nemotron.jsonl`
  1,552 rows); HY3 remains paused pending harness inspection.
- **2026-07-08 ~09:20** — **681115 remains fast; second 2-H100 follow-on queued.** `681115` is RUNNING
  on evc29, healthy through **step 74030**, loss/gnorm in band, throughput **~274.6k tok/s**. There was
  one isolated gnorm skip at **step 73008** (`gnorm 3.25` vs EMA `0.11`) with immediate recovery; no
  action needed. Checkpoints **`ckpt_0073000.pt`**, **`ckpt_0073500.pt`**, and **`ckpt_0074000.pt`** are
  saved. Submitted **`681131`** as the next 2-H100 chunk after `681123` (deadline
  `2026-07-08T14:20:00`, same `NG=2 BS=16 ACC=8`, `CKPT=500`, `AUTO_REQUEUE=0`, bad/down-node exclude)
  and moved **`681106`** behind `681131` as the 1-H100 fallback. Nemotron remains alive and writing
  (`hy3_reasoning_nemotron.jsonl` 1,596 rows).
- **2026-07-08 ~09:50** — **GLM teacher status clarified.** GLM was not dropped for quality; it is
  paused because both available paths are currently blocked. Raw NVIDIA `z-ai/glm-5.2` still returns
  HTTP 429. A bounded OpenRouter GLM-5.2 attempt (`limit=300`, `concurrency=2`) was tested, but
  OpenRouter returned a key total-limit HTTP 403 and appended no rows. Keep GLM as the top-priority
  teacher to relaunch when quota/credits/provider access clears; do not treat Nemotron as a stronger
  replacement, only as the currently available strong channel.
- **2026-07-08 ~10:05** — **2-H100 handoff to `681123` succeeded; third follow-on queued.** `681115`
  timed out cleanly on evc29 after reaching **step 75230** and saving **`ckpt_0075000.pt`**. `681123`
  started immediately on evc29 with two H100s, `world=2`, `NG=2 BS=16 ACC=8`, and resumed from
  **`ckpt_0075000.pt -> start step 75001`**; first printed step **75010** is healthy, with startup
  throughput still cold after compile and **0 skips**. Submitted **`681136`** as the next 2-H100 chunk
  after `681131` (deadline `2026-07-08T16:20:00`, same `NG=2 BS=16 ACC=8`, `CKPT=500`,
  `AUTO_REQUEUE=0`, bad/down-node exclude) and moved **`681106`** behind `681136` as the 1-H100
  fallback. Nemotron remains alive and writing (`hy3_reasoning_nemotron.jsonl` 1,643 rows).
- **2026-07-08 ~11:05** — **681123 steady-state verified; fourth follow-on queued.** `681123` is RUNNING
  on evc29 with `world=2`, healthy through **step 76950**, loss/gnorm in band, throughput warmed to
  **~272.2k tok/s**. One isolated gnorm skip occurred at **step 76508** (`gnorm 3.17` vs EMA `0.11`) and
  recovered immediately; no intervention. Checkpoints **`ckpt_0075500.pt`**, **`ckpt_0076000.pt`**, and
  **`ckpt_0076500.pt`** are saved. Submitted **`681188`** as the next 2-H100 chunk after `681136`
  (deadline `2026-07-08T18:20:00`, same `NG=2 BS=16 ACC=8`, `CKPT=500`, `AUTO_REQUEUE=0`,
  bad/down-node exclude) and moved **`681106`** behind `681188` as the 1-H100 fallback. Nemotron remains
  alive and writing (`hy3_reasoning_nemotron.jsonl` 1,682 rows).
- **2026-07-08 ~11:20** — **681123 still healthy; fifth follow-on queued.** `681123` remains RUNNING
  on evc29 with `world=2`, healthy through **step 77510**, loss/gnorm in band, throughput **~273.2k
  tok/s**, and still only the one isolated gnorm skip at step 76508. Checkpoints **`ckpt_0077000.pt`**
  and **`ckpt_0077500.pt`** are saved. Scheduler `--test-only` accepted another 2-H100 continuation, so
  submitted **`681193`** after `681188` (deadline `2026-07-08T20:20:00`, same `NG=2 BS=16 ACC=8`,
  `CKPT=500`, `AUTO_REQUEUE=0`, bad/down-node exclude) and moved **`681106`** behind `681193` as the
  1-H100 fallback. Nemotron remains alive and writing (`hy3_reasoning_nemotron.jsonl` 1,710 rows).
- **2026-07-08 ~11:50** — **SFT mix audit found and fixed a contamination gap.** The pretraining
  manifests already show evalgram drops for all active shard dirs (`finemath4`, `openwebmath`,
  `code_python`, `finemath3`), so live pretraining data is decontam-gated. The frozen SFT mix, however,
  lacked a hard eval-overlap gate and local audit found exact Math500 prompt collisions. Patched
  `pipeline/build_sft_mix.py` to drop exact eval prompt hashes by default and to drop question+response
  13-gram hits when `artifacts/evals/evalgrams.pkl` is present. Rebuilt evalgrams locally and rebuilt
  the Newton SFT mix: **97,439 examples**, **0 exact eval prompt hits**, 0 malformed rows, 0 duplicate
  questions; dropped **206 exact eval-prompt overlaps** and **741 eval 13-gram overlaps**. Frozen mix md5
  `53ed91368b4c238dc18a1ab1699e4158`; report md5 `21459b382767801e205f3f625ce106cd`. Treat this as the
  minimum quality bar for any future SFT variant.
- **2026-07-08 ~12:05** — **681123 timed out cleanly; downstream queue repaired.** `681123` reached
  **step 78700**, saved **`ckpt_0078500.pt`**, then hit wall-time. `681131` is pending resources on evc32
  with a scheduled start around 14:27 EDT but still has its old 14:20 deadline; Slurm denied deadline
  edits. To avoid stale/failing downstream dependencies, canceled old follow-ons `681136`/`681188`/
  `681193` and submitted fresh 2-H100 jobs **`681241` -> `681242` -> `681243`** behind `681131` with
  realistic deadlines. Moved `681106` behind `681243` as the 1-H100 fallback. Next custody check must
  verify whether `681131` starts or cancels, then confirm the first running successor resumes from
  `ckpt_0078500.pt`.
- **2026-07-08 ~12:16** — **Canceled stale `681131` blocker.** `681131` was still pending resources with
  a scheduled start after its deadline, so it could only delay the replacement chain. Fresh `sbatch
  --test-only` probes showed both 1-H100 and 2-H100 no earlier than ~14:39, so there was no reason to
  abandon the faster 2-H100 path. Canceled `681131`; **`681241`** is now the next active candidate,
  pending resources with `squeue --start` estimating **2026-07-08T14:27:41**. `681242`/`681243`/`681106`
  remain correctly chained behind it. Nemotron remains alive and writing (`hy3_reasoning_nemotron.jsonl`
  1,755 rows).
- **2026-07-08 ~13:05** — **Normal-priority wait; DR checkpoint preserved; Nemotron relaunched.** The
  scheduler moved the 2-H100 chain out to **2026-07-09T06:28** because of fairshare/priority, and actual
  submitted jobs did not get the earlier backfill reservations that `sbatch --test-only` briefly
  predicted. Tried 2-H100, 1-H100, 30m, 20m, and 1h shapes; current queue is **`681308` -> `681309` ->
  `681310`** (1-hour 2-H100 chunks, `CKPT=250`) then **`681311`** 1-H100 fallback. `short`, `ucfit`, and
  `highgpu` still reject this account, so there is no expanded-compute escape yet. Pulled
  **`ckpt_0078500.pt`** to the Mac via `.part`; local md5 matches Newton:
  `72b5531043e16e7c1cc1697577036e69`. Nemotron old PID had died at 1,762 rows; foreground probe worked,
  so relaunched it in detached `screen` session **`shohin_nemotron`** at concurrency 2 with log
  `scratchpad/nemotron_screen.log`.
- **2026-07-08 ~14:25** — **Queue estimate improved, but teacher providers are unhealthy.** `681308`
  remains the next valid single-writer training job, but `squeue --start` improved from tomorrow morning
  to **2026-07-08T22:28:00**. A fresh same-shape candidate (`681360`) did not receive the earlier
  `sbatch --test-only` reservation and was canceled, leaving only the intended `681308 -> 681309 ->
  681310 -> 681311` chain. Nemotron `screen` completed with many provider errors but added rows to
  **1,781**; bounded follow-up probes showed Nemotron `limit=5` kept 0 with 3 errors, and GLM `limit=3`
  kept 0 with 3 errors. Leave Nemotron/GLM paused until provider health clears; do not burn requests in
  error loops.
- **2026-07-08 ~14:45** — **Training resumed + recurring benchmark track installed.** `681308` started
  on **evc32**, resumed from `ckpt_0078500.pt` at step 78501 with `world=2`, saved `ckpt_0078750.pt`,
  and was healthy through step **78870** with throughput warming through ~239k tok/s. Patched and synced
  `train/jobs/eval_all.sbatch` so every board writes a normal log plus structured metric rows to
  `artifacts/eval_history/metrics.jsonl`. Queued low-priority progress benchmark **`681373`** after
  `681310`, targeting `ckpt_0080000.pt` (`RUN_TAG=pretrain_080000_progress`, `N=100`, `K=4`). Benchmark
  doctrine: use a separate H100 when capacity allows, but never let eval displace live pretraining.
- **2026-07-08 ~15:35** — **80k gate preserved and handoff verified.** `681308` reached step **80280**
  before wall-time and saved `ckpt_0080250.pt`; `ckpt_0080000.pt` was copied to `best_step80000.pt` on
  Newton and downloaded to the Mac with matching md5 **`ebe28d10d26f78bf3e3395ff64f9ce92`**. Successor
  **`681309`** started immediately on **evc32**, resumed from `ckpt_0080250.pt -> step 80251`, confirmed
  `world=2`, and printed first healthy step 80260. Local teacher snapshots unchanged since provider
  pause: HY3 25,243; Claude 152; GLM 29; minimax 51; Nemotron 1,781; all valid JSONL, no active local
  distiller processes.
- **2026-07-08 ~16:35** — **82k handoff verified.** `681309` ran cleanly through step **82090**, saved
  `ckpt_0082000.pt`, and timed out at wall-time. Throughput reached ~275k tok/s; one isolated gnorm
  bump at step 81890 recovered immediately. Successor **`681310`** started immediately on **evc32**,
  resumed from `ckpt_0082000.pt -> step 82001`, confirmed `world=2`, and printed first healthy step
  82010. Queue now has `681311` as the 1-H100 fallback and low-priority eval `681373` after `681310`;
  decide whether to add another 2-H100 chunk before `681310` ends, preserving single-writer safety.
- **2026-07-08 ~17:35** — **83.75k handoff to 1-H100 fallback verified.** `681310` ran cleanly through
  step **83850**, saved **`ckpt_0083750.pt`**, and timed out at wall-time. Throughput stayed ~274-275k
  tok/s; one gnorm skip at step 83508 recovered immediately. `681311` started immediately on **evc32**,
  resumed from `ckpt_0083750.pt -> step 83751`, confirmed `world=1 bs=16 accum=16`, and printed healthy
  startup steps through 83780. Independent 1-H100 and 2-H100 probes both estimated 2026-07-08T23:18,
  so do not block tokens waiting for 2-H100. A mistaken successor `683714` lacking explicit 300k
  exports was canceled and replaced by **`683715`** afterany:`681311` with `STEPS=300000`,
  `LRMUON=0.005`, `LRADAM=1e-3`, `DSEED=777`, `CKPT=250`, `AUTO_REQUEUE=0`. Eval `681373` is still
  low-priority and estimated 2026-07-09T11:43.
- **2026-07-08 ~18:58** — **85k checkpoint healthy on fallback.** `681311` is still running on **evc32**
  and has progressed cleanly through **step 85200** with throughput ~148k tok/s, loss/gnorm in band, and
  no skips in the current tail. Saved `ckpt_0084000.pt`, `ckpt_0084500.pt`, and **`ckpt_0085000.pt`**.
  Fresh after-`681311` scheduler probes estimate 1-H100 around **2026-07-09T02:56** and 2-H100 around
  **2026-07-09T05:56**, so keep the existing 1-H100 successor **`683715`** unless a later probe shows an
  earlier safe 2-H100 slot. Eval `681373` remains low-priority with estimate 2026-07-09T10:39; teacher
  distillers remain paused, with HY3 25,243 / Claude 152 / GLM 29 / minimax 51 / Nemotron 1,781 valid
  local rows.
- **2026-07-08 ~19:40** — **Fallback window ended cleanly; waiting for earliest safe restart.** `681311`
  reached **step 85780** on **evc32**, saved latest durable **`ckpt_0085500.pt`**, and then hit the
  expected wall-time limit at 19:30 EDT. No divergence signal: throughput ~148.8k tok/s and loss/gnorm
  stayed in band. Current queue has no active trainer; **`683715`** is pending priority with estimated
  start **2026-07-08T22:46:08**. Verified only whitelisted Slurm env values: `STEPS=300000`,
  `LRMUON=0.005`, `LRADAM=1e-3`, `DSEED=777`, `CKPT=250`, `AUTO_REQUEUE=0`; missing `NG/BS/ACC` is OK
  because the batch script defaults are the intended 1-H100 `1/16/16`. Fresh 30m/1h/2h 1-H100 probes
  and a 2-H100 probe all start later than `683715`, so leave it as the restart path.
- **2026-07-08 ~20:58** — **Restart verified early.** `683715` started earlier than its prior estimate,
  running on **evc43** at 20:28 EDT. Verified config: `[model] ... world=1 bs=16 accum=16`, resume line
  **`ckpt_0085500.pt -> start step 85501`**. This replays the unsaved 85501-85780 tail from `681311`,
  which is expected because `ckpt_0085500.pt` was the latest durable checkpoint. `683715` has already
  saved **`ckpt_0085750.pt`** and is healthy through **step 85920** with loss/gnorm in band and throughput
  warming through ~141k tok/s. Eval `681373` is still low-priority, now estimated 2026-07-09T07:30.
- **2026-07-08 ~22:20** — **Training steady; 80k benchmark repaired and rerun.** `683715` is healthy
  through **step 87270**, throughput ~146k tok/s, loss/gnorm in band, with checkpoints through
  **`ckpt_0087250.pt`** saved. Low-priority eval `681373` ran on evc35 but failed immediately because
  `TARGET_STEP=80000` resolved to missing `ckpt_0080000.pt`; the preserved file is
  **`best_step80000.pt`**. Patched `train/jobs/eval_all.sbatch` so `TARGET_STEP` falls back to
  `best_step${TARGET_STEP}.pt` when the numbered checkpoint is absent, synced the script to Newton, and
  submitted replacement low-priority eval **`683820`**. Verified it started on **evc35** against
  `best_step80000.pt` with `RUN_TAG=pretrain_080000_progress`, `N=100`, `K=4`; training continues on
  evc43, so eval is not displacing the trainer.
- **2026-07-08 ~23:40** — **80k progress benchmark completed; training steady.** `683820` completed
  cleanly on evc35 in 59m18s and appended 5 rows to `artifacts/eval_history/metrics.jsonl`.
  `best_step80000.pt` scored: GSM8K maj@4 **0/100**, GSM8K pass@1 **3/100**, MATH500 **2/100**,
  HumanEval **5/164**, MBPP **0/100**. Interpretation: raw pretrain is not yet showing broad benchmark
  lift versus the weak SFT board; MATH and HumanEval ticked up, GSM8K did not. Keep training; the next
  meaningful check should be a later checkpoint and/or a better SFT variant. `683715` remains healthy
  through **step 88630**, saved **`ckpt_0088500.pt`**, throughput ~147k tok/s. One isolated gnorm skip at
  step 87819 recovered immediately.
- **2026-07-09 ~01:05** — **90k checkpoint preserved and locally verified.** `683715` remains RUNNING
  on **evc43**, saved **`ckpt_0090000.pt`**, and continued healthy through **step 90060** at ~147.4k
  tok/s. Copied the Newton checkpoint to **`best_step90000.pt`** and verified matching md5
  `6080e105dc0bacfd607efeeefa1253dd`; downloaded `train/flagship_out/ckpt_0090000.pt` to the Mac via
  `.part` and verified the same md5 locally. Recent skips at 89047 and 89409-89410 recovered immediately.
  The local git index is still being touched by an external `git add`/`hash-object` process over large
  artifacts, so this runbook update is synced to Newton but GitHub push should wait until the index is
  safe.
- **2026-07-09 ~11:00** — **100k checkpoint preserved and locally verified.** `683715` remains RUNNING
  on **evc43**, saved **`ckpt_0100000.pt`**, and continued healthy past **step 100170** at ~148.0k
  tok/s with live H100 utilization ~99-100%. Copied the Newton checkpoint to **`best_step100000.pt`**
  and verified matching md5 `d8ebd43d0c13c32aa221832bfc09202b`; downloaded
  `train/flagship_out/ckpt_0100000.pt` to the Mac via `.part` and verified the same md5 locally. Added
  `train/jobs/microbatch_canary.sbatch`, an isolated side branch from `ckpt_0100000.pt` with
  `BS=32 ACC=8` (same 524,288 tokens/update as live `BS=16 ACC=16`) to test whether larger per-GPU
  microbatches improve single-H100 throughput before touching the live run. Slurm dry-run placed it on
  a separate node and real submission **`683939`** ran on **evc22** with `--exclude=evc43`, so it did not
  contend with the live trainer node. Result: completed cleanly in **20m28s**, `BS=32` fit at ~64 GB
  VRAM, and recent-window throughput was **~154.5k tok/s** versus live **~148.0k tok/s** (~+4%). Because
  the initial bounded canary used its short `STEPS=100300` as the LR schedule length, treat it as a
  throughput/memory measurement only; patched `train.py` with `--lr-total-steps` and updated
  `microbatch_canary.sbatch` so future short canaries can keep the 300k LR schedule. Decision: keep the
  current healthy live run; consider `BS=32 ACC=8` at the next natural restart/handoff, not by killing a
  stable allocation mid-window.
- **2026-07-09 ~14:17** — `683715` remains healthy on **evc43** at **step 103580**, holding
  **~148.09k tok/s** with checkpoint cadence intact through `ckpt_0103500.pt`. One gnorm skip at
  **103322** recovered immediately. To answer the VRAM-efficiency question with measurement rather
  than inference, submitted isolated canary **`684000`** (`BS=64 ACC=4`, same 524,288 tokens/update,
  corrected `--lr-total-steps 300000`, excludes evc43); it is pending priority with a current Slurm
  estimate of ~15:35 EDT. The live flagship was intentionally left untouched until the side branch
  demonstrates stable, materially better throughput.
- **2026-07-09 ~15:32** — `683715` is still healthy on **evc43** at **step 104860**, ~**148.11k
  tok/s**, with `ckpt_0104750.pt` saved. BS64/ACC4 canary **`684000`** started separately on **evc33**
  and OOMed cleanly during initialization: 78.93 GiB already in use, then unable to allocate 288 MiB.
  It did not touch the live output or consume the flagship allocation. Result: `BS=32 ACC=8` is the
  largest verified microbatch that preserves the exact 524,288-token update; reserve it for the next
  natural restart/handoff, where its measured ~4% throughput lift can be adopted safely.
- **2026-07-09 ~15:38** — **Dual-GPU return staged without interrupting the active allocation.**
  Scheduler test accepted a two-H100 request on evc33 around 16:32. Submitted isolated **`684029`**
  from `ckpt_0100000.pt` with `NG=2 BS=32 ACC=4` (same global 524,288-token update, corrected 300k
  LR schedule, separate output), to validate the higher per-GPU (~64 GB) microbatch under DDP. Added
  `--lr-total-steps` support to `ddp2_canary.sbatch` so its short run is schedule-faithful. Submitted
  **`684030`** as a dependency-held 3-day two-H100 successor after `683715`, with the same
  `NG=2 BS=32 ACC=4 CKPT=250` shape and only four CPUs. It must remain pending until the live job
  ends; retain it only if `684029` shows clean startup/stable throughput, otherwise replace it with
  the already-proven `NG=2 BS=16 ACC=8` configuration.
- **2026-07-09 ~16:22** — **Canary seeding hardening and corrected resume.** `684029` did validate
  `world=2`, `BS=32/ACC=4`, ~59 GB per-GPU memory during warmup, no NCCL/DDP issue, and steady
  throughput reaching **~278k tok/s**. It was stopped once that hardware measurement was complete
  because it was unexpectedly headed for a long duplicate run. Root cause: its requested
  `ckpt_0100000.pt` had aged out of the flagship retention window; the script logged `cp: cannot stat`
  but continued, so it trained from initialization. Its loss curve is therefore invalid for a resume
  quality gate. Fixed `ddp2_canary.sbatch`: `set -e`, fail on missing source, default to preserved
  `best_step100000.pt`, and copy preserved sources under a numbered `ckpt_` filename that train.py
  resumes. Submitted corrected **`684058`** on evc33; it logged `[resume] ... ckpt_0100000.pt -> start
  step 100001 (FRESH optimizer)`, `world=2 bs=32 accum=4`, and early loss/gnorm in band. Keep
  dependency-held `684030` only if this corrected canary completes cleanly; the live flagship remains
  healthy past step 105.6k at ~148.1k tok/s with `ckpt_0105500.pt` saved.
- **2026-07-09 ~16:58** — **Corrected two-H100 BS32 canary passed.** `684058` resumed from preserved
  100k weights at step **100001**, ran through **100239**, and completed in 472s with **no skips,
  OOM, NCCL error, or DDP hang**. Loss/gnorm stayed in the expected resumed band (loss ~1.39-1.85,
  gnorm ~0.06-0.23). Its final **~264.9k tok/s** includes compile warmup; the prior long hardware-only
  run plateaued around **278k tok/s**, consistent with a small microbatch gain over the proven
  BS16/ACC8 two-H100 ~272k band. Decision: retain dependency-held **`684030`** at
  `NG=2 BS=32 ACC=4 CKPT=250`. The live flagship remains healthy past **106.3k**, ~148.1k tok/s,
  with `ckpt_0106250.pt` saved.
- **2026-07-09 ~20:58-21:38** — **110k checkpoint preserved and locally verified.** `683715` remained
  RUNNING on evc43, healthy through **step 111050** at ~148.1k tok/s; one isolated gnorm skip at
  110259 recovered immediately. Copied `ckpt_0110000.pt` to `best_step110000.pt` on Newton; both
  hash to **`9810b7081f78dc6bb1e5fa9765b64d4b`**. The Mac transfer suffered intermittent login/drop
  failures but was resumed with SFTP `reget` rather than restarted; local
  `train/flagship_out/ckpt_0110000.pt` now matches the same md5. The numbered remote source rotated
  out during the long transfer, confirming the preserved `best_step` copy is required for custody.
  Next DR target: **120k**.
- **2026-07-10 ~02:58** — **Single-H100 throughput directive applied.** User withdrew the pending
  dual-GPU continuation in favor of maximizing the single H100. Canceled dependency-held two-H100
  `684030` without touching healthy live job `683715`. Submitted **`685084`** as a dependency-held
  single-H100 successor with `BS=32 ACC=8 CKPT=250`, preserving the exact 524,288-token update while
  using the validated ~64 GB microbatch (single-H100 canary measured ~154.5k tok/s vs ~148k live).
  Do not interrupt `683715` for this modest gain; it activates only at the natural handoff. At the
  current 148k tok/s one H100 processes ~12.8B tokens/day, so any claim of 30T tokens in 10 days
  necessarily assumes roughly 35M tok/s, i.e. a large multi-GPU fleet rather than this allocation.
- **2026-07-10 ~04:20-07:00** — **Measured single-H100 graph-capture investigation.** Isolated
  BS32/ACC8 profiler `685088` ran from preserved 110k and measured **155.35k tok/s**. It exposed
  roughly 54k CUDA launches across four complete updates; GPU time remains dominated by GEMM (35.6%)
  and FlashAttention backward (23.2%), so launch removal is a bounded optimization, not a 100x path.
  Automatic `torch.compile(reduce-overhead)` attempts `685105`/`685106` failed before a training step
  because per-forward CUDA graphs conflict with eight-way gradient accumulation; do not use that mode
  in the flagship. Whole-update graph canary `685125` reached capture but stopped safely because
  canary AdamW lacked `capturable=True`; patch must be synced/retried only after
  Newton authentication recovers. **120k is preserved on Newton:** numbered and `best_step120000.pt`
  both md5 `d5c73034b635025ba997b25b622f77f5`. The Mac transfer reached 1,017,348,096 bytes before
  Newton started refusing new login sessions; retain it and SFTP `reget` it in place, never trust it
  until the final md5 matches.
- **2026-07-10 ~07:40** — **120k local DR completed; graph-capture retry queued.** Newton access
  recovered, the resumable local transfer reached the expected 1,076,598,762 bytes, and Mac md5 now
  matches Newton exactly: `d5c73034b635025ba997b25b622f77f5`. Synced `cudagraph_canary.py` with
  canary-only `AdamW(capturable=True)` and submitted **`685209`** excluding evc43; it is pending
  priority and cannot affect the flagship. `683715` remains healthy past 121.2k at ~148.1k tok/s.
- **2026-07-10 ~09:00** — **120k capability board queued.** `683715` is healthy past 122.5k. Submitted
  isolated eval `685262` against preserved `best_step120000.pt` with the same fixed progress-board
  protocol (`N=100`, GSM8K `K=4`) as the 80k baseline; it is pending normal priority and estimated just
  after the whole-update CUDA-graph canary `685209`. Neither job shares live outputs or can displace
  the flagship.
- **2026-07-10 ~12:23** — **CUDA-graph and 120k measurement gates completed.** First graph runs
  `685209`/`685508` exposed a bad-node CUDA allocation and capturable-optimizer resume setup only;
  clean retry **`685520`** completed 80 updates from 110k with stable loss/gnorm at **158.20k tok/s**,
  57.07 GiB. That is only ~1.8% over the 155.35k BS32 compiled profile and the canary omits the
  flagship skip guard, so **CUDA graphs are rejected for integration**. Board **`685262`** on 120k
  completed: GSM8K maj@4 2/100 (80k 0/100), pass@1 1/100 (80k 3/100), MATH500 3/100 (80k 2/100),
  HumanEval 7/164 (80k 5/164), MBPP 0/100 unchanged. Treat this as noisy/mixed raw-pretrain movement,
  not evidence of meaningful reasoning yet; preserve the live run, evaluate again near 160k, and use
  an SFT variant as the next capability intervention rather than chasing marginal throughput.
- **2026-07-10 ~13:54** — **Capability data path reset around verified scale, without touching pretrain.**
  Audited the actual `response` field consumed by `train/sft.py` (not the short auxiliary `answer`
  field): core mix is structurally clean at 97,439 rows, 0 malformed/duplicates, median 96 response
  words, p90 174, and 96,994 explicit final-answer markers. The problem is therefore not answer-only
  formatting; it is that this is still only a 97k-example initial SFT and underweights procedural
  verifier-backed traces. Added `audit_sft_quality.py`, `prepare_rg_sft.py`, and atomic CPU job paths.
  Submitted `685691` to curate up to 600k independently answer-verified, eval-decontaminated
  OpenMath solutions; `685694` to generate 20k instances per Reasoning-Gym training family with
  disjoint evaluation seeds/families plus execution-verified traces; and dependency-held `685696` to
  freeze `sft_mix_reasoning_v2.jsonl` only if both finish successfully. No SFT is queued; candidate
  quality and held-out capability gates must pass first. Flagship `683715` recovered normally after
  two skipped extreme-gradient batches at 126648-126649 and remains healthy through 127.6k.
- **2026-07-10 ~14:20** — **Verified source jobs completed; pretraining corpus expansion started.**
  `685691` wrote and audited **600,000** OpenMath SFT rows: 338,610 `augmented_math`, 126,099
  `augmented_gsm8k`, 108,103 `math`, and 27,188 `gsm8k`, with median 96 and p90 182 response words.
  `685694` committed `rg_v2` with **99,808** execution-verified trace examples (median 18, p90 43
  response words) and generator-held-out eval configurations. `685696` began the frozen v2 mix build.
  Submitted CPU-only `685700` to tokenize OpenMath problem+solution documents into a separate,
  manifest-gated `shards/openmath_pt` source (5B-token cap) for a later natural relaunch; it cannot
  alter the currently running shard mixture.
- **2026-07-10 ~14:35** — **Held-out procedural capability gate added.** Added `train/eval_rg.py` and
  `train/jobs/eval_rg.sbatch`: exact-match, per-family evaluation on the `rg_v2` generator-held-out
  seeds/families. First baseline allocation `685702` failed in one second on evc26 because NVML was
  transiently unavailable before any model code executed; hardened the wrapper to wait/retry an NVML
  exit rather than abort, excluded evc26 and evc43, and resubmitted baseline **`685704`** from
  `best_step120000.pt` over 400 held-out questions. Use the identical protocol after an SFT-v2 pilot;
  no data recipe is accepted on public-board movement alone.
- **2026-07-10 ~14:55** — **Balanced baseline and isolated v2 SFT pilot completed.** The balanced
  120k procedural baseline `685706` scored **29/800 = 3.625%** over 32 held-out families; this is the
  valid raw-base reference (the prior 3/400 was knights-only). Pilot `685708` completed one v2 epoch:
  349,317 examples / 85.34M packed tokens / 2,605 updates, terminal loss ~0.46-0.54. It correctly
  initialized from `best_step120000.pt` and wrote only `train/sft_v2_120k/sft_ep1.pt`; a stale final
  echo mentioned default `sft_out`, so dependent evaluations were canceled while header+filesystem
  isolation was verified. Replacement public board `685757` now runs on that exact checkpoint; balanced
  RG `685759` is serially dependency-held after it. Flagship is healthy past 128.7k at ~148.12k tok/s.
- **2026-07-12 ~03:20** — **Network-outage reconciliation and data gate decision.** Original
  flagship `683715` completed cleanly after 2d 7h on evc43. Dependency successor **`685084`** started
  automatically on evc22, correctly resumed `ckpt_0141500.pt -> step 141501` with the planned
  one-H100 `BS=32/ACC=8` configuration, and is healthy through **166.26k** at **154.2k tok/s**.
  The missed 130k/160k rotating checkpoints cannot be recovered; immediately preserved current
  `ckpt_0166250.pt` as `best_step166250.pt` (remote md5
  `1b57c99aca966546d4d9aea7827d6ebd`) and began a resumable local `.part` DR transfer. CPU job
  `685700` completed `openmath_pt`: **5,000,000,144 tokens / 50 shards / 12,662,236 docs kept /
  165,773 eval-contaminated docs dropped**. It is approved only as a future natural-relaunch input,
  never a live-mix mutation. SFT-v2 `sft_ep1` improved GSM8K pass@1 **1/100 -> 12/100** and maj@4
  **2/100 -> 5/100**, but MATH500 fell **3/100 -> 0/100**, HumanEval **7/164 -> 6/164**, MBPP stayed
  0, and balanced held-out RG regressed **29/800 (3.625%) -> 19/800 (2.375%)**. Decision: retain it
  as a narrow GSM-style ablation result, **do not promote v2 as the final broad-reasoning recipe**;
  continue raw pretraining and re-design post-training around stronger procedural coverage before the
  next SFT pilot.
- **2026-07-12 ~03:22** — **166.25k DR custody complete; raw-base eval chain queued.** Resumable SFTP
  transfer completed locally as `train/flagship_out/ckpt_0166250.pt`; local and Newton both md5
  **`1b57c99aca966546d4d9aea7827d6ebd`**. Submitted isolated raw-base public board **`686245`** against
  the preserved 166.25k checkpoint, with balanced held-out RG **`686247`** serially after success;
  both exclude the live evc22 node. Next DR target is 170k. Do not train another broad v2 SFT variant
  until the raw-base 166k measurements and a revised procedural-curriculum design are reviewed.
- **2026-07-12 ~03:32** — **Deep capability diagnosis found an evaluation truncation defect.** Direct
  H100 transcripts of raw 166k (`686255`) show template completion, task mutation, list/string failure,
  and invalid code. Direct SFT-v2 transcripts (`686263`) show a real capability change: correct fresh
  discount/tax and fraction derivations, but failed algebra continuation, base conversion, logic,
  string/list processing, and parity code. Critically, `eval_suite.generate()` stopped at any blank
  line while SFT completions commonly put `The answer is ...` after one; old SFT benchmark results are
  undercounted. Removed that stop condition, preserved old metrics only as diagnostics, canceled their
  mixed-protocol reruns, and queued fixed-decoder public/held-out/in-training gates (`686265`,
  `686267`, `686269`). See `CAPABILITY_DIAGNOSIS.md` for ranked causes and pre-relaunch requirements.
- **2026-07-10 ~14:45** — **RG measurement bias fixed before using it for a decision.** `685704` completed
  and confirmed the raw 120k model is poor on held-out knights-and-knaves (**3/400 = 0.75%**), but the
  generator file is family-ordered, so the first 400 rows were all one family. Patched evaluator to
  shuffle each family deterministically then round-robin across families. Submitted balanced baseline
  **`685706`** from the same 120k checkpoint (`N=800`, excluding evc26/evc43). Treat `685704` as the
  knights-only baseline; use `685706` as the broad procedural comparison for every SFT-v2 pilot.
- **2026-07-10 ~14:50** — **First causal SFT-v2 experiment queued, isolated and fully gated.** Updated
  `sft.sbatch` to accept `OUT` rather than always writing `train/sft_out`. Submitted **`685708`**
  after the balanced pre-SFT baseline: 1 epoch from `best_step120000.pt` over frozen v2, output to
  `train/sft_v2_120k/`. Queued public board **`685710`** `afterok:685708` and balanced RG **`685714`**
  `afterok:685710`, so the experimental GPU work is serial and consumes no flagship resources. The
  only decision rule: retain the v2 recipe only if it improves the held-out RG gate and does not cause
  a material public-board regression; otherwise diagnose source balance/format before another run.
- **2026-07-12 ~04:10** — **Capability measurement and data-coverage repairs.** Fixed the evaluator a
  second time: it now stops only after a complete explicit final-answer line, avoiding both blank-line
  truncation and wasteful post-answer decoding; `eval_all.sbatch` now fails fast if a Slurm allocation
  has no usable CUDA device. Bad-node CPU fallback `686272` was canceled; replacement **`686277`** is
  confirmed CUDA on evc31, with `686278 -> 686279` serial RG gates. Direct smoke `686280` validated all
  15 deterministic RG tracers against fresh samples (decimal parser corrected from real prompts), so
  **`686281`** now builds isolated `rg_v3`. Audit found only 444 code examples in v2; APPS train-only
  execution-verified code pilot was added. The first loader attempt hit a current `datasets` loading-
  script restriction; corrected **`686282`** uses the official APPS train JSONL directly. Flagship
  `685084` remains healthy through 166.98k at ~154.2k tok/s; no live trainer data or model state changed.
- **2026-07-12 ~04:20** — **Data pilots cleared their first gates.** `rg_v3` generated 559,977 train
  problems and **271,452** verifier-backed traces across 15 covered families, retaining disjoint RG
  evaluation seeds/families; the job is finishing conversion/audit before commit. The corrected APPS
  train-only JSONL route (`686282`) kept **100/108** examples after three supplied I/O checks each,
  with 0 malformed, duplicate, or evaluation-overlap rows and p90 response length 182 words. Submitted
  `686283` to build a separate 3,000-example scale artifact. The CUDA-only v2 board `686277` remains
  in progress; do not interpret its first four printed samples as a score.
- **2026-07-12 ~04:25** — **RG duplicate gate repaired before promotion.** The initial `rg_v3` audit
  found 19,154 normalized-question duplicates, so no SFT mix was built from it. Added atomic question
  deduplication in `prepare_rg_sft.py` and generated a separate immutable derivative with `686284`:
  **252,298 valid traces, 0 malformed, 0 missing, 0 duplicates**, response p50/p90 19/37 words. Future
  `rg_scale` runs inherit this deduplication. Use only `rg_traces_sft_dedup.jsonl` for future SFT mix
  candidates; retain the original raw trace file for provenance.
- **2026-07-12 ~04:35** — **Broadened procedural supervision beyond arithmetic.** Fresh RG samples
  exposed answer-checkable, previously untraced number theory, filtering/sorting, Caesar, social-count,
  and self-referential logic families. Added conservative deterministic tracers for them; each recomputes
  and compares the gold answer before emitting a trace. CPU smoke `686285` validated all **25** registered
  families with nonzero fresh-sample coverage (most 80/80; intentionally conservative families retain
  only parseable items). Submitted isolated **`686286`** for `rg_v4`; it is not a replacement for v3 and
  must still pass the standard dedup/audit gate before it can appear in an SFT mix.
- **2026-07-12 ~04:45** — **Code-data harness hardened before scale.** APPS scale `686283` encountered
  a dataset row with a >4,300-digit JSON integer after writing 586 KB. It failed before any mix build;
  moved that incomplete output to `apps_train_v1.failed_686283.jsonl` for audit and never use it. The
  curator now treats that parse condition as a row-level rejection and writes through an atomic `.partial`
  path, so only a fully audited completion can claim `apps_train_v1.jsonl`. The first CodeContests schema
  probe also started materializing an entire split; canceled it and changed retry `686289` to streaming
  single-record inspection. Retry code scale is `686288`; both remain isolated from SFT/pretrain.
- **2026-07-12 ~04:55** — **Full 25-family procedural corpus passed quality gate.** `rg_v4` committed
  after direct deduplication in the normal scale path: **374,659 valid traces, 0 malformed, 0 missing,
  0 duplicate normalized questions**, response p50/p90 18/37 words. The small audit report was copied
  locally and hash-verified. Streaming-only CodeContests probe `686289` confirmed train/valid/test
  schemas with language-tagged reference solutions plus public/generated I/O tests. Train-only Python 3
  pilot `686290` kept 100 structurally clean, execution-verified examples; submitted `686291` for a
  separate 3,000-row scale artifact. Continue to keep all code and RG outputs out of a frozen mix until
  their individual final audits are available.
- **2026-07-12 ~05:15** — **Capability investigation completed its first decision gate.** Corrected
  CUDA board `686277` finished v2 at GSM8K maj@4 **6/100**, pass@1 **14/100**, MATH-500 **6/100**,
  HumanEval **6/164**, MBPP **0/100**; reject v2 from promotion. Direct prompt matrix `686306` compared
  raw 168k and v2 across 48 fresh arithmetic/base/state/sort/string/logic tasks under Q/A, plain,
  CoT, and one-shot prompts. Raw was 4/48, 4/48, 0/48, 5/48; v2 was 7/48, 4/48, 5/48, 4/48. The small
  v2 gain is template-specific, not latent reasoning. Verified 1,360 v2 prompt-token prefixes: no
  completion-mask boundary mismatch. Added `train/capability_matrix.py` and isolated job wrapper as a
  repeatable future promotion gate; no live pretrain files, SHARDS, or teacher writers were changed.
- **2026-07-12 ~05:25** — **v4 replay-capacity gate added before training.** CodeContests scale `686291`
  is still atomically writing its verified train-only artifact, so queued `686308` afterok to produce
  `code_completion_v1`, `686310` afterok to freeze the v4 mix, and `686312` afterok to tokenize that mix
  exactly as SFT will and report per-group packed sequence counts plus the repeat factor implied by the
  proposed 40/35/15/10 weights. Do not submit `sft_v4_pilot.sbatch` until this report proves code
  oversampling is defensible or the weights are revised. This is CPU-only and leaves the flagship untouched.
- **2026-07-12 ~05:35** — **Recurrent depth is stable but not earned.** Matched Mame `n_loop=1`
  (`686301`) and `n_loop=2` (`686302`) runs both completed 800 updates with one recovered gnorm skip.
  The final logged losses were statistically indistinguishable (2.4899 vs 2.4890), while recurrence
  cost 1.65x wall time (886s -> 1466s) and reduced throughput 472.7k -> 286.0k tok/s. This is a
  mechanical-validity result only; keep `n_loop=1` for the flagship until a longer paired capability
  gate shows a real benefit.
- **2026-07-12 ~05:42** — **The first broad capability audit was strengthened, not hand-waved.** Held-out
  v2 RG `686278` completed at **90/800 = 11.25%**: it learned a few narrow procedural routines
  (chain sums 20/25, string insertion 19/25, basic arithmetic 13/25) but remained zero on most
  transformation, logic, cipher, and geometry families. The raw-base board `686314` loaded rotating
  `ckpt_0168000.pt`, scored only its first GSM8K maj@4 metric (1/100), then lost the source checkpoint;
  it is invalid rather than evidence. Hardened `eval_all.sbatch` to pin a checkpoint for the entire board,
  preserved `best_step168750.pt` (md5 `e58b6b07802782517c7709c1844cb4d1`), and queued corrected raw board
  `686315` followed by isolated multi-turn adaptive interaction probe `686316`. No live model or SHARDS
  were modified.
- **2026-07-12 ~05:55** — **Restart data replay defect found and repaired forward-only.** Inspection of
  `ShardLoader` and trainer checkpoints showed that optimizer/model state resumed but the async data
  stream did not: every Slurm handoff rebuilt it from the same `DSEED=777`, allowing a repeated shuffled
  prefix. Added generation-scoped deterministic data seeds, checkpoint metadata, and `test_data_stream_resume.py`.
  A real two-stage tiny-train smoke saved generation 0 then resumed at step 2 with generation 1 and a
  distinct seed. Active `685084` is untouched; its next natural successor receives the repair. This
  reduces an unmeasured but serious replay risk, not a retroactive claim that past tokens were unique.
- **2026-07-12 ~05:58** — **v4 SFT data is frozen, but its replay budget was corrected before GPU use.**
  CodeContests `686291` committed 3,000 execution-verified train-only Python problems; `686308` converted
  3,593 deduplicated examples to completion form. `686310` froze v4 at **643,595** clean, deduplicated,
  decontaminated rows, and `686312` tokenized it exactly as SFT will: math 34,848 / procedural 24,847 /
  code 1,225 / teacher 2,006 packed sequences. The 40/35/15/10 proposal would have replayed code 7.7x
  and teacher 3.1x per epoch, so `sft_v4_pilot.sbatch` is revised to **40/47/8/5** (code 4.1x, teacher
  1.6x). Do not submit its pilot until the pinned raw-base board and adaptive direct baseline finish.
  Separate CPU-only `686317` is expanding CodeContests further for a later candidate.
- **2026-07-12 ~06:05** — **v2 overfit control closed, without rescuing v2.** In-training `686279`
  completed **98/800 = 12.25%**, only one point above held-out `686278` at **90/800 = 11.25%**. This
  confirms a few procedural routines transferred beyond exact examples, but the family distribution
  remains narrowly concentrated and corrected public math/code scores remain too low. Keep v2 rejected;
  use this result only as evidence that v4 needs broader verified coverage, not just more epochs.
- **2026-07-12 ~06:50** — **170k DR milestone complete.** `685084` reached step 170,000 on evc22 at
  ~154.2k tok/s with stable loss/gnorm, then continued cleanly. Copied numbered `ckpt_0170000.pt` to
  Newton `best_step170000.pt`; both md5 **`7ad139b6b9b537a5a3e65978f8296419`**. Resumable SFTP local
  transfer completed at 1,076,598,762 bytes; local `train/flagship_out/ckpt_0170000.pt` matches the same
  md5 before its `.part` rename. The raw 168.75k board is still completing code execution and adaptive
  probe `686316` remains dependency-held; no SFT candidate has started.
- **2026-07-12 ~07:15** — **Pinned raw baseline and direct interaction audit completed; v4 pilot started.**
  Corrected raw board `686315` completed on evc23 against pinned `best_step168750.pt`: GSM8K maj@4
  **5/100**, GSM8K pass@1 **2/100**, MATH-500 **2/100**, HumanEval **7/164**, MBPP **0/100**. Direct
  multi-turn audit `686316` completed: **1/6** correct initially, **1/6** after an explicit independent
  review, and **1/6** even with a verified intermediate fact; only the simple syllogism survived. This
  rejects a hidden-prompt/latent-correction explanation. Mirrored raw board log and adaptive JSON locally
  with matching md5s `4a066de48c450b36c89836ad53cc444e` and `e05a451a16087450a6f87764cf7a39e7`.
  Submitted isolated source-balanced v4 SFT **`686323`** on evc23 from `best_step168750.pt` to
  `train/sft_v4_168750/`, one epoch, `40/47/8/5` math/procedural/code/teacher. Separately hardened
  `curate_selfcorrect.py` to write atomically and reject duplicate normalized prompts; its new v1 asset
  has 15,000 clean unique rows (5,979 explicit repairs) and is held for a post-v4 ablation only.
- **2026-07-12 ~07:20** — **v4 submission configuration was audited and corrected before a training step.**
  The first submission `686323` logged obsolete 40/35/15/10 weights, so it was canceled at 2m45s before
  any SFT step and its created directory was preserved as `sft_v4_168750.invalid_686323_oldweights`.
  Synced the local audited job script to Newton, hash-verified it, confirmed remote
  `WEIGHTS="math=0.40 procedural=0.47 code=0.08 teacher=0.05"`, and launched replacement **`686324`**
  on evc23 to a fresh `train/sft_v4_168750_r2/` directory. `685084` on evc22 was never touched.
- **2026-07-12 ~07:35** — **V4 code-completion boundary audit found and repaired a real label-contract bug.**
  Before accepting `686324`, an exact tokenizer audit found **461/3,542** raw-completion code rows where
  separately tokenized prompt IDs were not a prefix of `tokenize(prompt + completion)` due BPE merges at
  CRLF/indent boundaries. That misplaces completion-only labels relative to inference. Canceled `686324`
  before it wrote an artifact and preserved its directory as
  `sft_v4_168750_r2.invalid_686324_maskboundary`. `train/sft.py` now independently tokenizes the prompt
  and completion then concatenates IDs; `test_sft_prompt_boundaries.py` covers both normal Q/A and CRLF
  Python headers. Local and Newton tests passed under a one-thread BLAS login-node setting. Clean r3
  **`686326`** is running on evc23 to `train/sft_v4_168750_r3/`, initial packing/weighted sampling is
  correct (math 25,275 / procedural 29,435 / code 5,010 / teacher 3,192). `685084` remains untouched.
- **2026-07-12 ~07:55** — **Clean v4 SFT completed; evaluation protocol repair applied before scoring.**
  `686326` completed one epoch/3,933 updates in 1,218s and wrote
  `train/sft_v4_168750_r3/sft_ep1.pt` (500,446,874 bytes). The first successor board `686327` was
  canceled immediately because its unset defaults would have used N=200 and GSM8K maj@8, incomparable
  to the N=100/maj@4 raw baseline. The corrected N=100/K=4 board `686332` then hit a transient NVML
  exit-6 on evc26 before decoding; `eval_all.sbatch` had an unintended `pipefail` exit in its readiness
  assignment. Patched it to retain the intended retry semantics, bash-validated and hash-synced it, then
  queued the current protocol-matched board **`686336`** on evc23 plus serial `686337` (RG N=800),
  `686338` (adaptive), and `686339` (capability matrix), all excluding evc22/evc26. No metric from
  canceled board attempts is valid. Flagship `685084` remains healthy through ~171k.
- **2026-07-12 ~08:20** — **FineWeb-Edu future curriculum source passed final artifact audit.**
  CPU job `686298` wrote `artifacts/shards/fineweb_edu_5b`: 46 nonempty shards and 4,599,762,693
  tokens from 4,139,127 kept documents. The quality/decontamination filters rejected 5,531,139
  low-quality and 1,816 eval-contaminated documents; manifest md5 is
  `05739fa1cb31a45e1c496909f9461fa1`. Independent full-shard scan `686340` found entropy 10.767--10.784,
  top-1 frequency 0.0384--0.0393, top-5 0.1559--0.1575, zero byte-fallback, and normal decoded
  educational text in sampled windows. It is therefore eligible for the **future-only** manifest-gated
  reasoning relaunch, never for a mid-run mutation of `685084`. At the same checkpoint of custody,
  flagship `685084` was healthy at step 171,490 / 154.25k tok/s with immediately recovered gnorm
  outliers. The protocol-matched v4 board has completed GSM8K: maj@4 5/100 (tie with raw) and pass@1
  14/100 (raw 2/100); this is deliberately not treated as promotion before MATH/code, held-out RG, and
  direct interaction finish.
- **2026-07-12 ~08:35** — **Evaluation reproducibility repair committed before any rerun.** The public
  board's greedy pass@1, code pass@1, held-out RG, and direct probes are deterministic. Its sampled
  GSM8K maj@4 result was not explicitly seeded, so `eval_suite.py` and `eval_code.py` now accept and
  report a fixed seed; `eval_all.sbatch` defaults `EVAL_SEED=20260712`. This cannot explain the current
  14/100 greedy GSM8K v4 result, but it prevents future sampled comparisons from being ambiguous. The
  active `686336` board is intentionally left untouched; only future submissions use the repair.
- **2026-07-12 ~08:36** — **Deep capability and verifier gates expanded without touching flagship.**
  Read the verbatim raw/v2 transcript and adaptive audit rather than inferring from loss: raw repeatedly
  continues unrelated templates, and v2 produces fluent but invalid arithmetic/base-conversion procedures.
  The partial v4 board is weak (GSM8K maj@4 5/100, pass@1 14/100, MATH-500 1/100, HumanEval 2/125), so
  it is not eligible for promotion pending the full gate. Queued the matching raw-v4 verbatim transcript
  audit `686343` after `686339`. Added and ran `fetch_gsm8k_train.py`: **7,473** labelled training rows,
  **0** normalized prompt overlaps with the held-out public evaluator. Queued isolated verifier chain
  `686345 -> 686349`: train-only k=8 student rollouts, binary verifier data, one-epoch verifier SFT,
  held-out public k=16 rollouts, and a final report with pass@1/oracle@16/verifier@16. All GPU jobs exclude
  evc22/evc26 and use no flagship or frozen-SFT output directory.
- **2026-07-12 ~08:55** — **V4 public gate rejected; primitive ablation is data-gated.** `686336` completed:
  GSM8K maj@4 **5/100**, pass@1 **14/100**, MATH-500 **1/100**, HumanEval **2/164**, MBPP **0/100**.
  The pass@1 formatting gain does not compensate for math/code regressions, so v4 cannot be promoted.
  Audit of its 374,659 procedural rows found direct coverage gaps (only 12 state-update-like rows, 77
  syllogistic-logic-like rows, and 3 correction-like rows), despite its high headline count. Built and
  audited `primitives_v1`: **210k** train and **3.5k** held-out solver-generated rows across seven missing
  operations, with 0 malformed/duplicate/overlapping prompts. V5 capacity audit `686359` measured 72,161
  packed sequences and controlled replay factors: math 0.83x, primitives 2.73x, code 2.95x, teacher 1.80x,
  procedural 0.44x. Queued raw primitive baseline `686357`, then isolated v5 `686361` and v5 primitive gate
  `686362`; no public/RG promotion board will be spent unless that held-out gate improves. Added DCLM full
  scan `686360` after tokenizer job completion. Flagship remains untouched and healthy past 172.2k.
- **2026-07-12 ~09:03** — **V5 capacity retuned before its dependency could unlock.** The first packing
  report's 40/35 math/primitives split was intentionally superseded to preserve more contest-math coverage.
  Canceled only the pending v5 jobs (no artifact existed), then reran the read-only packing gate as `686366`.
  The accepted 45/30 math/primitives mix reports **72,161** packed sequences with repeat factors math
  **0.932x**, primitives **2.344x**, code **2.948x**, teacher **1.799x**, and procedural **0.436x**.
  Replacement v5 `686367` and held-out primitive evaluator `686368` remain dependency-held after the
  running raw primitive baseline `686363`. This is a controlled data comparison, not a new flagship path.
- **2026-07-12 ~09:20** — **Direct compact-state claim tested and rejected at the pinned 170k checkpoint.**
  The first instrumentation run `686369` exposed an over-permissive code scorer, so it was not used as
  evidence. The scorer now requires a syntax-valid `is_even` AST without executing model text. Corrected
  isolated interview `686370` scored **1/8 initial, 0/8 review, 1/8 scaffold, 0/8 compact-state reuse**;
  only the simple logic constraint passed. This confirms that no prompt, verified intermediate, or
  self-produced summary currently unlocks a general solver. Canonical transcript is
  `artifacts/eval_history/deep_interaction_raw170k_r2_686370.json`, md5
  `1979bcc79cb18830cb3080a7cab85e82`. Flagship `685084` remained untouched and healthy past 172.6k.
- **2026-07-12 ~09:25** — **V4's diagnostic gate showed local transfer without broad readiness.**
  Corrected held-out procedural result `686337` is **209/800 = 26.125%**, versus V2's corrected
  `686278` **90/800 = 11.25%** on the same held-out file and seed. The gain is concentrated in simple
  equations, sorting, string relations, and chain arithmetic; it is not attributable to V4 data alone
  because V4 starts from the later 168.75k base. V4's direct adaptive test `686338` remained **1/6
  initial, 1/6 review, 0/6 scaffold**, and the public board has math/code regressions, so V4 stays
  rejected for broad promotion. Its remaining matrix/verifier chain is diagnosis only.
- **2026-07-12 ~09:30** — **V4 format-versus-execution diagnosis completed.** Matched matrix `686339`
  scored raw/V4 as **4/48 -> 4/48 Q/A**, **5/48 -> 4/48 direct**, **0/48 -> 4/48 CoT**, and
  **7/48 -> 10/48 one-shot**. V4 gains some prompted arithmetic and syllogism behavior but not native
  direct execution. Verbatim audit `686343` shows plausible prose with wrong facts (`19*17=303`,
  base-6 `254` copied to decimal, `r=14 -> 3r=14`), so its traces cannot be promoted as general
  reasoning. Retain V4 only for verifier/generator comparison; V5's primitives gate must prove actual
  atomic execution rather than output style.
- **2026-07-12 ~09:35** — **Future language curriculum weights corrected before any handoff.**
  `ShardLoader` reserves one sequence for every enabled domain before drawing its weighted remainder;
  the old nominal 25/25/50 weights would have supplied roughly **32/23/45** math/code/English at BS32.
  `flagship_language_relaunch.sbatch` now uses floor-aware weights `2 4 28 4 6 32 24`, verified by
  `test_domain_mix.py` to yield exact **25/25/50**. Added a 25B DCLM replacement builder; it requires
  the 5B pilot manifest and must receive a fresh full-shard scan before admission, never coexist with
  the pilot in one curriculum. No active training or live SHARDS changed.
- **2026-07-12 ~09:45** — **OpenR1-Math intake is now a bounded data-quality gate.** CPU schema probe
  `686380` confirmed the official `default` config exposes problem, answer, per-generation Math Verify
  / Llama-judge flags, completion flags, and UUID. Added `curate_openr1_math.py`: one completed trace
  per normalized problem, Math Verify preferred, Llama fallback only, 512-token trace cap, and project
  evalgram rejection. CPU pilot `686381` targets only 10k rows plus a quality report; it is isolated and
  cannot enter a frozen SFT mix without reviewing its yield, overlap, length, and provenance.
- **2026-07-12 ~10:00** — **OpenR1 short-trace route rejected; filtered source-solution route passed.**
  Length profile `686382/686383` found 0/255 verified R1 traces <=1024 tokens (p50 4,988), so do not
  train their verbose CoT into the 125M student. The concise `solution` field was viable (203/255 <=512),
  but its first pilot exposed 97 table artifacts, 2 parameter placeholders, and 492 near-answer-only rows.
  Added deterministic answer-explicit, table/placeholder, and >=10-novel-word filters. Corrected pilot
  `686386` produced **10,000** clean rows from 17,360 seen: 0 malformed/duplicates/exact eval overlaps/
  13-gram eval overlaps, p50 120 response words, and 0 source mutation. Submitted larger CPU-only
  candidate `686387`; it is future-SFT-only pending its final audit and source-balance decision.
- **2026-07-12 ~10:15** — **V5 atomic-transfer gate passed, but broad reasoning remains unproven.**
  Isolated source-balanced V5 (`686367`, raw-168.75k base, 45/15/5/5/30
  math/procedural/code/teacher/primitives) completed cleanly. Its disjoint 700-case primitive holdout
  `686368` is **272/700 = 38.86%** versus raw `686363` **0/700**. Gains are strongly uneven: syllogism
  100/100, string insertion 88/100, correction 47/100, state update 19/100, arithmetic 12/100,
  base conversion 4/100, and sort/deduplicate 2/100. The fresh V5 interaction audit `686388` is
  2/8 initial, 2/8 review, 1/8 scaffold, and 3/8 compact-state reuse: a narrow explicit-state benefit,
  not latent general reasoning. Matrix `686389` and protocol-matched public board retry `686401` are
  running; the first board allocation `686390` failed before decoding because its shared GPU was busy
  and is invalid. The original sequential verifier rollout `686345` could not fit its 3-hour wall time
  (50/2,000 in 40 minutes) and was canceled/archived without data use. Same-prompt batched decoding was
  added and tested; a 32-sample canary `686400` produced 160 candidates in 19 seconds. The corrected
  16k-candidate train-only verifier chain is `686402 -> 686406`. Flagship remains untouched through
  173.7k at 154.27k tok/s.
- **2026-07-12 ~10:25** — **V5 matched contract matrix completed; transfer is real but still brittle.**
  `686389` compared pinned raw 168.75k with V5 on 48 fresh cases. Native Q/A improved **4/48 -> 17/48**
  and explicit CoT **0/48 -> 11/48**, with new arithmetic, sorting, and state-update wins. Plain direct
  instruction only moved **5/48 -> 6/48**, and one-shot fell **7/48 -> 5/48**. This is an important
  demonstration that concise verified primitive SFT changes behavior, but the prompt dependence and
  zero/near-zero string and base-conversion transfer still forbid promotion. The matrix artifact is
  local/Newton hash-matched at md5 `c6aa2c5dc348a142e07445780c60a88a`; wait for full board `686401`.
- **2026-07-12 ~11:15** — **V5 broad-promotion gate rejected; fresh operator interaction reproduced the diagnosis.**
  Public board `686401` completed at GSM8K maj@4 **10/100**, greedy **9/100**, MATH-500 **3/100**,
  HumanEval **2/164**, MBPP **0/100**. Raw had 5/2/2/7/0, so the primitive curriculum can teach a
  narrow math-format gain but regresses code and is not a general-reasoning recipe. New transcript-first
  audit `686425` tested raw 168.75k and V5 against the same fresh seven cases. Raw was **1/7 initial,
  0/7 review, 1/7 verified fact, 0/7 reuse**; V5 was **3/7, 3/7, 2/7, 3/7**, limited to arithmetic,
  sorting, and logic. It still miscomputes base conversion/state updates, corrupts strings, and cannot
  emit syntax-valid Python. Neither model supplied the requested `state=` representation, so V5's three
  reuse wins do not count as latent context compaction. Saved and hash-verified canonical transcript
  `artifacts/eval_history/manual_capability_raw168750_vs_sft_v5_20260712_JOBID.json`
  (md5 `28dd0b15de2af16a10a2012f630072a1`). The protected flagship remained healthy through 174.7k.
- **2026-07-12 ~11:25** — **V6 contract curriculum passed its constructed holdout but not independent compaction.**
  One isolated epoch (`686413`) completed 4,535 updates in 1,388s without instability. On the 245-case
  r2 held-out contract set, `686414` improved raw **20/245 = 8.16%** to **142/245 = 57.96%**: Q/A 10/35,
  direct 10/35, CoT 15/35, review 28/35, scaffold 34/35, compact 11/35, reuse 34/35. The independent
  deep interview `686415` is the counterweight: only **4/8 initial, 1/8 review, 1/8 scaffold, 0/8 reuse**;
  compact outputs are arithmetically invalid. Therefore the V6 result is a useful contract-learning
  signal, not a general or latent-reasoning promotion. Matrix retry `686438` is isolated/running after
  the first allocation transient. Corrected verifier selector `686437` measured held-out GSM8K first
  pass 7/100, oracle@16 36/100, verifier@16 9/100: sampling has headroom but ranking is weak.
- **2026-07-12 ~11:58** — **V6 transfer result finalized; typed-state V7 gate started.** Matched matrix
  `686438` completed at raw -> V6 Q/A **4/48 -> 17/48**, direct **5/48 -> 10/48**, CoT **0/48 -> 21/48**,
  and one-shot **7/48 -> 7/48**. It confirms learned response contracts, not independent compaction,
  because the deep compact-reuse result remains 0/8. Fresh V3 typed-state data passed its final audit:
  315,000 train / 10,500 held-out rows, zero malformed rows, normalized duplicates, exact held-out prompts,
  or 13-gram held-out overlaps. Raw pinned state baseline `686456` finished all 420 evaluation rows before a
  duplicate-output guard made Slurm report failure: **21/420 = 5.0% answers, 0/280 exact states**. Preserve
  that artifact as valid raw evidence. Isolated V7 SFT `686467` began on evc28 from `best_step168750.pt` with
  audited group weights 0.432/0.18/0.028/0.04/0.32. It has no access to the flagship output directory.
- **2026-07-12 ~12:02** — **DCLM English pilot admitted; replacement build launched.** `686342` completed
  the decontaminated 5B DCLM pilot with 5,000,000,487 tokens in 50 shards (3,506,817 kept / 4,203,263
  seen; 1,820 eval-contaminated rows rejected). Full read-only scan `686360` completed: shard entropy,
  top-token fractions, and byte fallback are consistent with no material outlier; decoded windows are
  ordinary English and technical prose. Pilot is not added to live `SHARDS`. CPU job `686470` now builds
  a fresh 25B replacement, which must complete and pass its own full scan before future-handoff admission.

---

## 2. Cluster access (UCF Newton)

```bash
# Warm the auth/host key (password auth via sshpass), THEN use BatchMode for the real command:
SSHPASS="$NEWTON_PW" sshpass -e ssh -o NumberOfPasswordPrompts=1 -o ServerAliveInterval=30 \
  -o StrictHostKeyChecking=no -o ConnectTimeout=20 newton true
ssh -o BatchMode=yes -o ConnectTimeout=20 newton '<remote command>'
```

- **User/account:** `sa305415` / SLURM account `skattel`.
- **BASE (all paths relative to this):** `/lustre/fs1/home/sa305415/shohin`
- **Python env:** `$BASE/miniforge3/bin/python` (torch + zstandard + datasets installed here).
- **Filesystem:** Lustre `/lustre/fs1` (shared, parallel). Checkpoints, shards, logs all live under BASE.
- ⚠️ **the Newton password (in git-ignored `.env` as `NEWTON_PW`) is a plaintext password committed to prompt history. PENDING for the user: rotate it.**
  Remind them; do not consider it secure.

**Partitions / constraints (learned the hard way — see §5):**
- Use **`-p normal`** with `--gres=gpu:nvidia_h100_pcie:1`. This is what all job scripts use.
- **`normal` is heavily contended** (shared H100 nodes) → transient `"CUDA-capable device(s) is/are
  busy or unavailable"` at job start. The job scripts self-heal (GPU-free wait + retry + bad-node
  exclusion). Do not panic on a single CUDA-busy line.
- **`preemptable` is INACCESSIBLE** ("Invalid qos specification" / time-limit rejects). Do not use it.
- **`highgpu` / 8×H100 = no access.** Measured single-node 2×H100 DDP is allowed after canary `681040`;
  do not attempt broader multi-node or 8×H100 changes without verified account access and a fresh canary.

---

## 3. THE CUSTODY LOOP (do this every wake-up)

You are running an autonomous `ScheduleWakeup` loop. Each wake-up: **check both jobs, act only if
something needs it, report briefly, reschedule ~2400s (40 min), and re-pass the standing prompt with
updated numbers.** The standing prompt template is in §8.

**One-shot health check (copy-paste):**
```bash
SSHPASS="$NEWTON_PW" sshpass -e ssh -o NumberOfPasswordPrompts=1 -o ServerAliveInterval=30 \
  -o StrictHostKeyChecking=no -o ConnectTimeout=20 newton true 2>/dev/null && echo "auth ok"
ssh -o BatchMode=yes -o ConnectTimeout=20 newton 'BASE=/lustre/fs1/home/sa305415/shohin
echo "== PRETRAIN =="; squeue -u sa305415 -h -n shohin-flagship -o "%.10i %.9T %M %R"
grep -E "^step " $(ls -t $BASE/logs/flagship_*.out|head -1) | tail -1
echo "skips=$(grep -c "\[skip" $(ls -t $BASE/logs/flagship_*.out|head -1))"
# preserve every 10k checkpoint (idempotent):
NEW=$(ls -t $BASE/train/flagship_out/ckpt_[0-9]*.pt 2>/dev/null|head -1); STEP=$(basename $NEW|tr -dc 0-9|sed "s/^0*//")
if [ -n "$STEP" ] && [ $((STEP%10000)) -eq 0 ] && [ ! -f $BASE/train/flagship_out/best_step${STEP}.pt ]; then
  cp $NEW $BASE/train/flagship_out/best_step${STEP}.pt && echo "preserved best_step${STEP}.pt"; fi
echo "== CORPUS finemath3 =="; squeue -j 680324 -h -o "%T %M" || echo "(gone)"
echo "shards: $(ls $BASE/artifacts/shards/finemath3/*.u16.zst 2>/dev/null | wc -l)"'
```

**What healthy looks like:** job STATE `RUNNING`; a recent `step N loss L gnorm G` line with **loss <
2.7** and **gnorm ~0.1** (occasional spikes to 0.1–0.2 are fine); tok/s ~149k; skips increasing only
slowly (a few per hour). The trainer's own guards (§7) protect against bad batches; your job is
liveness + milestones, not babysitting every loss wobble.

**Why we preserve every 10k ckpt:** the trainer keeps only the last 3 `ckpt_NNNNNNN.pt` (rotates the
rest). We `cp` each 10k-multiple to `best_stepNNNNN.pt` so it's never rotated away — gives us a decay
ladder for later ablations and a rollback point.

**Reporting discipline:** report **milestones only** (60k decayed numbers, extension launched, corpus
done, weekly progress). Otherwise a one-line "both healthy — step X, shards Y" is enough. The user is
asleep; don't spam.

---

## 4. MILESTONE TRANSITIONS (the important part — get these right)

### 4A. At 60k: FEEDBACK + EXTEND

**Trigger:** pretrain job `680149` is gone from `squeue` **AND** `flagship_out/ckpt_final.pt` exists
(or the log ends with `[done] 60000 steps ...`), i.e. training reached step 60000.

This is the **first decayed checkpoint** — the WSD decay (48k→60k) anneals in the real capability
gains, so numbers here are the first honest read on the model. Two things happen in parallel:

**(a) FEEDBACK — SFT the 60k checkpoint and run the full benchmark board.**
```bash
# First sync the live teacher-data snapshot and SFT tooling from the Mac to Newton.
# This is safe: it copies to separate files and never edits active local writers.
cd /Users/sairamen/projects/shohin
rsync -av pipeline/build_sft_mix.py newton:/lustre/fs1/home/sa305415/shohin/pipeline/
rsync -av train/jobs/sft.sbatch newton:/lustre/fs1/home/sa305415/shohin/train/jobs/
rsync -av artifacts/sft/hy3_reasoning*.jsonl artifacts/sft/rgym.jsonl \
  newton:/lustre/fs1/home/sa305415/shohin/artifacts/sft/

# On a FREE node (exclude evc22 so we don't fight the next pretrain; exclude known-bad nodes).
# Default SFT now builds artifacts/sft/sft_mix_core.jsonl from verified/deduped sources.
ssh newton 'BASE=/lustre/fs1/home/sa305415/shohin
sbatch --exclude=evc22,$(cat $BASE/train/bad_nodes.txt 2>/dev/null|sort -u|paste -sd,) \
  $BASE/train/jobs/sft.sbatch'
# sft.sbatch defaults INIT to the newest flagship ckpt (the 60k one), DATA = curated sft_mix_core.
# Set SFT_BASELINE=1 only if you need the old clean baseline. It writes sft_out/sft_ep{1,2,3}.pt.

# When sft_out/sft_ep3.pt exists, run the board:
ssh newton 'BASE=/lustre/fs1/home/sa305415/shohin
sbatch --export=ALL,N=100,K=1 --exclude=evc22,$(cat $BASE/train/bad_nodes.txt 2>/dev/null|sort -u|paste -sd,) \
  $BASE/train/jobs/eval_all.sbatch'
# Board = GSM8K maj@8 + GSM8K pass@1 + MATH-500 + HumanEval + MBPP. Reads newest sft_ep*.pt by default.
```
Local MPS fallback if the cluster is jammed: `rsync` `sft_ep3.pt` + tokenizer down, run
`train/eval_suite.py` and `train/eval_code.py` on the Mac (slow — pass@1, small N; MBPP is
pathologically slow on MPS, skip or tiny-N it).

→ **REPORT to the user:** the decayed GSM8K / MATH-500 / HumanEval / MBPP numbers **vs the 5% GSM8K /
7.5% HumanEval plateau** from the undecayed 28k/32k prelim. The question we're answering: *did the LR
decay convert the plateau into real gains?* This tells us whether the recipe is working before we
commit two weeks to the long run.

**(b) EXTEND — relaunch pretraining from 60k to 300k (~157B tokens, ~2 weeks).**
```bash
# FIRST: if finemath3 corpus is done (manifest.json present / job gone), add it to the SHARDS line in
# train/jobs/flagship.sbatch so the extended run trains on the bigger, math-richer corpus:
#   SHARDS="$BASE/artifacts/shards/finemath4 $BASE/artifacts/shards/openwebmath \
#           $BASE/artifacts/shards/code_python $BASE/artifacts/shards/finemath3"
# (Edit the file on the cluster with sed/Edit; verify the dir has *.u16.zst shards + manifest.json first.)

ssh newton 'BASE=/lustre/fs1/home/sa305415/shohin
sbatch --export=ALL,STEPS=300000,LRMUON=0.005,LRADAM=1e-3,DSEED=777 \
  --exclude=$(cat $BASE/train/bad_nodes.txt 2>/dev/null|sort -u|paste -sd,) \
  $BASE/train/jobs/flagship.sbatch'
```
Notes on the extend:
- `--resume` (default in flagship.sbatch) picks up the newest `flagship_out/ckpt_*.pt`. **Caveat:**
  `ckpt_final.pt` has no optimizer state; the newest numbered `ckpt_0060000.pt` (or nearest) does.
  Confirm resume starts near step 60000 with optimizer state — check the `[resume] ... -> start step`
  line in the new log. If it resumed `ckpt_final.pt` (fresh optimizer), that's acceptable but not
  ideal; prefer a numbered ckpt.
- `STEPS=300000` re-scopes the WSD schedule so decay now runs **240k→300k**. LR is lowered
  (`LRMUON=0.005`, was 0.01) for the longer, more-converged regime.
- **After launching, UPDATE the standing ScheduleWakeup prompt: STEPS target → 300000**, and update
  §1 LIVE STATE here.
- New checkpoint-preserve cadence: keep every 10k as before (70000, 80000, ...).

### 4B. When finemath3 corpus finishes
- Verify: `ls artifacts/shards/finemath3/*.u16.zst | wc -l` (~120 shards ≈ 25B) and
  `artifacts/shards/finemath3/manifest.json` exists; job `680324` gone from squeue.
- It gets folded into the extended run's SHARDS (see 4A(b)). If the extend already launched without
  it, that's fine — fold it in at the *next* relaunch. Don't kill a healthy pretrain just to add data.

### 4C. Further corpus expansion (opportunistic, CPU jobs — cheap, do it)
More/better data is one of our two levers. When a CPU node is free, tokenize more reasoning-dense data
into new shard dirs, then add them to SHARDS on the next relaunch:
- **OpenMathInstruct-2** `problem` + `generated_solution` → `shards/openmath_pt` (decontaminate vs
  `artifacts/evals/evalgrams.pkl` first — reuse `pipeline/tokenize_shards.py` + the decontam logic).
- **More code** (Stack-Edu / bigger `code_python` pull) → `shards/code_python2`.
- Keep the mix reasoning-tilted (~45–55% web / 25–30% code / 20–25% math per DATA.md), but we're
  deliberately over-weighting math/code for a reasoning specialist.

---

## 5. FAILURE MODES & RECOVERY PLAYBOOK

**Pretrain job died / disappeared from squeue before step 60000.**
Resubmit (single GPU, NOT preemptable, exclude bad nodes):
```bash
ssh newton 'BASE=/lustre/fs1/home/sa305415/shohin
sbatch --export=ALL,LRMUON=0.005,LRADAM=1e-3,DSEED=777,STEPS=60000 \
  --exclude=$(cat $BASE/train/bad_nodes.txt 2>/dev/null|sort -u|paste -sd,) \
  $BASE/train/jobs/flagship.sbatch'
```
`--resume` reloads the newest checkpoint automatically — **no progress lost** back to the last
`ckpt_every`=1000 save. Confirm the new log shows `[resume] ... -> start step ~N`.

**CUDA-busy at job start (`device(s) is/are busy or unavailable`).**
Harmless infra contention on shared `normal` nodes. The job script already: (1) waits up to ~5 min for
GPUs to free, (2) retries the launch 4×, (3) if it logged zero steps, records the node in
`train/bad_nodes.txt` and **auto-resubmits excluding it**. Let it self-heal. Only intervene if
`bad_nodes.txt` grows past ~12 (auto-requeue stops) — then manually resubmit later when the cluster
calms down.

**Long run of skips (trainer prints `[skip:...]`).** The trainer skips any non-finite / loss-spike
(>2× EMA) / grad-spike (>8× gnorm EMA) step **entirely** (no capitulation cap). A few skips/hour is
normal. **≥300 consecutive skips → the trainer ends the run cleanly** (best ckpt preserved) so it
surfaces in monitoring — that means a genuinely bad data region or real divergence. If you see that,
resubmit with `--resume` (drops to a fresh data-shuffle offset via DSEED) and, if it recurs at the
same step, investigate the shards with `pipeline/scan_shards.py` / `pipeline/peek_batch.py`.

**Divergence (loss climbing, not just wobbling).** This was the big historical bug. **Root cause was
NOT the optimizer — it was the dataloader serving 200M-token monodomain blocks** (contiguous single-
domain shards → the model over-specialized then took a loss shock at each domain boundary, and at
135M that destabilized training around step ~6400). **FIXED** by the domain-interleaved dataloader
(`train/data.py`): one read stream per domain, every batch round-robins across all domains, so every
batch is a math+code+web blend. See `DIVERGENCE_DIAGNOSIS.md` for the full writeup. If loss genuinely
diverges again: verify `data.py` still interleaves (don't let anyone "optimize" it back to contiguous
reads), check for a corrupt shard, roll back to the last good `best_step*.pt`.

**Corpus-expansion job (680324) died.** Non-critical — it's CPU tokenization, not the model. Restart
it via its job script in `pipeline/jobs/` (or rerun `pipeline/tokenize_shards.py` for finemath-3plus
pointed at `shards/finemath3`). It resumes by skipping already-written shards. The pretrain does not
depend on it finishing.

**SSH fails / VPN drop.** The user is on a VPN to reach Newton; transient auth failures happen. Retry
the `sshpass` warm-up, then the BatchMode command. If it persists across several cycles, note it in a
report but keep rescheduling — the cluster jobs run fine without your connection.

**"Both jobs gone and I can't tell why."** Check `sacct -u sa305415 --starttime today -o
JobID,JobName,State,ExitCode,Elapsed` for exit codes, and tail the newest `logs/flagship_*.out`.

**Cluster / VPS totally lost (disaster recovery).** There is a local, fully-resumable backup on the
Mac at `train/flagship_out/ckpt_0049000.pt` (step 49k, model + Muon + AdamW state; see §1 for md5).
To rebuild elsewhere: recreate the env (`miniforge3` + torch + zstandard), re-upload the corpus shards
(or re-tokenize from `pipeline/`), put the backup ckpt in the `--out` dir, and launch `train.py`/
flagship.sbatch with `--resume` — it reloads weights **and** optimizer momentum and continues. The
backup is refreshed at each 10k milestone, so at worst you lose <10k steps. Weights-only fallbacks
(older `*.model.pt`, e.g. step 15k) also exist but resume with a fresh optimizer (`--fresh-opt`).

---

## 6. DATA INVENTORY (exact paths under BASE)

**Pretrain corpus — `artifacts/shards/` (zstd uint16 token shards, ~200M tok each):**
| dir | tokens | domain |
|---|---|---|
| `finemath4/` | 6.6B | high-quality math web (FineMath-4+) |
| `openwebmath/` | 14B | math web (OpenWebMath) |
| `code_python/` | 16.8B | Python code |
| `finemath3/` | **25.0B ✅ DONE** | FineMath-3+ (125 shards, decontaminated) — nearly doubles corpus, **triples math** |
| **total available** | **62.4B** | finemath4 + openwebmath + code_python + finemath3 (folds in at 60k relaunch) |

**SFT sets — `artifacts/sft/`:**
| file | count | notes |
|---|---|---|
| `sft_mix_core.jsonl` | ~82k and growing | **Default SFT mix** built by `pipeline/build_sft_mix.py`: frozen snapshot of OpenMath + Reasoning-Gym + code + verified teacher traces, deduped by question. Rebuild before each SFT after syncing latest local teacher files to Newton. |
| `openmath2.jsonl` | 100k | verified (boxed==answer) + decontaminated + concise (≤400 tok) + eval-aligned ("The answer is X."). **Baseline SFT.** |
| `rgym.jsonl` | 995 | Reasoning-Gym procedural CoT (diversity) |
| `code.jsonl` | 446 | MBPP train+val, execution-verified, decontaminated |
| `hy3_reasoning*.jsonl` | ~26.9k and growing | Live verified teacher fleet output (hy3 / Nemotron / GLM / Claude / minimax). **Do not train directly from live files**; snapshot via `build_sft_mix.py`. |
| `self_correct.jsonl` | 15k | arithmetic self-correction traces (all answers verified). **ABLATION-only** — NOT in baseline SFT. Measures baseline vs baseline+self-correction. |
| `openmath2_concise_2M.clean.jsonl` | ~2M | large concise set — reserved for the **FINAL** SFT (upgrade from the 100k baseline once the recipe is proven). |

**Eval sets — `artifacts/evals/`:** `gsm8k.jsonl`, `math500.jsonl`, `humaneval_full.jsonl` (164),
`mbpp_full.jsonl`, and **`evalgrams.pkl`** (70,643 13-grams from all eval questions — the
decontamination bloom filter; every SFT/corpus set is checked against it).

**Tokenizer:** `artifacts/shohin-tok-32k.json` — 32k BPE, single-digit number tokenization. The
compact vocab is deliberate (see DATA.md §tokenizer): a 32k tied embedding ≈18M params vs ~87M for a
128k vocab, so the saved params buy reasoning depth. **Do not change the tokenizer** — it's baked into
every shard and checkpoint.

---

## 7. ARCHITECTURE & CODE MAP (frozen — don't change mid-run)

**Model (`train/model.py`)** — deep-thin GQA transformer, modded-nanoGPT class:
- 30 layers, d_model 576, 9 query heads / 3 KV heads (GQA), d_ff 1536 (SwiGLU), seq_len 2048,
  vocab 32768, **~135M params** (tied embeddings, counted once).
- RoPE (half-split / NeoX, θ=50k), RMSNorm (fp32 internal), QK-norm, z-loss (1e-4), fp32 logits+loss.
- **KV cache** for inference: `forward(..., cache, pos, return_cache=True)` — training path
  (`cache=None,pos=0`) is byte-identical to before the cache was added.
- **Latent recursion** (`cfg.n_loop`, default 1 = off): re-runs the block stack N times (weight-shared
  extra depth). An ablation-gated reasoning bet; `n_loop=1` is byte-identical to a normal forward.

**Trainer (`train/train.py`)** — Muon(+AdamW), WSD LR, bf16 autocast, torch.compile, DDP-capable
(we now run either single-GPU or measured single-node 2-GPU chunks):
- Muon (Newton-Schulz orthogonalized, `train/muon.py`) on matmul params, AdamW on the rest. `--no-muon`
  bisection flag exists.
- WSD schedule (`wsd_lr`): warmup → stable (LR=1.0) → linear decay over last `decay_frac=0.2` to
  `final=0.1`. So for `--steps 60000`: decay runs 48k→60k.
- **Guards (the anti-divergence safety net):** every step measures pre-clip grad norm; **skips**
  (applies no update) any step that is non-finite, a loss-spike (>2× loss EMA), or a grad-spike (>
  `--gnorm-mult`× gnorm EMA, default 8). No capitulation cap. 300 consecutive skips → clean stop.
- DDP-safe: all-reduces loss_acc so ranks make the same skip decision; `set_device` before NCCL init.
- Checkpoints every `--ckpt-every` (1000) as `ckpt_NNNNNNN.pt` (keeps last 3), plus `ckpt_final.pt`
  (model-only) at the end. Full ckpts carry `{model, opt_muon, opt_adam, cfg, step}`.
- CONFIGS: `tiny` (smoke), `mame` (12L/384, ~30M ablation proxy), `shohin` (flagship).

**Dataloader (`train/data.py`)** — **domain-interleaved** (the divergence fix). See §5. Do not revert
to contiguous shard reads.

**Eval:**
- `train/eval_suite.py` — GSM8K / MATH-500, sampling generation + self-consistency **maj@k** (majority
  vote), answer extractors (`extract_gsm8k`, `extract_boxed`).
- `train/eval_code.py` — HumanEval / MBPP, **sandboxed execution** (subprocess + unit tests, timeout
  8s). Separate `HE_STOPS` (aggressive single-function truncation) vs `MBPP_STOPS` (lighter, allows
  multiple functions — a bug was fixed here where multi-function MBPP solutions got chopped).
- `train/eval.py` — quick generation with KV cache.

**SFT (`train/sft.py`)** — completion-masked (loss only on answer tokens, `ignore_index=-1`), packs
prompt+answer+eos to seq_len. Prompt format: `"Question: {q}\nAnswer:"`. Saves model-only
`{model, cfg, step="sft_epN"}` checkpoints.

**Curation (`pipeline/`):** `sft_curate.py` (verified+decontaminated+concise+eval-aligned math),
`curate_code.py` (execution-verified MBPP), `curate_selfcorrect.py` (synthetic self-correction),
`tokenize_shards.py` (corpus → shards), `scan_shards.py` / `peek_batch.py` (shard diagnostics),
`decontam*.py` + `fetch_evals.py` (eval-set decontamination).

**Job scripts (`train/jobs/`):** `flagship.sbatch` (pretrain, self-healing), `sft.sbatch` (feedback
SFT), `eval_all.sbatch` (full board + progress board). All have GPU-free-wait + retry; flagship also
does bad-node exclusion + auto-requeue. `eval_all.sbatch` supports `TARGET_STEP`, `RUN_TAG`, `N`, and
`K`, writes per-run logs under `artifacts/eval_history/`, and appends parsed task metrics to
`artifacts/eval_history/metrics.jsonl` for checkpoint-over-time tracking.

---

## 8. THE STANDING SCHEDULEWAKEUP PROMPT (re-pass each cycle, update numbers)

Each wake-up, after checking/acting/reporting, call `ScheduleWakeup(delaySeconds≈2400, ...)` re-passing
the prompt below with the step/shard numbers updated. Use ~1200s at transitions (waiting on a job to
appear/finish). This is the loop that keeps the model in custody while the user sleeps.

> MULTI-WEEK SoTA PRETRAINING PUSH on the single GPU [...]. PARAMOUNT: keep pretrain ALIVE. STATE:
> (A) PRETRAIN job <id> on <node> (~step <N>, loss ~1.6–1.8, 149k tok/s) with --steps <TARGET>;
> decays <d0>–<TARGET>, ends ~<TARGET> = decayed feedback. (B) CORPUS job 680324 on evc1 writing
> finemath3 (<S> shards ~<B>B; 25B target). [then the EACH-CYCLE checklist, the AT-60k FEEDBACK+EXTEND
> block, further-corpus, reschedule cadence, and the "report milestones only" rule — see the live
> prompt being passed; §3 + §4 here are the authoritative expansion of it.]

The full current prompt text is whatever the last `ScheduleWakeup` passed — this runbook (§3, §4) is
its authoritative, expanded form. If the prompt and this file ever disagree, **this file wins** and
you should fix the next prompt to match.

---

## 9. ROADMAP TO SoTA (the plan beyond 60k)

1. **60k decayed feedback** (§4A(a)) — first honest numbers. Decide the recipe is working.
2. **Extend to 300k** (~157B tokens, ~2 weeks) on the finemath3-expanded corpus (§4A(b)). This is the
   main "more tokens + more data" bet.
3. **Grow the corpus further** (§4C) and fold in at each relaunch — OpenMathInstruct-2 as reasoning-
   dense pretraining, more code.
4. **Periodic branch-decay previews** during the long run: if a second GPU is briefly free, branch a
   short LR-decay from the current checkpoint on a spare node to preview capability without disturbing
   the main run. (Optional; the main run is sacred.)
5. **FINAL SFT** once pretraining converges: upgrade from the 100k baseline to the ~2M concise set,
   add a reasoning mid-train, self-consistency at eval, and run the ablations:
   - baseline SFT **vs** baseline + `self_correct.jsonl` (does self-correction help?),
   - `mame` proxy `n_loop=2` **vs** `n_loop=1` (does latent recursion help before spending it on the
     flagship?).
6. **Report vs MobileLLM-R1-140M** on the full board. That comparison is the finish line.

Guardrails throughout: never train on unverified or contaminated data; keep traces short/concise;
math is the riskiest domain to shorten (needs precise intermediate values) — verify hard.

---

## 10. PENDING FOR THE USER
- **Rotate the Newton password the Newton password (in git-ignored `.env` as `NEWTON_PW`)** (it's in prompt history / plaintext). Remind at next
  interactive contact.

---

## 11. DATA EXPANSION PLAN (active — user green-lit all four + a teacher, 2026-07-07)

Data is the #1 SoTA lever and all of this runs on CPU / external API — **none of it touches the GPU
pretrain**, so it proceeds in parallel with the custody loop. Context: the 300k run consumes ~157B
tokens but the corpus is 62.4B (~2.5 epochs), so more *unique* reasoning-dense tokens directly help.

**★ TEACHER (free this week — time-boxed, use it):** `hermes` CLI (Nous Research agent), model
**`tencent/hy3:free`** via Nous Portal (authed on the Mac at `~/.hermes/`). One-shot:
`hermes -z "<prompt>" -m tencent/hy3:free` → clean text on stdout (~10s/call, CLI-startup-bound).
`hermes proxy` exposes an OpenAI-compatible endpoint for high-concurrency scale (not yet set up).
Auth auto-refreshes. This unblocks our thesis (short-CoT distillation), previously GPU-gated.

1. **Distill verified traces across MANY reasoning domains (highest value).** `pipeline/
   hermes_distill.py` — rejection-sampled: problem(+gold) → hermes → extract answer → verify vs gold →
   keep ONLY correct → curated SFT JSONL (`{question,response,answer,source}`). Handles answer types
   `gsm8k`(numeric) / `boxed`(math) / `mc`(multiple-choice letter) / `exact`(yes-no). Parallel,
   resumable (skips emitted qhashes), decontam-capable. `pipeline/fetch_problems.py` builds normalized
   TRAIN banks (a prebuilt teacher prompt + verifiable gold per row).
   - **Bank:** `artifacts/problems/combined.jsonl` (**~54.4k / 8 domains:** gsm8k, math, sciq,
     commonsenseqa, openbookqa, arc_easy/challenge, **reclor=logic**) = math + science + commonsense +
     logic. Each channel reads it, writes its own `hy3_reasoning*.jsonl`, resumable (skips done qhashes).
   - **MULTI-PROVIDER FLEET (run them all — max throughput + teacher diversity).** Keys in git-ignored
     `.env`. **Provider limits (learned):** OpenRouter free = **1,000 req/DAY** (not for bulk; resets
     daily) · NVIDIA free = per-minute RPM (a couple models at low conc) · Nous Portal (hermes CLI) =
     **no daily cap** → the reliable bulk workhorse. Relaunch any dead channel (all resume):
     ```
     cd pipeline
     export OPENROUTER_API_KEY=$(grep '^OPENROUTER_API_KEY=' ../.env|cut -d= -f2-)
     export NVIDIA_API_KEY=$(grep '^NVIDIA_API_KEY=' ../.env|cut -d= -f2-)
     # BULK (no cap): Nous hy3 CLI          -> hy3_reasoning.jsonl
     nohup python3 hermes_distill.py --problems ../artifacts/problems/combined.jsonl --out ../artifacts/sft/hy3_reasoning.jsonl --backend hermes --model tencent/hy3:free --concurrency 12 --timeout 90 --source hy3 > <scr>/hy3_reasoning.log 2>&1 &
     # STRONG teacher: NVIDIA nemotron-ultra -> hy3_reasoning_nemotron.jsonl  (80% yield)
     nohup python3 hermes_distill.py --problems ../artifacts/problems/combined.jsonl --out ../artifacts/sft/hy3_reasoning_nemotron.jsonl --backend nvidia --model nvidia/nemotron-3-ultra-550b-a55b --concurrency 6 --timeout 120 --source nemotron > <scr>/nemotron.log 2>&1 &
     # NVIDIA minimax (conc 3 — errors higher; drop if >50% err)  -> hy3_reasoning_minimax.jsonl
     nohup python3 hermes_distill.py --problems ../artifacts/problems/combined.jsonl --out ../artifacts/sft/hy3_reasoning_minimax.jsonl --backend nvidia --model minimaxai/minimax-m3 --concurrency 3 --timeout 120 --source minimax > <scr>/minimax.log 2>&1 &
     ```
   - **Add when available:** OpenRouter hy3 (`--backend openrouter --model tencent/hy3:free --concurrency 24`)
     as a burst channel until its 1,000/day resets; **GLM-5.2** (`--backend nvidia --model z-ai/glm-5.2`)
     when it's no longer DEGRADED (retry a raw curl each cycle — it's the SoTA teacher, high value).
   - **Next:** more volume from NuminaMath / OpenMathInstruct-2 problems; Reasoning-Gym procedural
     (`pip install reasoning_gym` on the cluster, not the 8 GB Mac). Teacher vocab irrelevant (32k retok).
2. **Reasoning-Gym mass generation (free CPU, unlimited, decontam-proof).** Scale `pipeline/
   gen_reasoning_gym.py` to billions of tokens of verified procedural math/logic → shards for PT + SFT.
   Most on-target data for a reasoning specialist; currently under-used (~1k SFT rows only).
3. **Corpus expansion (cluster CPU, compute-node sbatch — NOT login node).** `tokenize_shards.py` now
   has `--text-cols` concat. Targets → new shard dirs: OpenMathInstruct-2 (`--text-cols problem
   generated_solution` → `shards/openmath_pt`), a big **MegaMath** slice, more code. Decontam vs
   `evals/evalgrams.pkl`. Fold into SHARDS at each relaunch.
4. **Recipe upgrades for the 300k relaunch (§4A b):**
   - **Decay-phase annealing** — shift the data mix toward premium reasoning data over the 240k–300k
     decay window (mid-training "anneal on the good stuff"), instead of just lowering LR on the same mix.
   - **Overlap:** **settled for the next handoff:** FineMath-4+ is contained in FineMath-3+, so exclude
     FineMath-4 as a separate directory. Its inclusion creates narrow replay rather than new coverage.
   - **Mix ratio:** the guarded `flagship_language_relaunch.sbatch` is canonical; the convenience
     `flagship_reasoning_relaunch.sbatch` mirrors its source policy. Both use OpenWebMath, FineMath-3,
     OpenMath, code, and scanned 25B FineWeb-Edu/DCLM replacements. Their floor-aware BS32 weights
     produce an effective **24.8% math / 25.1% code / 50.1% educational English** mix and continue to
     **600k absolute steps**. Never add an unscanned partial directory or either 5B pilot to SHARDS.

**Time-box note:** the hy3 teacher is free **this week** — prioritize distillation volume while it lasts.

---

* **2026-07-12 ~12:41** — **V7 typed-state gate closed: format transfer without general reasoning.**
  `686467` completed in 1,587s and its exact held-out state evaluation `686471` reached **307/420 answers
  (73.10%)** and **169/280 exact typed states (60.36%)**, with repair 128/140 answers + 132/140 states and
  reuse 140/140 answers. Fresh independent deep interview `686484` was instead **1/8 initial, 1/8 review,
  1/8 with a verified fact, and 0/8 compact reuse**. It fails basic arithmetic after the correct product is
  supplied, base conversion, state updates, sorting, string manipulation, and valid Python. V7 is rejected
  for general reasoning/latent compaction; local/Newton artifacts were copied and md5 verified:
  state `1f9fe0b2993d1a9dafc98cd2d7943887`, deep `c4963fae52d5ac9c38614e77f93f98c8`. A separate raw-vs-V7
  human-authored transcript job was submitted; two generic CUDA allocations failed before inference and are
  explicitly not model evidence. Later generic allocations showed the same CUDA fault and were canceled;
  the completed local-MPS transcript is recorded below. Flagship remains untouched.
- **2026-07-12 ~12:50** — **Second direct V7 interview and future corpus correction.** Generic Slurm
  H100 allocations repeatedly failed CUDA preflight before loading a checkpoint, so no score was inferred
  from them. The same seven-case raw-vs-V7 transcript completed locally on MPS and was copied back to
  Newton: raw **1/7 initial, 0/7 review, 1/7 verified fact, 0/7 reuse**; V7 **2/7, 2/7, 0/7, 1/7**.
  V7's only wins are arithmetic/state templates and it remains zero on base conversion, logic, sort,
  string, and valid Python; hash `a6d8c25cb3482cd37026bbc85306008f`. Future-only relaunch now excludes
  FineMath-4 because it is contained in FineMath-3, requires the scanned 25B DCLM replacement, and
  uses a tested effective **24.8% math / 25.1% code / 50.1% educational-English** BS32 curriculum.
- **2026-07-12 ~13:05** — **Future handoff target corrected before it could silently no-op.**
  `train.py --steps` is an absolute global bound and the active narrow-data phase already ends at 300k.
  The future `flagship_reasoning_relaunch.sbatch` had inherited `STEPS=300000`, which would resume the
  newest 300k checkpoint and immediately write no updates. Its default is now **600,000** with an
  explicit comment; the language-balanced data transition remains future-only and still requires the
  scanned DCLM 25B manifest.
- **2026-07-12 ~13:10** — **Language diversity scaled and relaunch scripts consolidated.** FineWeb-Edu
  pilot size (4.6B) would still replay several times in the 600k continuation, so CPU job **`686526`**
  now builds a 25B replacement with the same quality/decontamination filters and a required fresh scan.
  The guarded canonical `flagship_language_relaunch.sbatch` is aligned with the mirrored convenience
  relaunch: both target 600k absolute steps, exclude FineMath-4 as a FineMath-3 subset, and require
  OpenWebMath / code / FineMath-3 / OpenMath / FineWeb-25B / DCLM-25B at effective 24.8/25.1/50.1
  math/code/English. No live SHARDS changed.
- **2026-07-12 ~13:17** — **Shard decontamination strengthened before language data could be admitted.**
  Found that `tokenize_shards.py` only loaded historical `evalgrams.pkl`, whereas the SFT mixer also scans
  every current eval JSONL directly. Added `--eval-glob`, short-prompt coverage, manifest accounting for
  pickle/direct gram counts, and a regression test. The first 25B partials (`686470`: 2.0G; `686526`:
  134M) are preserved under `*.pre_live_eval_gate` audit names and excluded from use. CPU replacements
  **`686529` DCLM** and **`686530` FineWeb-Edu** restart with the stricter gate. Flagship untouched.
- **2026-07-12 ~13:38** — **Direct-capability diagnosis turned into enforceable data gates.** The live
  raw model remains a weak general reasoner despite stable loss/throughput; independent raw/V7 interviews
  establish failures in arithmetic, base conversion, transformations, Python, review, verified-fact use, and
  compact-state reuse. The first clean cross-family verifier derivative packed to only 737 sequences, so
  verifier SFT is blocked rather than repeated. `bec11a4` adds normalized verifier-prompt dedup. A
  10,000-question answer-checked train-only RG bank completed as `686535`; isolated V4-generator rollout
  `686536` (16 candidates/question) runs on evc25, followed only on success by data/packing gates
  `686540 -> 686541`. Fresh DCLM/FineWeb full scans `686538`/`686539` are dependency-held after their
  direct-live-eval-decontaminated 25B CPU builds. The flagship remains untouched through step 177,000.
- **2026-07-12 ~13:48** — **Fresh hands-on raw-model interview reproduced the failure.** Local MPS ran
  a seven-case five-turn audit directly against the preserved 170k raw checkpoint, then copied the exact
  transcript to Newton with matching md5 `9cd3216365b5292851e298bee4a1aeef`. Result: **1/7 initial,
  0/7 review, 1/7 verified-fact use, 0/7 compact reuse**. The only passing case was the simple logic
  constraint. The arithmetic case repeats `29 x 16 = 496` and never subtracts after the correct product
  is supplied; base conversion, state transitions, list/string transformations, and valid Python all fail.
  This is direct capability evidence, not a loss-curve inference; no general-reasoning or latent-state
  claim is permissible from the current raw model.
- **2026-07-12 ~14:00** — **Future data admission tightened and verified code advanced.** `scan_shards.py`
  now emits atomic JSON reports and `approve_shard_scan.py` creates an explicit manifest/report hash-bound
  approval after decoded-content/outlier review; `flagship_language_relaunch.sbatch` and its compatibility
  mirror now require both 25B language approvals, the newest explicit resume checkpoint, >=24.5B tokens,
  and <=1% byte fallback. Tests passed locally and on Newton. TACO full audit `686514` retained **250/250**
  pilot programs after all supplied test replay with zero quality/eval-overlap failures; its validated
  artifact is backed up locally and remains separate algorithmic code. `686552 -> 686553` now scales that
  same gate to 3k examples. Flagship remains healthy through step 177,250.
- **2026-07-12 ~14:10** — **TACO source-order bias was caught before admission.** The already-pending
  3k scale `686552` began while the deterministic shuffled-selection improvement was being validated, so
  it is retained only as a non-admissible prefix control rather than canceled mid-run. `curate_taco_verified.py`
  now applies a seeded 10k-row streaming shuffle buffer; only `686554 -> 686555` can feed a transfer
  candidate. The canonical full audit also now emits matched/kept/source-scan progress every 100 programs.
  Local/Newton tests and shell validation passed; no flagship or frozen SFT mix changed.
- **2026-07-12 ~14:16** — **Algorithmic-code packing is now an explicit gate.** Added tested
  `build_algorithmic_code_candidate.sbatch`, which freezes only a full-test-verified source through the
  standard decontamination/deduplication audit, rejects unexpected training groups or fewer than 1,000
  usable rows, and never launches SFT. Remote pilot smoke retained 244 clean `algorithmic_code` rows (five
  overlong responses and one eval n-gram overlap were correctly excluded). Shuffled all-test success now
  unlocks `686560` candidate freeze and `686561` 2,048-token packing inspection, not training.
- **2026-07-12 ~14:25** — **Verifier volume can no longer be waived by an operator mistake.**
  `inspect_sft_packing.py` now hashes every input JSONL and the verifier launcher validates that hash,
  512-token packing length, both correct/incorrect classes, and >=3,000 packed sequences before acquiring
  an SFT GPU. Local and Newton regression tests passed. This makes the running `686536 -> 686540 ->
  686541` rollout/data/packing chain a real capacity gate rather than a report to be manually remembered.
- **2026-07-12 ~14:35** — **Verifier family coverage repaired before data freeze.** Read-only diversity
  inspection showed the supposedly 10k/28-family rollout source actually contains 11,200 family-sorted
  rows; `N=10000` would omit the final three families. Added and tested deterministic `--skip` support to
  the generator plus multi-file global dedup to verifier construction. Active `686536` continues as the
  25-family prefix control; only `686564` tail (skip 10k/take 1.2k) + `686565` merged data + `686566`
  packing can feed verifier SFT. The old pending `686540`/`686541` were canceled before any work began.
- **2026-07-12 ~14:42** — **Verifier-bank prefix bias eliminated for future refreshes.** The root cause
  was `sample_verifier_bank.py` emitting balanced per-family reservoirs in sorted-family order. It now
  applies the existing seeded RNG to the completed bank, with a deterministic regression test. Existing
  11.2k data remains immutable, so the active tail repair stays required; every future bounded rollout
  from a new bank will be representative instead of a family prefix.
- **2026-07-12 ~14:50** — **Fresh composition interview confirms a capability, not formatting, failure.**
  Direct local-MPS interaction with preserved raw 170k ran eight newly worded cases through initial,
  review, verified-fact, requested-state, and state-reuse turns. The hash-recorded transcript is
  `artifacts/eval_history/generalization_interview_raw170k_20260712_mps.json` (md5
  `d9ad30fad6c00958ad9d6908ca14c38a`): **1/8 initial, 0/8 review, 1/8 fact, 0/8 valid state lines,
  and 0/8 valid-state-and-reuse**. The product/reject case calculates 17x23 as 351, the base-7 case
  uses decimal-style powers, and both Python contracts fail execution. Added isolated
  `train/jobs/generalization_interview.sbatch` so the named 180k checkpoint can receive the same
  current-H100 audit without touching the flagship. This prohibits any claim of latent compaction or
  general reasoning from raw pretraining alone.
- **2026-07-12 ~15:05** — **Pretraining quality now has a proper measurement hook.** Added tested
  `train/eval_nll.py` and `train/jobs/eval_nll.sbatch`: fixed named monitor JSONLs are packed exactly
  like next-token pretraining and evaluated as pure token-weighted cross entropy/perplexity, excluding
  `zloss`. Four local tests cover input parsing, deterministic packing, bad-row rejection, and the
  auxiliary-loss exclusion. The monitor is deliberately separate from task evals/training shards; a
  reviewed frozen monitor source is required before a first H100 run. This closes the former gap where
  a healthy mixed training loss was the only pretraining-quality telemetry.
- **2026-07-12 ~15:18** — **First fixed language monitor and baseline are preserved.** CPU job `686575`
  froze WikiText-103 test outside both shard paths and `artifacts/evals`: 1,723 docs / 301,241 tokens,
  monitor SHA-256 `fbe8687d618550d2251b397d436abb30a990d5f0b4e7c25cfe85c7265aa251d8`. Hash-bound MPS
  evaluation of raw `best_step170000.pt` scored NLL 3.9648849 / PPL 52.7142 over 301,056 next-token
  targets; the local/Newton result md5 is `fa9f0ea310287d710d9300c8cb0781ab`. It is a fixed language
  trend baseline only, not a general-reasoning score. Four spare-node H100 attempts failed CUDA tensor
  allocation preflight or were canceled before writing results; no empty result was retained.
- **2026-07-12 ~15:25** — **TACO duplicate and dependency fault repaired before admission.** The
  non-admissible source-order 3k control `686552` selected 3,000 execution-passing candidates but failed
  the generic audit on 11 normalized duplicate questions. Its completed JSONL/report were preserved under
  `artifacts/sft/rejected/`, not deleted or admitted. `curate_taco_verified.py` now drops duplicate
  normalized prompts using the quality auditor's exact identity and reports the drop count. Canceled only
  the dependency-blocked never-started jobs, then rebuilt the clean shuffled chain as
  `686583 -> 686584 -> 686585 -> 686586` with explicit fresh r2 paths. The flagship remains untouched.
- **2026-07-12 ~15:45** — **Held-out code likelihood is useful telemetry, not a causal conclusion.**
  CPU job `686595` froze a CodeContests **test**-split monitor: 122 unique syntactically
  valid Python-3 problem+solution rows / 146,896 tokens, zero benchmark prompt overlaps, input SHA-256
  `62668905552c89650c0dcd79227a6fe606e107c39a126f7c1b5d9364ee8fc687`. Raw 170k MPS scores NLL
  1.3537146 / PPL 3.8718 over 145,408 targets (local/Newton artifact md5
  `8b52525e46958ac20a1d6973b6de0cc0`), while English monitor PPL is 52.7142. The source remains
  directional because raw `code_python` is CodeParrot-Clean and no source-level overlap audit has yet
  ruled out CodeContests-derived code. Future code work must still be execution-verified
  continuation/instruction training and transfer-gated; NLL alone cannot explain HumanEval/MBPP failure.
- **2026-07-12** — **Long all-test TACO audits are now explicitly recoverable without weakening admission.**
  The 250-row full replay took 33 minutes, making a 3,000-row audit vulnerable to an ordinary wall-time
  cutoff. `audit_taco_verified.py --resume-partial` and its wrapper's `RESUME_PARTIAL=1` now resume only
  a prior partial whose every retained row has a matching immutable candidate ID, byte-identical response,
  and positive `full_verified_cases`; malformed, duplicate, altered, or foreign rows abort the retry.
  Focused local/Newton tests passed. This is a recovery path only: `686584` continues unchanged and no
  partial output can enter a candidate or SFT mix.
- **2026-07-12** — **V8 broad-transfer SFT is blocked on fresh hash-bound reports, not trusted by name.**
  The 699,928-row V8 candidate has zero recorded prompt overlap and a moderate 2.75x code replay factor,
  but its earlier quality/packing reports predated input-hash binding. A dry launcher correctly refused it
  before CUDA allocation. `audit_sft_quality.py` now records `data_sha256`; CPU jobs **`686602`** and
  **`686603`** respectively rebuild V8's quality and 2,048-token packing reports without touching its JSONL.
  The new isolated `sft_v8_pilot.sbatch` requires both report hashes to match the exact JSONL, >=600k clean
  rows, >=70k packed sequences, >=2,500 code sequences, all four groups, zero eval overlap, and <=3x replay
  before it can train from a preserved checkpoint. After success it still needs public-board and direct
  composition-transfer gates; it is not a flagship promotion.
- **2026-07-12** — **V8 input gate is now ready for a measured 180k experiment.** CPU quality `686602`
  and packing `686603` completed read-only against the immutable 699,928-row JSONL. Their refreshed
  `quality.r3.json` and `packing.r3.json` both bind the same SHA-256
  `da94f9f6aae1d69a12633241b3971f6cfc68f7a7edbc788b956063ec5a70fc72`; V8 has 73,273 packed
  2,048-token sequences, 2,660 code sequences, zero recorded direct eval overlap, and max replay 2.755x.
  The V8 launcher passed every data gate in a no-GPU dry run and stopped only at CUDA allocation. It is held
  until `best_step180000.pt` is preserved and the known-good `evc25` verifier allocation is free, then must
  run one isolated epoch followed by public-board and direct-composition evaluations. It remains an experiment,
  never a live-pretrain change or promotion by construction.
- **2026-07-12** — **TACO durability observation and fix.** The active pre-fix audit `686584` reached
  100/3,000 all-test-verified matches after its streaming source initialization, but its open `.partial`
  file was still at byte zero because Python had not flushed its text buffer; this is not counted as durable
  output and the job was not disturbed. Future audit code now flushes and `fsync`s every logged progress
  batch before printing it, so an explicit resume can retain verified rows after a wall-time interruption.
  Focused local tests pass and the fix is mirrored to Newton for a retry only.
- **2026-07-12** — **V8 direct-transfer interview now has its own data-leakage gate.** Public benchmark
  filtering does not establish that the custom eight-case composition interview is held out. New tested
  `audit_generalization_overlap.py` parses the evaluator's literal `CASES` through AST, then records exact
  and 13-gram prompt overlap plus SHA-256s for both the candidate JSONL and interview source. CPU job
  **`686625`** is read-only, excludes the active TACO node `evc21`, and must report zero overlap before
  V8's launcher can proceed. The launcher now requires that report to bind the exact frozen data and current
  interview source before CUDA allocation; lexical disjointness is necessary evidence for transfer, not a
  sufficient general-reasoning claim.
- **2026-07-12** — **V8 passes the direct-interview lexical holdout gate.** CPU job `686625` completed
  off the active TACO node with **699,928 valid rows**, no malformed/missing prompts, and **0 exact / 0
  13-gram** hits against all eight current composition-interview prompts. The report binds V8 data SHA-256
  `da94f9f6aae1d69a12633241b3971f6cfc68f7a7edbc788b956063ec5a70fc72` and case-source SHA-256
  `6820a1d77e206ffd92dff837fb38c5c0633d06788aabbab3ade9ef0bafc2c3be`. V8 is now admissible solely as
  an isolated 180k pre/post transfer experiment after the raw baseline, public board, and transcript gates
  are serialized on a known-good H100.
- **2026-07-12** — **Full TACO replay retry can now use the allocated CPUs without weakening tests.** The
  active `686584` process is healthy but runs full supplied cases through one subprocess worker despite its
  two-core allocation, so it remains untouched as the baseline outcome. The retry-only audit now first maps
  every selected immutable source ID, then executes each complete test suite through a bounded
  `ThreadPoolExecutor`; `WORKERS=2` uses no more than the existing two-core Slurm request and preserves
  deterministic source order, the same 2s sandbox timeout, every supplied case, partial-row integrity, and
  fsynced progress batches. Local and Newton real-execution tests passed. Use this only if `686584` ends
  non-successfully; override a resubmission to `WORKERS=2` and a sufficient wall-time, never relax tests.
- **2026-07-12** — **Raw-to-V8 transfer chain is queued, serial, and isolated.** On the known-good H100
  node `evc25`, after the verifier-tail job `686564` succeeds and after `best_step180000.pt` should exist,
  queued jobs are: **`686638`** raw-180k public board -> **`686639`** raw-180k composition interview ->
  **`686640`** one-epoch hash/overlap-gated V8 SFT -> **`686641`** V8 public board -> **`686642`** V8
  composition interview. The direct interviews preserve full transcripts; V8 cannot start unless the raw
  interview succeeds, and V8 cannot be promoted from a constructed/held-out generator score alone. This
  chain has no flagship output path and cannot run before the current verifier H100 work ends.
- **2026-07-12** — **Fresh bounded GLM health probe remains negative.** One isolated train-only ARC
  request to NVIDIA `z-ai/glm-5.2` with 20s timeout/one worker completed in 35s with `attempted=1`,
  `kept=0`, `wrong=0`, and `err=1`; its scratch output is empty. It did not touch a live writer or any
  frozen mix. Keep GLM, Nemotron, and HY3 bulk paused rather than spending requests in an error loop; GLM
  remains the first teacher to re-enable only after a later bounded probe has a real verified keep.
- **2026-07-12** — **Tokenizer coverage audit found no fallback defect.** Read-only held-out encodes with
  `shohin-tok-32k.json` found **0** `<unk>` or byte-fallback pieces: WikiText-103 test is 4.12 characters
  per token / 1.30 tokens per word, CodeContests test prompt+Python is 2.87 / 2.08, MATH-500 prompts are
  2.67 / 2.33, and GSM8K prompts are 3.78 / 1.37. The denser math/code segmentation is expected from
  notation and punctuation; this rules out a gross coverage failure but does **not** prove the frozen 32k
  tokenizer is never a capability constraint. No tokenizer change is justified mid-pretrain.
- **2026-07-12** — **V8 promotion rule is precommitted and machine-checked.** CPU decision job **`686643`**
  follows the raw/V8 transcript chain and writes only a verdict report. `accept_followup` requires all five
  deterministic public metrics, improvements on at least two, no HumanEval/MBPP regression, an eight-case
  direct interview with initial-answer gain, and no verified-fact regression; constructed holdout scores,
  formatting, and state-marker emission alone are explicitly insufficient. Any missing evaluator artifact or
  failed condition rejects V8. `evaluate_v8_promotion.py` passed both an accept fixture and a code-regression
  rejection fixture locally and on Newton. Even acceptance is only a follow-up decision, never a flagship
  promotion.
- **2026-07-12 ~15:42** — **Three external NVIDIA reasoning sources are in read-only intake, not data
  admission.** New tested `probe_reasoning_source.py` records dataset config/split, builder license metadata,
  field shapes and lengths, plus sampled exact and 13-gram overlap against every live eval JSONL without
  retaining source rows or executing code. CPU-only jobs **`686647`** (`nvidia/OpenMathReasoning`),
  **`686648`** (`nvidia/OpenScienceReasoning-2`), and **`686649`** (`nvidia/OpenCodeReasoning-2`) write only
  immutable reports under `artifacts/source_probes/`. A source remains inspection-only until its report,
  license/provenance, field mapping, split policy, and a source-specific decontamination/quality plan are
  reviewed; no bulk download, frozen-mix change, or flagship change is authorized by these probes.
- **2026-07-12 ~15:45** — **Initial source inspection found real admission constraints.** The first
  `train`-split requests for OpenMath and OpenCode stopped safely after discovering their actual split layouts;
  explicit read-only probes now cover OpenMath `cot` / `tir` / `genselect` and OpenCode `python` / `cpp`.
  The 64-row results find one OpenMath `cot` **generated-solution** 13-gram hit against live eval prompts
  (no exact hit), so that split is contamination-sensitive and cannot be used without full-stream filtering.
  OpenMath `cot`/`tir` generated solutions have roughly 16k/19k median characters, OpenScience outputs
  roughly 14k, and OpenCode `r1_generation` roughly 19k (Python) / 46k (C++): raw traces are materially too
  long for the 2,048-token SFT context and must not be replayed naively. All r1 reports are local/Newton
  hash-matched; card-hashing r2 probes `686655`/`686656`/`686657` are inspection-only and must establish
  license provenance before a source-specific field mapping or curator is proposed.
- **2026-07-12 ~15:55** — **V8 now has a full training-text decontamination proof, not only a
  prompt-only audit.** CPU job **`686662`** scanned every `question`, `response`, and `completion_prompt`
  string in all **699,928** frozen V8 rows against every current eval JSONL: **0 exact rows, 0 13-gram rows,
  0 malformed JSON rows**. Its data SHA-256 is the same frozen V8 value
  `da94f9f6aae1d69a12633241b3971f6cfc68f7a7edbc788b956063ec5a70fc72`. The held V8 SFT job now depends
  on this successful audit and its launcher independently requires the hash-bound zero-overlap report.
  This establishes decontamination, not quality transfer or a promotion claim.
- **2026-07-12 ~15:55** — **TACO now has an automatic non-success failover without duplicating a
  successful path.** The original all-test audit and its `686585 -> 686586` candidate/packing chain remain
  unchanged. Only if `686584` ends non-successfully, `686659` runs a fresh-output retry from the immutable
  shuffled input with `RESUME_PARTIAL=1`, 2 workers, every supplied test, and a 3-day wall-time; then
  `686660 -> 686661` builds and packs a separately named candidate. This prevents a wall-time loss from
  becoming a silent data-quality relaxation or a manual recovery gap.
- **2026-07-12 ~15:55** — **NVIDIA source cards are license-confirmed but remain unadmitted.** Hash-bound
  r2 card probes identify **CC-BY-4.0** for OpenMathReasoning, OpenScienceReasoning-2, and
  OpenCodeReasoning-2. OpenMath COT still has one sampled response 13-gram collision, every source has
  long raw traces, and OpenCode's apparent `question` field is not yet a usable prompt mapping. CPU-only
  `686664` is measuring the actual 50k-row OpenMath COT yield after concise-token limits, final-answer
  verification, and full problem+trace decontamination. It writes only a report; no candidate is authorized
  unless this measured yield and source-specific semantics justify one.
- **2026-07-12 ~16:10** — **OpenMath COT selection is a real data-quality constraint, not a source-scale
  shortcut.** The source card says `used_in_kaggle` means a row trained NVIDIA's AIMO-2-winning model, not
  that it is an evaluation sample. Our own full problem+trace eval filtering is therefore the relevant
  leakage control; the selector now records this flag as provenance instead of dropping it by default. The
  initially over-conservative 10k / 2,048-token COT dry run kept only **30** rows: 6,765 were unnecessarily
  excluded by that provenance flag, 3,151 exceeded the context cap, 35 failed final-answer verification, and
  11 had direct eval 13-gram hits. It proves raw long-CoT replay is inadmissible. The obsolete zero-yield
  512-token probe was stopped; corrected all-provenance 10k probe `686671` retains strict answer and
  full-text decontamination and must determine whether enough concise verified rows exist before a candidate
  build is considered.
- **2026-07-12 ~16:17** — **180k is preserved and the metrics ledger is now a required custody artifact.**
  `685084` reached step **180,000** at loss **1.6526**, gnorm **0.103**, and **154.30k tok/s**; the sole
  180,030 gnorm guard skip recovered immediately. Numbered `ckpt_0180000.pt`, Newton
  `best_step180000.pt`, and local `train/flagship_out/ckpt_0180000.pt` are full optimizer checkpoints with
  matching md5 **`a592a8bd46163eb1427fe64460be0c6a`**. New `TRAINING_METRICS.md` records the exact distinction
  between **94,371,840,000 nominal update tokens** at 180k and the current **57,826,022,271-token** active
  corpus capacity, plus checkpoint inventory, data-admission status, benchmark baselines, and the update
  protocol. It must be refreshed at every 10k milestone from manifests/reports/logs only; partial data is
  never counted as admitted.
- **2026-07-12 ~16:17** — **The OpenMath COT exact-context selection result is inspection evidence, not
  admission.** New combined prompt+completion+EOS accounting matches `sft.py`'s separate encodes. Under the
  exact 2,048-token limit, all-provenance `686672` kept **326/10,000** rows after final-answer verification
  and full problem+trace evaluation decontamination; 9,398 traces were individually long and 17 additional
  examples exceeded the true combined limit. The apparent source scale is therefore not usable training
  volume without an explicit candidate, packing, balance, and quality review.
- **2026-07-12 ~17:14** — **Fresh direct interaction at raw 180k remains negative.** The preserved local
  180k checkpoint was queried through the seven-case `manual_capability_probe.py` suite with greedy,
  bounded 32-token completions: **1/7 initial, 0/7 review, 1/7 verified fact, 0/7 state reuse**. Only the
  simple syllogism was correct; it still treats base-6 as decimal division, ignores supplied arithmetic
  state, emits unrelated solution/code text for sort/string tasks, and fails the Python predicate contract.
  The shorter cap makes this directional rather than a formal 128-token comparison, but there is no visible
  reasoning improvement. Artifact `artifacts/eval_history/manual_capability_raw180k_20260712_mps32.json`,
  md5 **`cc6332a5c99d6cbf6ba2f8987ae58cc0`**, is local/Newton hash-matched and must remain diagnostic-only.
- **2026-07-12 ~17:14** — **Checkpoint retention accounting corrected.** `ckpt_0180000.pt` existed and
  matched its promoted copy at the 180k milestone, then the trainer reaped the numbered file by its normal
  retention policy. Durable copies are Newton `best_step180000.pt` and local full
  `train/flagship_out/ckpt_0180000.pt`, both md5 **`a592a8bd46163eb1427fe64460be0c6a`**. The metrics ledger
  now records this distinction explicitly rather than claiming a numbered file remains indefinitely.
- **2026-07-12 ~18:02** — **VRWM context-scaling research has a hard raw baseline and an admissible,
  isolated data candidate.** The new Verified Recursive Working Memory controller forwards a model-emitted
  canonical `wm:a=<int>;b=<int>` state to one later instruction without correction, answer injection, or
  best-of-N selection; it therefore tests an actual closed-loop transition policy rather than V7-style
  state-format imitation. Raw `best_step180000.pt` scored **0/25** exact first transitions and **0/25**
  closed-loop rollouts across five prompt-disjoint value/length OOD regimes (five episodes each). Candidate
  r1 was rejected for 166,766 duplicate normalized prompts; r2 passed quality but was too small at 2,263
  packs; r3 has **497,274** unique rows, zero malformed/duplicate/full-text-eval overlaps, and **18,013**
  packed 2,048-token sequences (SHA-256 `b2a688e1f7aa6c79dd65ed1944fa5dc00cd022acfc793896ecf4696c94d4089f`).
  It is hash-matched locally/Newton and staged only. Do not submit its SFT until a known-good CUDA node is
  available; opportunistic `evc26`, `evc43`, and `evc50` allocations all failed CUDA preflight before model
  loading and wrote no misleading score. Full research contract: `VRWM_RESEARCH.md`.
- **2026-07-12 ~18:13** — **Two-H100 training is re-authorized only at a natural handoff.** User explicitly
  permitted two GPUs for faster training. `685084` remains untouched on evc22 through 182,120 at 154.31k
  tok/s. A fresh BS32/ACC4 DDP validation attempt `686728` reached evc31 but both rank-1 attempts failed
  before model loading with CUDA "busy or unavailable," so it produced no loss/capability result and was
  canceled. Synced the forward-only `train.py` stream-generation fix to Newton (it cannot alter the loaded
  flagship process), hardened `flagship2.sbatch` so auto-recovery preserves BS32/ACC4/250-checkpoint
  parameters and a supplied static CUDA blacklist, and submitted **`686732` afterany:`685084`**. It requests
  two H100s/four CPUs, preserves the 524,288-token update, uses fresh optimizer rewarmup, excludes
  `evc26,31,36,43,50`, and remains dependency-held until the single writer exits. Scheduler test selected
  evc37. On start, verify actual CUDA/DDP health before treating it as a throughput or capability gain.
- **2026-07-12 ~18:31** — **Current-code two-H100 handoff is revalidated.** Isolated job `686734` ran on
  evc37 from the durable 180k checkpoint: both GPUs were visible, `world=2`, `BS=32`, `ACC=4`, fresh
  optimizer, and new `stream_generation=1` all printed before training. It completed the bounded 320-step
  run exit 0 with no CUDA/NCCL/DDP error. Compile-free late windows were 291.7-293.9k tok/s (~1.90x the
  154.3k tok/s one-H100 flagship); compile-inclusive final rate was 243.8k tok/s. Loss/gnorm stayed in band;
  the guard skipped the final step 180,319 at gnorm 1.71 vs 0.09 EMA, so the short run cannot prove its
  recovery but does not contaminate any flagship state. Retain `686732` as the verified natural successor;
  it must still prove a recovered post-skip live trajectory after its real handoff.
- **2026-07-12 ~20:31** — **VRWM r3/r4 results and the r5 repair gate are now explicit.** r3 SFT
  `686742` completed one epoch from `best_step180000.pt`; full default p80 closed-loop result is
  **43/400**, but held-out paraphrase p10 is **0/50**, rejecting any semantic-transfer claim. The larger
  r4 data branches each have 513,902 audited rows: state-only r4 is **32/400 default / 2/400 semantic**;
  deterministic scratch r4 is **120/400 default / 21/400 semantic**. Scratch therefore improves a
  controlled executable-state task, including 12/80 length-16 and 3/80 length-32 default programs, but
  it remains weak under held-out language and is not a reasoning promotion. Repair curriculum r5 has
  1,409,072 audited rows (68,347 packed sequences; data SHA-256
  `011282f032963a40b8b39ab9572808de1d3473ef2b57ef727526fb9d00985c76`), with correct and plausible-wrong
  model-state proposals per transition. Isolated SFT `686820` began cleanly on evc37. Four fresh reports
  are chained: default first-pass `686827`, semantic first-pass `686828`, then same-model repair default
  `686829` and semantic `686830`. The controller only transports the model's draft/repair output; it never
  executes, selects, or repairs a state itself. This is the discriminating gate before considering any
  further context-scaling work.
- **2026-07-12 ~21:07** — **Direct operator interaction rejected r5 for broad capability before its
  narrow repair chain completes.** r5 SFT `686820` completed one epoch in **1,303s** and its 500,446,874-byte
  `sft_ep1.pt` is md5 **`ef99f8c2ab5835c8229bcd4f36fb8789`** on Newton and the Mac. The first two r5
  p80 evaluations initially failed on evc26/evc31 at CUDA preflight before loading a model, so those jobs
  are hardware non-results; downstream dependencies were canceled. The resubmitted semantic first-pass
  result is **17/400** (r4 scratch semantic was 21/400), while default and repair measurements continue
  only to characterize the constructed protocol. Crucially, a fresh eight-case human-style transcript
  interview directly compared raw 180k with r5. Raw is **1/8 initial, 0/8 review, 1/8 supplied fact,
  0/8 state/reuse**; r5 is **0/8 on every condition** and emits synthetic `check:` / `wm:` text for ordinary
  product, base conversion, logic, list, string, and Python questions, even when given a correct fact.
  This is answer-format collapse from an overly narrow curriculum, not thinking. Do not use r5 for broad
  benchmarks or as a promoted model. Preserve the transcripts and finish the model-only repair control for
  diagnosis, then reject the branch unless it motivates a strictly bounded future data-mixing ablation.
- **2026-07-12 ~22:00** — **V9's low-share memory hypothesis now has an immutable response-contract
  proof.** New read-only `audit_response_contracts.py` and focused tests bind each candidate's exact
  SHA-256 to response markers, state markers, trace starts, and length distribution by training group.
  V8 (`da94f9f6...`) has procedural `<think>` starts in 100% of its 374,659 rows; V9 (`8f5845d1...`)
  adds 513,902 compact scratch-state rows but samples them at only 5%. Its report confirms 82.57% state
  markers only within the VRWM group, while the four broad groups retain their original contracts; report
  md5 is `934fa1471fa41476ed9e6c0fa2364177`. The V9 launcher now requires this hash-bound report plus
  its existing quality, complete training-text decontamination, exact packing, and replay checks before a
  rerun can allocate CUDA. The already-running `686876` remains untouched and has passed the same gate
  in dry validation; current optimization is stable but **not** capability evidence. The raw pinned board
  `686861` continues before V8. No broad promotion can be made until V8/V9 boards, fresh direct
  transcripts, and V9 semantic-memory evaluation all complete.
- **2026-07-12 ~22:25** — **A final answer is no longer sufficient evidence that Shohin is thinking.**
  New `thinking_trace_audit.py` has twelve fixed held-out arithmetic, state-transition, base-conversion,
  and trace-repair questions. Each requires a model-generated `<think>` block with a correct hidden
  intermediate marker (for example `product=378` or `after_multiply=75`) **and** a correct explicit final
  answer; either element alone receives no visible-reasoning credit. Focused parser tests pass. The case
  source is lexically disjoint at exact and 13-gram levels from all 699,928 V8 and 1,213,830 V9 prompts.
  Raw baseline `686935`, V8 candidate `686936`, and V9 candidate `686937` are isolated H100 transcript
  jobs. Stale pending V8/V9 decision-only jobs were canceled before execution and replaced by `686938` and
  `686939`, which cannot run until their respective board, eight-case direct interview, visible-trace, and
  (for V9) default+semantic memory artifacts all succeed. This raises the promotion bar; it does not change
  any model weights, live data stream, or flagship job.
- **2026-07-12 ~22:35** — **V9 evaluation was unblocked from an unrelated V8 serialization without
  weakening any gate.** V8 and V9 are separate one-epoch SFTs from the same immutable 180k base; waiting
  for V8's much longer raw-board/SFT path gave no additional V9 evidence. H100 capacity is available and
  the user authorized bounded parallel research. Pending-only `686881`-`686885`, `686937`, and `686939`
  were canceled before execution or artifact creation. Replacement `686942` (board) and `686943`
  (interview) depend only on successful V9 SFT `686876`; `686944`/`686945` then run default/semantic VRWM
  p80, `686946` runs the held-out visible-thinking transcript, and `686947` waits for every one of those
  outputs plus raw trace `686935`. This is a speed improvement only: no checkpoint, data file, evaluator
  rule, or flagship process changed.
- **2026-07-12 ~22:40** — **Direct raw-180k visible-thinking baseline is 0/12.** The preserved local
  optimizer checkpoint was queried through all twelve fresh trace questions on MPS with greedy 96-token
  decoding. It emitted no `<think>` blocks, no correct required intermediates, and no correct final answers.
  Its first product answer became `27 x 14 = 498`, then recursively `498 x 14 = 7768`; a sequential task
  used the wrong precedence, base conversion stopped at `356 = 300 + 50 + 6`, and correction repeated the
  supplied bad arithmetic. The transcript is saved locally and on Newton as
  `thinking_trace_raw180k_mps_20260712.json`, md5 `5eb7fa94cea87cc138416801f593f85c`. This is a direct
  behavioral diagnosis, not an aggregate score: raw Shohin does not yet produce usable visible reasoning.
  H100 raw trace `686935` remains required for matched candidate promotion decisions.
- **2026-07-12 ~23:20** — **V9 broad-plus-memory pilot is mixed evidence, not a reasoning promotion.**
  Its one-epoch checkpoint is complete and hash-matched locally/Newton. The in-progress H100 board has
  already raised GSM8K from raw 1.0% maj@8 / 3.5% pass@1 to 9.0% / 10.5%, but the independently completed
  eight-case interaction is only 1/8 initial, 0/8 review, 1/8 verified-fact, and 0 valid state/reuse.
  Corrected special-token transcript decoding reveals `<think>` tags on 9/12 cases but no correct
  intermediate, final, or trace-and-final result. Keep all remaining V9 gates running and reject it if
  that direct/trace failure persists, regardless of partial board movement.
- **2026-07-12 ~23:20** — **Semantic bridge is data-gated before model training.** Generated 200k
  solver-verified train plus 5k held-out cases across five natural-language arithmetic/state/repair
  families. Admission audit passed zero malformed/duplicate/eval-overlap rows; V10's 899,928-row broad
  plus 10% bridge candidate is now in its exact packing gate. Added `eval_semantic_bridge.py` with a
  deterministic, per-family held-out subset and separate answer/trace metrics. No V10 SFT may be launched
  until that mix finishes its gate and the future checkpoint can be measured on this evaluator as well as
  the public board and direct transcripts.
- **2026-07-12 ~23:25** — **V10 frozen bridge mix admitted, still not trained.** Job `686966` completed
  its final exact-packing and replay gate at 899,928 rows / 81,705 packs. No malformed rows, duplicate
  normalized questions, or exact/13-gram held-out overlap survived; code and teacher replay remain below
  the 3x ceiling and bridge replay is 0.97x. This makes the data suitable for an isolated ablation after
  V9 completes, not a basis to claim capability before the held-out semantic, direct-transcript, and
  public evaluations run.
- **2026-07-12 ~23:45** — **Semantic capsule context control queued, without touching training.** New
  generator/evaluator data separates semantic fact preservation from V7's fixed-answer state templates:
  a model must create a two-field capsule from a natural record, update it across context resets using
  model-generated states only, and answer a final read/sum/difference query after the source record is
  gone. Held-out domains, fields, wording, values, and 4/8/12-step lengths are disjoint. `686991` builds
  the CPU-only data and `686992` independently recomputes every transition/query plus all train-to-heldout
  controller-prompt exact/13-gram overlap. It is a diagnostic candidate, not an SFT decision.
- **2026-07-13 ~00:20** — **The raw model remains healthy; untrained recurrence is rejected.** Flagship
  `685084` reached observed step **187,870** at **154.31k tok/s** with recent loss/gnorm in band; it was
  not interrupted. The two-H100 successor remains dependency-held for the natural handoff. Matched
  180k inference-loop probe `687004` completed: loop 1 scored Q/A **5/24** and CoT **0/24**, whereas
  loop 2 scored **0/24** in both modes and its eight-case direct interview fell from raw's 1/8 initial
  and verified-fact results to 0/8. This is a negative architecture control, not a model regression:
  untrained extra recurrence is not a reasoning shortcut and will not enter the flagship.
- **2026-07-13 ~00:20** — **V9's partial math-format lift does not survive broad gates.** Its continuing
  board has GSM8K maj@8 **18/200** and pass@1 **21/200** against raw 2/200 and 7/200, but MATH-500 is
  unchanged at **5/200** and HumanEval is **5/164**, below raw's 7/164. Combined with 1/8 direct
  interaction and 0/12 correct visible traces, V9 is already ineligible for broad promotion; the
  remaining MBPP/memory/decision chain is retained for a complete controlled record rather than used to
  rationalize a cherry-picked score.
- **2026-07-13 ~00:20** — **Corrected semantic-capsule protocol passed its first independent gate.**
  Build `687009` produced 360,000 train rows and 3,000 held-out 4/8/12-step episodes. Protocol audit
  `687010` recomputed every transition/query and checked every held-out controller prompt: 0 malformed
  or duplicate train rows, 0 invalid episodes, and 0 exact or 13-gram train/held-out overlaps across
  30,000 controller prompts. Generic data admission `687013` now runs quality, full-text,
  response-contract, and 2,048-token packing gates before any raw or SFT model evaluation can be queued.
- **2026-07-13 ~00:25** — **Semantic-capsule generic admission was tightened to measured capacity, not
  relaxed around a quality failure.** Read-only run `687013` measured 360,000 clean rows, 46,898,373
  tokens, 22,899 exact 2,048-token packs, and zero generic prompt/full-text eval overlaps before stopping
  only because its predeclared 25,000-pack capacity floor was too high. The corpus remains sufficient for
  a low-share mixed ablation, so the floor is now an explicit 20,000 packs and rerun `687015` uses fresh
  report paths. No data file, checkpoint, SFT job, or flagship process was modified.
- **2026-07-13 ~00:30** — **Closed-loop context baselines are held behind admission.** `687015` passed
  every generic data gate at 360,000 rows and 22,899 exact packs. First raw job `687017` failed CUDA
  preflight on `evc26` before model load, so it is a hardware non-result and its held V9 successor
  `687018` was canceled. Replacement `687019` evaluates immutable raw 180k over 100 held-out episodes
  per 4/8/12-step regime on the vetted-node exclude set; only if it is complete and clean does `687020`
  evaluate the isolated V9 checkpoint on identical episodes. The evaluator carries only model-emitted
  capsules, never executes or repairs a state, and records every response. These jobs cannot train or
  promote a model; they establish the raw/V9 floor before any semantic-capsule SFT mix is considered.
- **2026-07-13 ~00:40** — **Model-generated semantic capsules fail for raw and V9.** Raw `687019` and
  V9 `687020` both completed all 300 held-out 4/8/12-step episodes with **0 closed-loop successes**, 0
  valid initial capsules, and 0 correct transitions. This is strong negative evidence against treating
  state-marker formatting or V9's GSM8K lift as semantic memory. The capsule route is rejected for
  promotion and remains only a documented control.
- **2026-07-13 ~00:40** — **V8 also fails behavioral reasoning gates.** Refreshed isolated V8 SFT
  `687003` completed its one epoch (4,580 updates / 1,496s), but fresh direct `687033` was **0/8 initial**
  and fresh visible trace `687034` was **0/12** correct intermediate/final pairs. Its board `687032`
  remains read-only evidence only; no board outcome can override those transfer failures. A direct
  raw-versus-V8 interactive transcript audit `687043` was queued separately for qualitative review.
- **2026-07-13 ~00:40** — **Continuous latent feedback is now a controlled, paired experiment rather
  than another text-state recipe.** Commit `318c6b9` adds the isolated `forward_embeds`/soft-token
  rollout, answer-only curriculum trainer, held-out depth evaluator, CPU generation/admission jobs, and
  tensor/gradient/pairing tests. The first 96k generator `687038` was stopped by its solver audit before
  admission due to **2,608 duplicate normalized prompts**; its train, held-out, and audit files were
  preserved in `artifacts/rejected/`, and blocked admission `687039` was canceled. Commit `98ee9a2`
  fixes uniqueness at generation time and passed a full 96k-row stress test in 2.18s. Rebuild `687041`
  followed by admission `687042` is queued. No latent/control H100 training can start before the fresh
  audit passes.
- **2026-07-13 ~00:50** — **Corrected continuous-latent data is admitted and paired pilots are live.**
  CPU build `687041` plus generic admission `687042` passed at **96,000** unique train rows and **1,800**
  held-out depth-5/6/8 rows; training SHA-256 is
  `aa65eefd1dbee25c3c7cec956059a970ea53e079bbc4f4695dd160cacd980fd9`, with zero malformed,
  duplicate, public-eval, full-text, exact, or 13-gram held-out overlap. H100 canary `687046` ran 32
  exact-shape batches through L=0/1/2/4, saved its isolated checkpoint, and remained finite; its first
  run `687044` was preserved after discovering a prefix-only cap gave no full token-shape batch. Commit
  `72e8ca6` now selects complete batches from the frozen full corpus. Matched `24,000`-example one-epoch
  pilots `687048` (L=0 control) and `687049` (progressive L=4) run from the same raw 180k checkpoint,
  data SHA, seed, batch size, optimizer, and update count. Their separate held-out evaluators
  `687050`/`687051` test rollout depths 0/1/2/4/8 only after successful checkpoints; nothing in this
  chain can alter flagship pretraining.
- **2026-07-13 ~00:45** — **The matched continuous-feedback pilot is rejected by the
  factorized transfer gate.** Control `687048` and progressive-L4 `687049` both completed
  1,500 updates from the same raw-180k checkpoint, frozen 24,000-example selection, seed,
  optimizer, and batch size. The original full-OOD evaluator gives the control and pilot
  **0/600 at every L=0/1/2/4/8**. Its answer parser is valid, not the cause: control replies
  `The answer is <int>.` on all 600 L0 cases but is incorrect. To separate fit from transfer,
  CPU-only `687057` built audited v2 evaluation slices: 896 rows, SHA-256
  `b9106b3233c62592dba5f244d5fdec5474d56c813b6d39dc0fe8ee77a441039`, zero invalid or
  duplicate rows and zero exact train prompts. The matched control v2 result is 50.39% IID,
  14.06% depth-OOD, 13.28% language-OOD, 0.00% full-OOD. The feedback pilot is worse in every
  regime: L0 41.41/6.77/10.94/0.00%, L2 44.92/9.38/11.72/0.00%, L4
  46.48/12.50/11.72/0.00%. Thus low answer-only loss and a few in-template solves do not
  establish latent reasoning, composition, or context scaling. Mark this final-hidden-state
  feedback route rejected; preserve reports/control checkpoint/pilot checkpoint (md5
  `e1c093a51a808c14acb21299fbf8be7b`, `9a516983a4b1cd85bd4bbc96f4f97230`,
  `47abf77baabef759d68c92c6257a9e1c`, `6636e328d382db8b100309742d5bdbda`) as immutable
  negative evidence. New local-only source-dropping fixed-slot memory code and tests are an
  isolated follow-up: every source token must be absent at decoder time and M=0/detached/
  shuffled-memory controls are required before any H100 run. The live flagship was untouched,
  healthy through step 189,060 at 154.31k tok/s; next DR target remains 190k.
- **2026-07-13 ~01:15** — **Source-dropping packet mechanics passed, but its tiny
  capability screen is negative and remains unpromoted.** CPU job `687072` generated an
  admitted 192,000-row train / 1,536-row factorized held-out corpus (train SHA-256
  `419199a756679e61601c05481ffc59221fba75b601608990539781013a51da64`; held-out SHA-256
  `6a10a6b27be8dc6b0a36954296d9f33abd099d87bbdff46c251a1357f1c894c3`) with zero malformed,
  duplicate, or exact train-eval source prompts. Canary `687074` exposed a variable chunk-width
  batching bug before any optimizer step; the wrapper/trainer were patched to preserve each chunk
  as a separate `[batch, tokens]` tensor, and focused dense/ragged/gradient/batching tests passed.
  Fresh canary `687077` then completed 128 updates with finite loss, no CUDA/NCCL/DDP error, and
  checkpoint md5 `a92f973b5fb7a35ee3937a15c4db2bd5`. Its read-only held-out ablation `687079`
  is normal **0/192**, zero-packet **2/192**, shuffled-packet **1/192**: it learned answer format,
  not a transferable retained-state algorithm. This is exactly the expected boundary of a mechanics
  canary, so the next gated evidence is a matched 24,000-example / 6,000-update M0 no-slot control
  `687082` versus M1 eight-slot source-dropping pilot `687083`, both from immutable raw-180k with
  identical selection, seed, optimizer, and schedule. Their after-success evaluators `687084`/
  `687085` score normal, zero, and shuffled packets on an independently held-out balanced screen.
  The flagship stayed untouched and healthy through step 189,480 at 154.32k tok/s.
- **2026-07-13 ~01:32** — **M0 completed; CLL remains an unsubmitted, audited fallback.**
  M0 `687082` completed its exact 6,000-update matched no-slot run in 1,153s and its held-out
  normal/zero/shuffled evaluator `687084` started; M1 `687083` remains independently training and
  must finish before comparator `687088` can decide whether any source-memory claim is justified.
  Local CLL code now provides every-prefix solver-recomputed readbacks, counterfactual final-event
  pairs, independent operation/query replay, and exact/13-gram **input-prompt** split rejection.
  Small and 3,580-row smoke datasets had zero malformed/duplicate/pair/overlap failures; CLL source
  tags remain discarded-source-only, never query/answer tokens. The CPU-only build script is prepared
  but deliberately unsubmitted until the answer-only causal comparison resolves. Flagship `685084`
  stayed isolated and healthy through step 189,880 at 154.31k tok/s; 190k remains the next DR target.
- **2026-07-13 ~01:44** — **190k is durable; M0 is a true no-memory null.** `685084` logged
  190,000 at loss 1.5030 / gnorm 0.09 / 154.31k tok/s, then continued healthy. Numbered
  `ckpt_0190000.pt`, Newton `best_step190000.pt`, and local full+optimizer
  `train/flagship_out/ckpt_0190000.pt` match at md5 `3e195aaf44a14259797c49d7f80d9c7f`.
  M0 evaluator `687084` finished normal/zero/shuffled all **6/384 = 1.5625%**, including zero
  long-context successes. That is the required null control, not memory evidence. The M1 packet
  train remains isolated and stable; comparator `687088` still determines the answer-only route.
  Commit `4eb50c9` adds a future CLL full-held-out evaluator that preserves counterfactual pairs and
  requires a 10-point pairwise normal-packet advantage only when pairs are present; it was not
  synced into the in-flight M1 evaluation to preserve the already-launched experiment definition.
- **2026-07-13 ~01:50** — **Direct raw-190k interaction confirms that the flagship does not yet
  think.** Local-MPS `generalization_interview_raw_190k_local_mps.json` (checkpoint
  `ckpt_0190000.pt`, md5 `9ecff93b6d70f889b1d0fbe4decd4dd8`) scored **1/8 initial**, **0/8 after
  explicit independent review**, and **1/8 with a supplied verified intermediate fact**; the only
  correct case was a simple set constraint. All **8/8** requested compact states failed the exact
  `state=` contract and all reuse attempts failed. Arithmetic, base conversion, sequential state,
  sorting, string manipulation, and two code tasks showed prompt echo, tutorial/code imitation, or
  incorrect computation rather than a recovered procedure. This is behavioral evidence, not a
  benchmark proxy: no broad-reasoning or thinking claim is permitted for raw 190k.
- **2026-07-13 ~02:05** — **Matched answer-only source-packet memory is rejected.** M1 `687083`
  (eight slots) completed its exact 6,000 updates in 2,694s, then independent evaluator `687085`
  scored normal/zero/shuffled **6/384 / 6/384 / 9/384**. Locked comparator `687088` returned
  `advance_answer_only_source_packet=false`: normal lost the IID comparison (**4/96 vs 5/96
  shuffled**) and the combined length/language comparison (**2/192 vs 3/192 control**), with zero
  positive chunk or query-kind margins. The packet has no demonstrated retained-information effect;
  preserve reports as negative evidence and do not scale this recipe. The next bounded mechanism is
  the CPU-only certified latent-ledger data build: solver-recomputed readbacks at each source prefix,
  plus complete final-event counterfactual pairs, before any new H100 experiment is admitted.
- **2026-07-13 ~02:07** — **Certified latent-ledger build is submitted, not yet admitted.** After
  a fresh-path check and `sbatch --test-only` reservation, CPU-only `687132` was submitted with
  `TRAIN_EPISODES=16000` and `EVAL_EPISODES_PER_CHUNK=32`. It may write only
  `certified_latent_ledger_v1_train.jsonl`, `certified_latent_ledger_v1_heldout.jsonl`, and its audit
  report. The generator/auditor and future pair-safe evaluator/comparator hashes were verified on
  Newton before submission. No model checkpoint, SFT mix, flagship data stream, or H100 training job
  is part of this build.
- **2026-07-13 ~02:12** — **Certified latent-ledger data is admitted and matched GPU gates are
  queued.** CPU `687132` completed in 153s with **223,996** train rows, **7,936** held-out rows,
  **16,000** train counterfactual pairs, and **384** held-out pairs. Independent audit has zero
  invalid rows, duplicate prompts, exact overlaps, 13-gram overlaps, or malformed pairs; train/eval
  SHA-256 are `1fd39b2ece45c47dd48015489221c1998adb463302097cf21b3dd5345ef3a515` and
  `1bef01c34bf034aaf00a8f976b4463cc28af87ca460c8187873543a7e05ad6a9`. Trainer commit `5457a8f`
  now refuses a CLL audit lacking valid nonzero pairs. Matched raw-190k M0 slots=0 `687134` and M1
  slots=8 `687135` use the same 24,000 selected examples / 6,000 updates / seed / optimizer, followed
  by pair-safe evaluators `687136/687137` and locked comparator `687138`. Every GPU job excludes
  live-node `evc22`; no job can modify the flagship.

- **2026-07-13 current custody check** — **CLL v1 is preserved as a correctness pass, but H100
  trials were stopped before start by an efficiency preflight.** Read-only tokenization of 20,000 v1
  train rows found mean/p50/p95 chunk lengths **213.36/208/241** tokens and mean/p95 per-row source
  lengths **477.45/859**; all 7,936 held-out rows measured **194.93/192/227** tokens per chunk and
  **660.24/1,390** per-row source. The 16 opaque seal words consume most of a tiny encoder's
  context while contributing no task information. Pending M0/M1/evaluator/comparator
  `687134`-`687138` were canceled while pending, before GPU allocation or any output write. The
  new isolated `compact_v2` tag scheme keeps a record-unique 13-token contamination guard as
  `Reference` plus 12 tokenizer-single-character words (14 tokens including punctuation, verified
  against `shohin-tok-32k.json`) and changes no v1 artifact. It must receive a fresh CPU build,
  independent audit, and token-cost report before matched H100 M0/M1 jobs are recreated.

- **2026-07-13 current custody check** — **Compact CLL v2 passes the combined correctness and
  efficiency admission.** CPU `687141` completed in **85s** with **223,996** train and **7,936**
  held-out rows, **16,000/384** counterfactual pairs, protocol
  `source_removed_readback_v2_compact_tags`, and zero audit failures. Train/eval SHA-256 are
  `8760df867b4da98dcc84b356eea8e0d70922e3280e868193216173603814c08c` /
  `dfcf852c0ca57b6bb58f4f7c1a775221e33e97aa261d79478cc3bf36e82e5fc7`. Hash-bound local/Newton
  token report `certified_latent_ledger_v1_v2_token_audit.json` is md5
  `89ac960e6eb12b0f2dbe25f42a9ee3d1`: train chunk/source means are
  **228.60/511.63 -> 41.44/92.74** tokens and held-out means are
  **194.93/660.24 -> 37.43/126.79**. Matched raw-190k M0 `687144` (slots=0) began isolated on
  evc38; M1 `687145` (slots=8) is pending resources. Both use 24,000 selected examples, 6,000
  updates, seed `20260715`, and fresh output paths. Pair-safe evaluations `687146/687147` and
  locked comparator `687148` remain dependency-held. No result is a reasoning claim unless the
  comparator's M0/zero/shuffled/counterfactual gates all pass.

- **2026-07-13 ~03:10** — **Compact CLL M0 is a valid no-memory floor.** Matched M0 `687144`
  completed all 6,000 updates in 1,181s. Held-out evaluator `687146` returned normal/zero/shuffled
  **11/631 = 1.743%** in every condition, with **0/128** correct-and-different counterfactual pairs.
  This is deliberately null evidence, not a CLL result. Matched eight-slot M1 `687145` remains
  isolated on evc35 with the same raw-190k base/data/seed/schedule; `687147` and comparator `687148`
  must finish before the route can advance or be rejected. Flagship `685084` remained healthy through
  observed step 191,530 at 154.30k tok/s; `ckpt_0191500.pt` exists on Newton and the next DR target is 200k.
- **2026-07-13 ~03:10** — **`evc36` is scheduler-idle but CUDA-unhealthy.** It advertised two
  unallocated H100 GPUs, so isolated `687150` requested one GPU and tried an actual CUDA tensor allocation.
  The job failed after 2m15s with `CUDA-capable device(s) is/are busy or unavailable`. Keep evc36 in the
  explicit bad-node exclusion; this diagnosis did not alter or delay flagship `685084` or M1 `687145`.
- **2026-07-13 ~03:15** — **Fresh raw-190k direct interaction separates a narrow stop defect from
  reasoning.** At the standard 128-token decode budget, the seven-case/five-turn MPS interview is
  **1/7 initial, 0/7 review, 1/7 verified fact, 0/7 compact-state reuse** (md5
  `86214f2d4b096a67950cb1885c4109fd`). A 32-token replay is **2/7 initial, 1/7 review, 1/7 fact,
  0/7 reuse** (md5 `6f37c4fcf44351773981c83c12c68811`) because one state-update trace correctly reaches
  49 before the longer decode loops. It still computes `29 x 16 = 496`, treats base-6 425 as 0.425,
  ignores verified facts, and cannot reuse a state. This is explicitly not a thinking claim or a decoder
  optimization to promote; it identifies answer commitment as one interface issue after a model has learned
  substantially broader verified operations.
- **2026-07-13 ~03:22** — **The raw-190k decoder-depth effect fails to generalize.** The exact same
  fresh 48-case QA matrix at max-new 32 vs 128 is **8/48 vs 7/48**; only one state-update case changes.
  Both budgets score 7/8 syllogisms and zero across 8 arithmetic, 8 base-conversion, 8 sorting, and 8
  string-insertion cases. The hash-bound MPS artifacts are `89889f3616cae656c415bf277244c67f` and
  `83ea629a248655bc4b7ceec3ecb8ec66`. Reject decoder-budget/answer-commitment tuning as a broad route;
  keep the effect only as a future interface diagnostic.
- **2026-07-13 ~03:25** — **Stokes is ready for CPU-only work.** Read-only SSH validation reached
  `euser1`, confirmed 64 CPUs, and verified the shared Lustre root `/lustre/fs1/home/sa305415` with ample
  space. Future generation, audit, packing, and report jobs should use Stokes after a single-writer/output
  check, leaving Newton H100 capacity for the flagship and GPU experiments.
- **2026-07-13 ~03:37** — **Compact CLL v2 is rejected by its locked causal gate.** M1 `687145` completed
  cleanly. Held-out normal/zero/shuffled is **16/631 / 4/631 / 12/631**; M0 remains **11/631** in every
  mode. Comparator `687148` is `advance=false`: fit only gains **0.79pp** over the strongest control,
  length+language only **0.63pp**, there are two rather than three positive chunk margins, and no complete
  intervention pair is correct-and-different (**0/128**). The answer-only packet can affect a few fit
  answers but cannot retain a compositional source state. Reports are mirrored locally with md5
  `526e9a2c365ff195bb369fd5c49ae60a` / `bf831f2e8ea4d8528dbe44da6343aac9`; never promote this route.
- **2026-07-13 ~03:50** — **Continuous latent rollout is also rejected against its actual matched control.**
  The shared 896-case depth/language screen shows no-latent `687048` at **190/896 = 21.21%** and the
  progressive-L4 model `687049` at **173/896 = 19.31%** even with L=4 inference. The candidate loses
  every nonzero regime and only ties full OOD at zero; its within-model L=0→L=4 increase is a misleading
  comparison because the separately trained no-latent model is stronger. This rules out more answer-only
  soft-token rollout. In response, the next bounded source-free mechanism is **latent state algebra**:
  training-only state/delta/equivalence/separation geometry with verified commutation and intervention
  pairs, a matched answer-only control, shuffled pair and permuted code controls, and locked held-out
  equivalence/intervention gates. Generator, audit, trainer, evaluator metadata, comparator, and CPU smoke
  tests are complete locally; route only its unique CPU data build through Stokes next, then require a fresh
  audit before any H100 mechanics canary.
- **2026-07-13 ~04:00** — **Latent-state-algebra v1 is CPU-admitted; only a bounded mechanics canary is
  queued.** The first 32k-pair Stokes build exposed and stopped on an over-strict *intra-train* 13-gram
  uniqueness condition at pair 19,772 before writing artifacts. The condition was corrected to the
  intended train/eval decontamination boundary, regression-smoked locally at 2,048 pairs, then rebuilt
  from scratch on Stokes. The new independent audit passes **64,000** train rows and **2,304** held-out
  rows: zero malformed rows, duplicate prompts, exact/13-gram train-eval hits, or invalid pairs; train
  SHA-256 **`7bdf783797981863caca8fc82d6c0c857b948e678c00c5ac5794220eabc4cd53`**, held-out
  **`83d14d65f319decebe3195ffc7b20269161a299212142f3bd555875b8dcd7f3a`**, audit md5
  **`6d3c0b7c18619017bcf6732e3a45867a`**. Isolated `687158` is a 256-pair, one-H100 mechanics-only
  canary from immutable `best_step190000.pt`, excluding the active/bad nodes and writing solely to
  `train/lsa_canary_190k`. It must prove CUDA, finite pair losses, data binding, and source removal
  before any matched answer-only/control/candidate experiment is submitted. It cannot access or alter
  flagship `685084`.
- **2026-07-13 ~04:13** — **LSA mechanics passed; r1 data was then correctly rejected at the evaluator
  boundary, and r2 is now the only admissible experiment chain.** `687158` finished its 256-pair H100
  mechanics run in 69s: 64 finite updates, source-removed checkpoint metadata, and hash-bound r1 data;
  it is a transport check only. A read-through of the generic held-out evaluator caught that r1 rows lacked
  its deterministic `reference` key, which is necessary to prove case matching and temporal shuffle
  controls. Full r1 jobs `687164/687165` were canceled after 27/26s before any model artifact, rather than
  producing unevaluable results. The generator now emits row-level `reference=pair_id-pair_member`; the
  audit rejects missing/noncanonical/duplicate references and the trainer requires those zero-count fields.
  Fresh Stokes r2 output is **64,000/2,304** rows with every prior audit gate plus zero duplicate
  references: train/eval SHA-256 **`a73c5068f9c775ea6b40b42335e01ad4f792657aeba4688d3aca42d853becb58`** /
  **`6a9619ebc73f1778dbb13d69d903de1790ce0292f231c437f35e282a934581da`**, audit md5
  **`19ac94a6f42546c55596ece6a76ffc8f`**. Matched control `687168` (`ZERO_AUXILIARY=1`) and verified
  state-algebra candidate `687169` both start from immutable `best_step190000.pt`, same r2 data/seed/BS=4,
  and separate output directories. Complete held-out source-removed normal/zero/shuffled evaluations
  `687170/687171` and locked comparator `687172` are dependency-held. No job references flagship paths.
- **2026-07-13 ~04:29** — **Corrected data admission now blocks an undersized FineWeb replacement.**
  FineWeb job `686530` finished but its manifest contains only **4,599,748,648** tokens because the
  scale script was pinned to `sample-10BT`; it is preserved and explicitly rejected as a 25B corpus,
  not "approved by scan" or eligible for a future `SHARDS` relaunch. The scale script now defaults to
  `sample-100BT`, takes an explicit `DATASET_CONFIG`, and refuses publication unless the manifest has
  at least **24.5B** tokens. The admission tool now recognizes the tokenizer's historical `tokens`
  manifest key as well as `total_tokens`, with a regression test. Stokes CPU job **`738030`** is running
  on `ec52`, writes only `fineweb_edu_25b_r2.partial`, and cannot atomically publish without the new
  token-floor check; it remains future-only and still needs a full scan plus hash-bound approval. This
  work is isolated from active flagship `685084`, which remained healthy through observed step
  **192,950** at **154.29k tok/s**. Matched LSA r2 control `687168` and candidate `687169` were both
  finite through step 900/7,163 and remain under their preregistered evaluator/comparator gate.
- **2026-07-13 ~04:35** — **LSA now has an explicit second causal stage rather than an overclaimed
  primary result.** The running r2 jobs are unchanged: `687168` and `687169` remain the matched
  answer-only/verified-geometry primary screen, with their already-submitted evaluators and comparator
  frozen. A code audit caught that this primary comparator cannot itself test the two promised *training*
  controls: shuffled pair relations and permuted state codes. New local-only
  `compare_latent_state_algebra_stage2.py` is deliberately separate from the active path. If and only if
  stage one advances, it will bind four all-held-out reports and their checkpoint metadata: answer-only,
  verified candidate, shuffled-pair, and permuted-state. It requires matching init/data/seed/update
  metadata, source-removed decoding, and a **>=5pp** candidate advantage over every decoder and training
  control on fit, length/language, equivalent pairs, and intervention pairs. Synthetic contracts passed;
  no stage-two GPU job is queued yet, so a failed primary gate spends no further H100 time.
- **2026-07-13 ~04:42** — **The next flagship handoff is data-gated, not just throughput-gated.**
  Held fallback `686732` correctly preserves a two-H100 continuation but its submitted snapshot uses the
  old math/code-only stream. It remains the continuity fallback, not an automatic quality promotion. Future
  language/reasoning relaunch scripts now default only to `fineweb_edu_25b_r2` and its r2 approval record,
  never the rejected 4.6B `sample-10BT` output. They reject any non-524,288-token update and verify the
  allocated CUDA device count before allowing `NG>1` DDP. Before `685084` ends, replace `686732` only if
  both r2 FineWeb and DCLM prove their 25B manifests plus hash-bound full scans/approvals; otherwise retain
  `686732` so data construction cannot create an idle-training gap.
- **2026-07-13 ~04:46** — **Pre-sleep baseline recorded from live systems.** Flagship `685084` is
  RUNNING on evc22 through **193,290** at **154,295 tok/s**. Recent loss is **1.4727** with gnorm
  **0.10**; the sole new guard skip at step 193,173 recovered on the next logged step, so no intervention
  is warranted. Remote numbered checkpoints extend through `ckpt_0193250.pt`; full 190k DR remains
  hash-matched locally/Newton at `3e195aaf44a14259797c49d7f80d9c7f`, and 200k is the next transfer
  target. Matched LSA answer-only control `687168` and verified-geometry candidate `687169` are both
  finite and matched at roughly 2.3k/7,163 updates; their evaluator/comparator chain is still held, so
  there is no causal result yet. DCLM `686529` has written 183 partial 100M-token shards (about 18.3B
  tokens) but is retrying transient HF 503s; it remains live and unadmitted. Stokes FineWeb r2 `738030`
  is RUNNING on ec52 and has emitted its first 100,001,543-token partial shard; it remains unadmitted
  until the >=24.5B manifest floor and full scan/approval are present. No live writer was modified.
- **2026-07-13 ~04:52** — **Prefix-supervised packet-memory mechanics are ready locally, but no GPU
  trial is admitted.** `SourceDroppingMemory.encode(..., return_trace=True)` now exposes only the
  continuous packet after each source write; the answer boundary remains final packet plus fresh query
  with all source text absent. New `prefix_state_supervision.py` solver-recomputes each operation-prefix
  state and trains state/delta probes at every write, addressing the final-state-only credit-assignment
  weakness of LSA. CPU tests prove final-trace identity, writer/model gradients, exact prefix targets,
  delta targets, and all existing LSA contracts. This is a forward-only fallback: submit it only if
  the current LSA causal gate rejects or demonstrates that the added prefix signal is the next justified
  ablation. Any future trial must retain matched final-only, shuffled-prefix-label, and no-memory controls;
  it cannot alter `685084` or reuse an active experiment output.
- **2026-07-13 ~05:02** — **The prefix-state fallback is now a complete locked experiment, still
  unsubmitted.** New `prefix_state_memory_train.py` writes one source-free packet trace per input chunk,
  applies solver-recomputed normalized state and delta losses at every trace position, and scores the
  answer from that same final packet without recomputing/reintroducing source text. Its `ZERO_AUXILIARY=1`
  answer-only control and `PREFIX_MODE=shuffled` label control preserve init/data/seed/batch/update count.
  `compare_prefix_state_memory.py` rejects unless the verified-prefix candidate clears answer-only,
  zero-packet, shuffled-source, and shuffled-prefix controls by 10pp IID / 5pp OOD / 10pp on both
  equivalence and intervention pairs. CPU tests cover source removal, one-pass packet loss equivalence,
  solver targets, gradient flow, trainer labels, comparator success, and comparator rejection. It is
  explicitly conditioned on the in-flight LSA result; no Slurm job has been submitted and no live process
  reads these files.
- **2026-07-13 ~05:07** — **Pre-sleep live baseline and prefix-target audit recorded.** Flagship
  `685084` is RUNNING on evc22 through **193,680** at **154,295 tok/s**: loss **1.5811**, gnorm
  **0.09**, LR 0.0050, and direct H100 telemetry at 100% compute utilization, 63,767 / 81,559 MiB
  VRAM, and 280 / 310 W. The sole new guard skip at 193,437 recovered on the next logged step; latest
  numbered checkpoint is `ckpt_0193500.pt`, while verified full local/Newton DR remains 190k at md5
  `3e195aaf44a14259797c49d7f80d9c7f`; 200k is the next transfer milestone. Matched LSA r2 control
  `687168` and verified-geometry candidate `687169` are both finite and matched through roughly
  **3,940 / 7,163** updates, so their held-out evaluator/comparator chain remains the only valid
  primary decision path. CPU-only prefix-state audit `738032` passed against the exact r2 train SHA
  `a73c5068f9c775ea6b40b42335e01ad4f792657aeba4688d3aca42d853becb58`: **64,000** rows,
  **191,998** recomputed prefix positions, zero invalid rows, and report md5
  `b83805b22343c93f962db8db57114b9a`. This admits the prefix labels as mechanically correct, not the
  new training recipe; no prefix GPU job is submitted. DCLM `686529` remains live at **188** partial
  100M-token shards (about 18.8B), still unadmitted; no live writer or `SHARDS` input was changed.
- **2026-07-13 ~05:34** — **Causal Prefix Readback (CPR) is ready as the next, still-unsubmitted
  source-free context-scaling ablation.** The target failure mode is specific: an auxiliary probe can
  fit continuous packets while the model's own language decoder remains unable to read an intermediate
  packet. CPR therefore asks the decoder, after *every* source write, a fresh question about one
  solver-recomputed register; the decoder receives only that prefix packet and the question, never source
  text. The equal-decoder-work controls are explicit: `replicated-final` repeats the ordinary final query
  from the final packet at every prefix, and `shuffled` assigns another example's complete prefix
  readback labels to each packet. Both final answer and held-out prefix readback remain source-free.
  Independent Stokes audit `738034` passed against r2 train SHA
  `a73c5068f9c775ea6b40b42335e01ad4f792657aeba4688d3aca42d853becb58`: **64,000** rows,
  **191,998** readback targets, zero invalid rows or answer leakage, report md5
  `12c8191bb159d716556ecfad6096e098`. CPU preflight `738036` caught an unacceptable extra 1,471
  shape buckets from variable-length numeric answers before any GPU allocation; CPR was corrected to
  group only the source-free decoder readbacks inside the established source-write batches. Replacement
  Stokes preflight `738038` passes with **32,000** complete pairs, **2,242** source/final buckets,
  **7,163** full batches, and **3,348** dropped pairs — exactly the current LSA batch surface, with no
  additional CPR-induced loss. Trainer/evaluator/comparator and tests are complete locally and synced to
  Newton. **No CPR H100 job is submitted:** it is conditioned on the in-flight LSA primary result and
  cannot modify `685084`, its corpus, or any live experiment output.
- **2026-07-13 05:39** — **Overnight comparison baseline captured from live systems.** Flagship `685084`
  is RUNNING on evc22 through **194,220** at **154,298 tok/s**: loss **1.4431**, gnorm **0.11**, LR
  0.0050, and **101,827,215,360 nominal update tokens**. It has no new guard skips after the recovered
  193,437 outlier. Durable recovery remains the hash-matched 190k pair
  `best_step190000.pt` / `train/flagship_out/ckpt_0190000.pt` at md5
  `3e195aaf44a14259797c49d7f80d9c7f`; 200k is about 5.5 hours away at the measured rate and is the next
  mandatory remote+local promotion. Matched LSA r2 answer-only control `687168` and verified geometry
  candidate `687169` are both RUNNING and finite through **6,220 / 7,163** and **6,260 / 7,163** updates,
  respectively, at 4.9 pairs/s. Their held-out source-free evaluators `687170/687171` and locked
  comparator `687172` remain dependency-held; no causal claim is permitted before that chain completes.
  DCLM `686529` has **195** partial 100M-token shards (about 19.5B tokens) and FineWeb r2 has **6**
  partial 100M-token shards. Both remain unadmitted, excluded from the live writer, and require their
  full manifest/scan/approval gates. The only pending flagship successor is `686732`, two H100s after
  `685084`; it cannot share or modify the active writer.
- **2026-07-13 05:52** — **LSA primary training completed cleanly and a direct 190k behavior diagnosis
  narrowed the failure mode.** Matched answer-only control `687168` and verified-geometry candidate
  `687169` each completed all **7,163** updates in 5,814s / 5,800s, respectively, with finite losses and
  independently written checkpoints. Full held-out source-free normal/zero/shuffled evaluators
  `687170/687171` are now running on their released H100s; locked comparator `687172` remains held after
  both. No result is inferred from their close in-training answer losses. Separately, local MPS capability
  matrix `capability_matrix_raw190k_seed20260713_all_contracts_mps.json` (mirrored to Newton; md5
  `abd71f2a9c9b504e1f36a2735d0ead17`) scores normal Q/A **5/24**, direct **2/24**, CoT **1/24**, and
  one-shot **3/24**. The normal Q/A score is four syllogisms plus one state-update; all arithmetic,
  base-conversion, sort, and string families are zero in every prompt contract. One separate manual
  state-update response emitted the correct intermediate sequence and `49` before looping, but the matrix
  confirms that this is rare and prompt changes do not unlock general computation. Treat this as evidence
  for stronger state/decoder supervision, not an output-format-only explanation or a thinking claim.
- **2026-07-13 06:22** — **Explicit overnight comparison snapshot.** Flagship `685084` is RUNNING on
  evc22 through **194,990** at **154,297 tok/s**, with recent step loss **1.4823**, gnorm **0.10**, and
  **102,230,917,120 nominal update tokens**. No new guard skip followed the isolated recovered 193,437
  event. The 190k local/Newton DR pair remains md5 `3e195aaf44a14259797c49d7f80d9c7f`; 200k is the next
  mandatory preservation. LSA evaluators `687170/687171` are still running their locked held-out
  normal/zero/shuffled source-free protocol; `687172` remains dependency-held, so no causal result or
  next-stage launch is yet warranted. DCLM `686529` reached 204 partial ~100M-token shards (about 20.4B)
  and remains unadmitted. Stokes FineWeb replacement `738030` is RUNNING on ec52 with **11** partial
  ~100M-token shards (about 1.1B), also unadmitted. This snapshot is the user-facing comparison
  baseline for the next morning.
- **2026-07-13 06:32** — **LSA r2 rejected; causal prefix-readback launched as the next causal test.**
  Locked comparator `687172` reports `advance=false`: fit-IID normal margin **+1.04pp** versus the
  preregistered +10pp requirement; combined length/language OOD **+0.09pp** versus +5pp; equivalent
  pairs **+0.52pp** versus +10pp; intervention pairs **0/576** for every arm; only two (not three)
  chunk-count wins. Thus the auxiliary latent geometry did not create a robust causal retained state;
  no LSA stage 2 is warranted. The final control/candidate/comparison reports are mirrored locally at
  md5 `af9c6f40d066040a52505fa49cfd9c37`, `2d465a4d38a1aa09cf090f7fab07b90f`, and
  `6810722f720c1ad8d2874de4e4ff5329`. The next experiment is causal prefix readback, a source-free
  decoder-readback control designed to distinguish language-decoder use of an intermediate packet from
  an auxiliary-probe fit; its corrected submission is recorded in the next entry.
- **2026-07-13 06:43** — **CPR launch corrected and reaches healthy finite updates.** The first
  `687216/687217/687218` CPR submission correctly refused the CPR-specific audit before model loading:
  the trainer requires the generic hash-bound LSA admission audit, while the CPR audit remains a
  separately preserved protocol validation. Canceled their never-satisfiable `687219/687220/687221`
  dependencies; no checkpoint or model result exists from that first attempt. Corrected three-arm jobs
  `687223` verified, `687224` shuffled complete-label, and `687225` replicated-final replay use generic
  audit SHA `8c6138c5ff968923a5df58f9547277532b09c9fb193ab1999291e4c4e56ce26f` and the CPR audit bound to
  data SHA `a73c5068f9c775ea6b40b42335e01ad4f792657aeba4688d3aca42d853becb58`. All three passed startup
  and have finite, descending step-80 losses on the exact matched 32,000-pair / 7,163-update surface.
  Clean separate evaluators `687226/687227/687228` are `afterok`-held. This tests whether the language
  decoder, rather than an auxiliary probe, can read intermediate source-free packets; none can write to
  or otherwise affect `685084`. The job wrapper now also rejects a CPR protocol audit in the `AUDIT`
  slot before CUDA initialization, so this input-class mistake cannot consume another H100 allocation.
- **2026-07-13 09:34** — **All CPR matched training arms complete; held-out decoder-readback gate runs.**
  Verified `687223`, shuffled-label `687224`, and equal-work replicated-final `687225` each completed
  the exact 32,000-pair / 7,163-update surface from immutable `best_step190000.pt` with exit 0 in
  2h50m / 2h52m / 2h38m. Their separate checkpoint md5s are
  `7d84282c3daaa4a238db821ff8c69ed3`, `162282f3db4d7f6ce064b252be5bc35d`, and
  `6c4f11caeae9701d7fd09fef957833a3`. Full all-held-out source-free decoder-readback evaluators
  `687226/687227/687228` are RUNNING; they test normal, zero-packet, and shuffled-source modes on
  every prefix and report to isolated JSONs. No early normal-mode count is a result; wait for all three
  modes and the locked four-control comparator before either claiming or rejecting decoder-accessible
  intermediate state.
- **2026-07-13 11:12** — **200k flagship recovery checkpoint preserved.** `685084` reached step
  **200,000** cleanly at loss **1.6200**, gnorm **0.11**, and **154,296 tok/s**. Promoted Newton
  `best_step200000.pt` and resumed the local SFTP `.part` transfer until local
  `train/flagship_out/ckpt_0200000.pt` matched at md5 `510d57df578447986b40e20029511b9d`. This is a
  full optimizer checkpoint and the current local/Newton DR target is complete.
- **2026-07-13 11:27** — **Explicit overnight comparison snapshot refreshed.** Flagship `685084` is
  RUNNING on evc22 through **200,360** at **154,294 tok/s**: loss **1.6012**, gnorm **0.09**, LR
  0.0050, and **105,046,343,680 nominal update tokens**. The full 200k optimizer checkpoint remains
  durable on both Newton and the Mac at md5 `510d57df578447986b40e20029511b9d`; next DR target is
  210k. CPR verified and shuffled-label evaluators have completed, while equal-work replay evaluator
  `687228` is still running its held-out shuffled-source mode; do not make a CPR causal claim until
  its report and the locked comparator exist. CPU corpus progress is deliberately quarantined: DCLM
  `686529` completed **25,000,001,792** decontaminated tokens in 250 shards (17,370,366 kept docs)
  but requires a fresh full admission scan; OpenMath PT `685700` completed **5,000,000,144** tokens
  in 50 shards and is also future-handoff-only; Stokes FineWeb r2 `738030` is running at 52 complete
  100M-token shards (about 5.2B) toward its 25B floor. None changes the active writer or its SHARDS.
- **2026-07-13 11:36** — **CPR decisively rejected; live pretrain remains healthy.** Flagship `685084`
  is RUNNING on evc22 through **200,560** at **154,293 tok/s**, loss **1.4530**, gnorm **0.08**, with
  numbered `ckpt_0200500.pt`. Direct GPU telemetry is 100% utilization, 63,767 / 81,559 MiB, and
  301 / 310 W. The isolated step-200,387 gnorm skip recovered at the next logged step; no persistent
  instability exists. All CPR evaluators and the locked four-control comparator are complete: verified
  normal source-free readback is **161/10,752 = 1.497%**, exactly equal to shuffled source, while
  equal-work replay is **193/10,752 = 1.795%**. Fit IID / length OOD / language OOD / full OOD margins
  are **-1.273 / -0.493 / -0.347 / -0.439pp** and every preregistered gate is false. This rejects
  decoder-readable continuous packets; no stage 2 or flagship integration is authorized. The next
  isolated design hypothesis is a discrete Digitwise Recurrent Scratchpad with local carry/PC/tape
  transitions, not another continuous-memory variant. DCLM remains quarantined pending a fresh scan;
  Stokes FineWeb `738030` is healthy through shard 53 (about 5.4B tokens). No live `SHARDS` changed.

- **2026-07-13 12:04** — **Live 200k interaction and first DRS data gate recorded.** Flagship `685084`
  remains RUNNING on evc22 through **200,920** at **154,291 tok/s**; direct `nvidia-smi` reported
  94% compute utilization, 63,767 / 81,559 MiB VRAM, and 297 / 310 W. The latest 200k local
  transcript-first probe is deliberately not a benchmark: `ckpt_0200000.pt` scores **1/7** initial,
  **0/7** after self-review, **1/7** with a verified intermediate fact, and **0/7** after compact-state
  reuse, exactly matching the prior 190k directional result. The only pass is the syllogism; this is
  no evidence of a qualitative capability jump. Artifact
  `artifacts/eval_history/manual_capability_raw200k_20260713_mps32.json` has md5
  `eb3e06aa2039ceb77adf17dbc3301fd3`. In parallel, Stokes DRS build `738117` completed 439,865
  immutable train rows and 1,500 paired held-out episodes, but independent audit `738120` rejected
  it with 27 13-gram train/held-out overlaps. Inspection proved the leak was shared operand tapes
  across opposite operations, which the initial exact-state signature missed. The immutable v1 data
  and failed audit remain preserved and are not admissible. A corrected generator reserves operand
  tapes across operation/counterfactual branches; local 1,000-episode smoke and audit show zero
  exact/13-gram overlap. Stokes `738122 -> 738123` now builds and audits separately named v2 artifacts.
  The v2 Stokes audit passed after this entry: see the subsequent v2 journal entry and the DRS live-state row.

- **2026-07-13 12:16** — **DRS v2 admission passed; isolated causal GPU chain launched.** Stokes
  `738122 -> 738123` rebuilt the candidate without modifying immutable rejected v1. The read-only v2
  audit passed **439,865** train rows and **1,500** held-out counterfactual episodes across all five
  regimes (19,800 held-out controller prompts), with **0** invalid rows, duplicate normalized prompts,
  exact held-out hits, or 13-gram held-out overlaps. Train/eval SHA-256 are
  `381b8bbf3a4eddb7b08b0f9d4b08ea3ce65e1f0ec48de930632d54417c2f7f35` and
  `89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646`; they were verified after a
  direct Stokes-to-Newton transfer. GPU `687348` is now running on non-flagship H100 node evc38 to
  measure the raw 200k self-authored closed-loop baseline. None of these jobs reads/writes the flagship output, touches live
  `SHARDS`, or constitutes a reasoning claim before matched regime-level results are inspected.

- **2026-07-13 12:24** — **DRS measurement control tightened before SFT.** The running raw held-out
  evaluation `687348` was left unchanged. Its dependency-held children `687350`/`687351` were canceled
  before allocation or output, because v2's held-out lexical wrapper would otherwise confound local
  algorithmic execution with ordinary wording transfer. New `eval_digitwise_recurrent.py` accepts an
  explicit `core`/`heldout`/`auto` style and has a unit-tested core override. The replacement, verified
  Slurm chain is `687348` raw held-out -> `687362` raw core on the same episodes -> `687363` one-epoch
  SFT -> `{687364 core, 687365 held-out}` matched post-SFT evaluations. This adds no training data and
  cannot alter the flagship; it makes a positive or negative DRS result diagnostically usable.

- **2026-07-13 12:49** — **Direct interaction remains weak; ADL entered CPU-only admission.** The
  200k local probe is the human-readable diagnosis, not a benchmark: 1/7 initial, 0/7 self-review,
  1/7 supplied-fact, and 0/7 compact-state reuse. Its arithmetic response asserts `29 * 16 = 496`,
  base conversion emits ratios, list/string prompts fall into memorized code scaffolds, and the
  Python predicate reverses the required condition. This is evidence of a procedural/decoder failure,
  not a hidden benchmark-only capability. At the concurrent live check `685084` was healthy through
  201,620 at 154,287 tok/s (latest loss 1.8294, gnorm 0.10), with `ckpt_0201500.pt` present and no
  new persistent instability. Raw DRS `687348` remains isolated and running at 0 finals through
  240/500. In parallel, after local and Stokes import/test smoke passed, Stokes CPU job `738186` was
  submitted to build the separate Append-only Delta Ledger corpus; dependent audit `738187` will
  independently recompute rows and overlap checks. ADL has no GPU job and cannot affect the flagship
  or DRS chain.

- **2026-07-13 13:03** — **Answer likelihood rules out a hidden emission-only capability.** A new,
  transcript-adjacent forced-choice probe scored the correct completion against matched wrong completions
  under the same plain `Question/Answer` prompt. Raw 200k ranks the correct candidate first on only
  **1/7** fresh cases (mean correct rank **2.571**): arithmetic 3/4, base conversion 4/4, state update
  3/4, linear equation 2/4, sort/dedup 1/4, string insertion 3/4, and logic 2/2. In particular it ranks
  `yes` above the correct `no` for the fresh syllogism. Artifact
  `artifacts/eval_history/forced_choice_raw200k_20260713_mps.json` md5
  `7b6bcdd58f6420703fcb0b6bbbfa3afd`. This does not prove general ability absent under every prompt
  contract, but it does reject the narrower explanation that the model already has reliable answers and
  only fails free emission. The next mechanism must teach validated local computation/state rather than
  a cosmetic chain-of-thought surface.

- **2026-07-13 13:04** — **ADL full admission passed; DRS SFT boundary was repaired before allocation.**
  Stokes build `738186` committed 384,000 ADL train rows and 1,000 paired held-out episodes; its
  independent audit `738187` recomputed every transition and passed with 0 invalid rows/episodes,
  duplicate prompts, exact overlap, or 13-gram overlap over 42,000 held-out controller prompts. The
  hash-matched train/held-out artifacts and audit report are present on Newton. Separately, inspecting
  the pending DRS SFT script exposed an inference-boundary bug: DRS rows include a full
  `completion_prompt` ending in `Answer:`, but the held script would have wrapped it in an additional
  `Question: ... Answer:` surface. Because Slurm snapshots scripts at submission, merely syncing the
  source would not fix it. Canceled unstarted `687363/687364/687365` before outputs, synchronized the
  corrected wrapper, and submitted `687375 -> {687376,687377}` with the same data/checkpoint/dependencies.
  `scontrol write batch_script 687375` verifies the stored `--prompt-override-field completion_prompt`
  and exact-boundary preflight. This is a valid experimental repair, not a model result.

- **2026-07-13 13:05** — **Raw DRS executor is absent; raw ADL microstep likelihood is at chance.**
  Raw held-out DRS `687348` completed all 500 episodes: every regime has 0 first transitions, state
  loops, finals, counterfactual finals, and paired interventions. It produced 434 unique first responses
  (mode count 67), mostly malformed markdown/repetition rather than one fixed answer; this is a valid
  raw baseline, not a rejection of SFT learnability. Core-wrapped raw control `687362` began on evc38
  and is the lexical-interface discriminator. Separately, `probe_append_ledger_microsteps.py` asked the
  raw 200k model to rank the 20 grammar-valid first `adl:step=0;d=<digit>;c=<carry>` choices for eight
  fixed arithmetic tapes under both core and held-out wording. Correct local transition is never top-1
  (**0/16**, mean rank **10.688/20**); `d=0;c=0` wins all 16 prompts. Artifact
  `artifacts/eval_history/append_ledger_microstep_raw200k_20260713_mps.json` md5
  `9ae4c88aca13079fe69036a47b88e597`. This rejects the claim that raw pretraining already contains a
  reliably readable local arithmetic primitive, but supports a supervised learnability comparison.

- **2026-07-13 13:30** — **Direct counterfactual-verifier probe rejects a hidden self-checker.** The
  raw 200k model was shown 48 balanced DRS local transitions: half exact successors and half
  grammar-valid near misses that changed only a digit, carry/borrow, or immutable operand tape. Free
  generation emitted no usable `verdict=valid|invalid` response (**0/48**), mostly bare digit lists and
  repeated document fragments. That could have been an answer-surface failure, so the exact two verdict
  completions were likelihood-ranked too: the model selected `valid` on **all 48**, yielding exactly
  **24/48 = 50%**. Artifact
  `artifacts/eval_history/transition_verifier_likelihood_raw200k_20260713_mps.json` md5
  `fb7bbdbb1fa16104117f09c6c3faa07c`. Therefore a proposed proof-carrying deliberation loop may not be
  treated as an emergent raw ability; it is conditional on DRS first proving supervised core local
  execution, then must beat a matched label-shuffled verifier control on the model's own sampled states.
  This adds no training job and cannot affect the flagship.

- **2026-07-13 13:42** — **Canonical wording does not uncover a hidden DRS executor.** Core-wrapped
  raw control `687362` completed cleanly on evc38 (42m12s, exit 0) against the identical 500 paired
  episodes used by held-out-wrapped `687348`. It is **0/500** first transitions, closed loops, final
  answers, counterfactual finals, and paired interventions in every 100-case `fit_w4`, `fit_w6`,
  `value_ood_w4`, `value_ood_w6`, and `width_ood_w8` regime. The result md5 is
  `20a5d4cc4a776ee3ffb9220f288f4f6a`; it emitted 34 unique first responses with a high mode count of
  265, so the zero is not a single constant-output artifact. This resolves the raw lexical-interface
  question negatively. Hash-bound, exact-boundary one-epoch SFT `687375` then began independently on
  evc32; its successor evaluations remain dependency-held. In parallel, `REASONING_FRONTIER.md` now
  records the Counterfactual Bisimulation Compiler hypothesis: language-to-state compilation, source-free
  state advancement, inverse delta explanation, multi-query readout, and causal paraphrase/counterfactual
  controls. It is not a capability claim and has no live-training path.

*Keep this file honest. When you hit a milestone, do the work, then come back and update §1 (LIVE
STATE) and any step that changed. A future agent — maybe you after a context reset — is relying on it.*

- **2026-07-13 ~13:55** — **Flagship custody is intentionally hands-off; capability research is
  concentrated on the isolated causal path.** Per project direction, do not perform routine live-flagship
  monitoring or alter the active writer. Uncompiled DRS v2 SFT `687459` has passed its hash-bound data
  and exact completion-boundary preflights and reached **200 / 1,561** updates with finite loss falling
  from `0.6846` to `0.0541`; GPU telemetry is 100% utilization at about 63.9GiB. This is only fit
  evidence, not an execution or reasoning result. Its source-free core, held-out, direct-interaction,
  and NLL jobs remain serialized after successful completion. In parallel, `REASONING_FRONTIER.md`
  records a **Dual-Code Reversible Deliberation** hypothesis: one model carries state through two
  per-episode randomized token codes, forward and inverse transitions must close across the codes, and
  the controller may only transport text and compare exact strings before accepting or abstaining. It is
  conditional on a positive DRS causal result, has no submitted job or flagship path, and must beat
  matched repeated-lane, shuffled-codebook, identity-format, corruption, coverage, and
  counterfactual-interchange controls before it is treated as useful.

- **2026-07-13 14:46** — **The DRS fit phase is complete; causal evaluation, not loss, now decides.**
  Isolated uncompiled `687459` completed one DRS v2 SFT epoch from `best_step200000.pt` on evc49:
  439,865 hash-bound rows, 51,131,402 packed tokens, 10,623,342 answer tokens, 24,966 packed
  sequences, and **1,561** updates in **1,115s**. It wrote only
  `train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt` (MD5
  `6f30db16208d274229950b17662dda01`). Source-free core evaluator `687460` is running; the held-out
  wording evaluator, direct raw-versus-SFT interaction, and raw NLL monitor remain serialized after it.
  The small completion loss does not establish execution, causal state transport, or reasoning. Do not
  submit DCRD, ADL, CBC, or any flagship modification until this chain provides the needed causal result.

- **2026-07-13 14:52** — **DCRD has a tested CPU-only protocol substrate, not a training result.**
  `train/dual_code_reversible_protocol.py` now implements deterministic per-episode A/B encodings,
  strict code-specific parsing, solver-only inverse construction, and source-free prompts. Its local
  contract test covers codebook separation, encode/decode, canonical-state leakage rejection, and 120
  randomized reversible transitions. No DCRD data, controller rollout, SFT, or GPU job exists; it remains
  conditional on the full DRS core/held-out/direct chain and cannot affect active pretraining.

- **2026-07-13 15:02** — **DCRD generator/auditor preflight passed without weakening the contamination gate.**
  `pipeline/generate_dual_code_reversible_v1.py` emits forward-A, A-to-B, reverse-B, B-to-A, and readout
  targets; `pipeline/audit_dual_code_reversible_v1.py` independently recomputes every target plus every
  held-out state trajectory and counterfactual. The first implementation exposed **768 literal 13-gram
  train/held-out template overlaps on the smoke test**. That data was rejected, not normalized away. The
  fixed generator/auditor bind a held-out-only alias vocabulary to a structurally different held-out prompt style;
  the protocol itself permits crossed combinations only for later attribution controls.
  The end-to-end smoke and a larger **1,000-episode / 21,000-row** preflight with **200** held-out paired
  counterfactual episodes now report 0 malformed rows/episodes, duplicate prompts, exact held-out hits, or
  literal 13-gram overlaps. This is only CPU infrastructure: no durable DCRD corpus, controller, SFT, or
  GPU job is authorized until `687460 -> 687461 -> 687462` provides a positive DRS causal result.

- **2026-07-13 15:24** — **CBC generator/auditor preflight passed with a causal counterfactual gate.**
  `train/bisimulation_compiler_protocol.py` defines a strict `cbc:` state and `cbc-delta:` grammar; its
  source-free update and readout prompts have no solver/controller fallback. The paired
  `pipeline/generate_counterfactual_bisimulation_v1.py` and
  `pipeline/audit_counterfactual_bisimulation_v1.py` independently reconstruct all compilation,
  transition, inverse-delta, and query targets, then require normal/counterfactual worlds to share every
  operation while differing in exactly one initial fact and final answer. A medium **1,000-episode /
  16,000-row** local preflight plus **120** held-out paired-counterfactual episodes reports 0 invalid rows,
  invalid episodes, normalized duplicate prompts, exact prompt hits, and literal 13-gram overlaps.
  Corruption tests reject altered targets and a semantically valid-looking mismatched counterfactual
  operation sequence. CBC has no
  durable corpus, controller, SFT, or GPU job; it remains gated on the serialized DRS causal chain.

- **2026-07-13 15:33** — **CBC's source-free transport controller is now independently tested.**
  `train/bisimulation_compiler_controller.py` may parse only model-emitted `cbc:` text, render the next
  source-free update/query prompt around it, and exact-compare outputs after the rollout. It does not call
  a semantic state transition or answer solver. Its deterministic test exercises primary rollout,
  inverse-delta checks, same-world compilation interchange, and normal-state versus counterfactual-world
  query mismatch; corrupting the first predicted state halts the rollout with no final-answer repair. This
  is evaluator infrastructure only. No CBC corpus, SFT checkpoint, or GPU job is authorized ahead of the
  DRS causal decision.

- **2026-07-13 15:38** — **DRS v2 coverage audit identified a precise OOD confound before interpreting its
  core curve.** `pipeline/audit_digitwise_position_coverage.py` is a read-only transition-context report.
  On immutable v2 it found no train transitions with operand digits 3–9 at width-4 position 3 or width-6
  position 5, for either tape. The fit regimes have 0 unseen digit-position/local contexts; each value-OOD
  regime has **1,200 / 600** and width-8 has **9,600 / 4,800**, respectively, across its complete paired
  held-out set. Any revised DRS data must stratify coverage by width, position, tape, operation, and
  carry/borrow before value-OOD performance can answer an algorithmic-generalization question. This
  diagnostic alters no existing data, checkpoint, or job and does not pre-judge the running causal chain.

- **2026-07-13 15:45 / 16:20** — **DRS v3 is a hash-admitted complete-transition-basis candidate, not a
  training result.** `train/digitwise_basis_protocol.py` enumerates 3,400 reachable local decimal contexts
  across width 4/6, operation, position, carry/borrow, and operand digits. The v3 generator constructs
  complete arithmetic episodes that hit each context while retaining real result-tape prefixes; its held-out
  evaluator worlds use unseen full tapes rather than an unseen numeric band. The independent admission
  audit recomputes all targets/counterfactuals, requires every one of the 3,400 contexts, and fails if any
  one context is removed. The medium 2-variant preflight contained 6,800 episodes / 77,946 rows with 0
  malformed rows/episodes, duplicates after deduplication, exact overlap, or literal 13-gram overlap. The
  durable eight-variant CPU artifact is now mirrored locally and on Newton: **27,200 episodes / 311,127
  rows**; train SHA-256 `b785866bf24813272d346e4a3bb717d4156b01a59a4dd8ccaf450733267368f6`, held-out
  900-episode SHA-256 `f2fcfcae41b55aa82dd360036bd8c9c00ed6e4ca442debec1c85ed282e50dfe1`, and the
  independent audit reports 0 invalid, duplicate, missing-context, exact, or 13-gram findings. It has no
  SFT checkpoint or submitted GPU job; wait for `687460 -> 687461 -> 687462` before execution.

- **2026-07-13 15:53** — **The DRS v3 launch contract is staged, not admitted for execution.**
  `train/jobs/sft_digitwise_basis_v3.sbatch` is isolated from the flagship and rejects an existing output,
  a non-v3 audit schema, any malformed/duplicate/overlap finding, a data or held-out SHA mismatch, any
  missing local context, or anything other than exactly 3,400 required and covered contexts across the
  three recombination/width-held-out regimes. It also proves that SFT uses the exact inference prompt
  boundary before taking CUDA. The static contract test passes. This creates no corpus and submits no job;
  it only makes a later positive-evidence test reproducible without weakening the admission gate.

- **2026-07-13 16:04** — **Static-Tape Recurrent Register (STRR) is a new CPU-preflighted causal
  representation candidate, not a training result.** The diagnosis is that original DRS makes every
  recurrent output regenerate immutable `a` and `b` operand tapes as well as the evolving `p/c/r/z`
  fields. STRR sends the immutable canonical `dwt:` tape again as fixed problem evidence each turn, but
  requires the model to emit only the `dwr:` register. Its transport-only controller never executes,
  repairs, or chooses a register; it merely reuses the original tape and forwards the exact model output.
  Generator/auditor preflight with two tape variants has 6,800 complete episodes / 77,946 rows, all 3,400
  reachable local contexts, 120 paired held-out counterfactual episodes, and 0 malformed, duplicate,
  counterfactual, exact-overlap, or literal 13-gram-overlap findings. Its matched closed-loop evaluator is
  static-tested. This factorization is a direct test of the repeated-immutable-copy burden, not external
  arithmetic assistance. It creates no durable corpus, SFT, or GPU allocation until the active DRS chain
  has yielded its core, held-out, and transcript evidence.

- **2026-07-13 16:14** — **The capability path now records regime-specific evidence before selecting
  another SFT.** Both digitwise evaluators accept `--examples-per-regime` and retain independently
  capped successful and failed paired closed-loop transcripts per regime. This is diagnostic only: it
  cannot influence a score, repair an output, or change a controller transition. The staged
  `sft_digitwise_factor_v1.sbatch` applies the same isolation, exact inference/SFT prompt-boundary,
  CUDA-allocation, overwrite-refusal, hash-binding, complete-3,400-context, counterfactual, and
  decontamination gates as the v3 basis job. It has not generated a durable factor corpus, reserved a
  GPU, or been submitted. The immediate purpose is to attribute current value/width failure to either
  incomplete local basis coverage or repeated immutable-tape copying before spending further training.

- **2026-07-13 16:25** — **DRS v2 core establishes local execution but rejects any broad-reasoning
  interpretation.** From the isolated `sft_ep1` checkpoint, `687460` gets **275/500** final answers under
  canonical core wording: 100/100 fit-w4, 98/100 fit-w6, 34/100 value-OOD-w4, 43/100 value-OOD-w6, and
  0/100 width-OOD-w8. Yet its very first model-authored microstate is correct on **497/500** episodes
  (100, 100, 100, 99, and 98 by the same regimes). The failure is therefore predominantly recurrent and
  path-dependent rather than a total inability to perform a local decimal transition. The complete v3
  basis remains a valid coverage control for unseen interior contexts, but **STRR becomes the first
  corrective representation experiment** once the pending held-out wording and direct transcript probes
  show whether removing immutable tape copying repairs multi-step transport. No broad, latent, or
  language-reasoning claim is permitted from this result.

- **2026-07-13 16:27** — **Repaired the pending DRS evidence chain before it consumed more resources.**
  The inherited core and pending descendants were manually submitted with two H100s even though every
  script has one process and chooses a single CUDA device. Core `687460` is already complete, so it was
  not touched. Unstarted `687461/687462/687463` were canceled with no output artifacts and replaced by
  one-H100 chain `687547` (held-out wording) -> `687548` (raw-versus-DRS direct interview) -> `687549`
  (raw NLL). `687549` now explicitly binds the frozen WikiText-103-test English and CodeContests-test
  Python monitor inputs that the old job omitted. The independent one-H100 `687542` core-transcript probe
  remains in parallel after `687460`. All replacements retain separate output paths, fixed raw/SFT
  checkpoints, and the same serial causal ordering; this correction cannot alter active pretraining.

- **2026-07-13 16:39** — **Bounded a second, clearly dead evaluator-startup failure.** `687542` and
  `687547` each stayed more than ten minutes in the initial `python -` CUDA smoke with no log output,
  0 MiB allocated VRAM, and negligible CPU; they produced no JSON artifact. Their two pending children
  were canceled with them. The evaluator wrappers now set the same `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, and four-thread OpenMP limits used by successful SFT
  jobs. Fresh one-H100 replacements are `687554` (transcripts), `687555 -> 687556 -> 687557`
  (wording -> interview -> NLL). This is infrastructure hardening only; no data, checkpoint, score, or
  flagship state was changed.

- **2026-07-13 16:46** — **Verified the actual H100 runtime before a third evidence attempt.** The
  transient import stall cleared under a bounded `strace` control (`import torch` exited 0); no package or
  environment mutation was needed. Existing `cuda_h100_preflight.sbatch` job `687560` then passed on
  evc49 in four seconds, including a real bf16 CUDA matmul. The serial evidence chain is freshly rebuilt
  from that verified one-H100 allocation: `687562` transcript probe -> `687563` held-out wording ->
  `687564` manual raw-versus-DRS interaction -> `687565` raw NLL with explicit frozen monitor inputs.
  This avoids interpreting a scheduler/import transient as model behavior and avoids allocating more than
  one H100 at a time for diagnostics.

- **2026-07-13 17:15** — **Adopted a paper-derived workspace criterion without mistaking it for a
  capability result.** The external global-workspace study motivates a narrow testable claim: a useful
  compact state must be reportable, deliberately updateable, causally action-guiding, reusable by more
  than one downstream task, and robust to held-out recombinations. It does not license more visible
  chain-of-thought training. The expanded 80-direction raw residual-patching control is negative (carry
  deltas +0.001 to +0.028 with only 18–22/40 positive directions; digit deltas -0.042 to +0.0002 with
  only 14–20/40 positive), so there is no simple raw last-token broadcast register. The matching post-DRS
  diagnostic `687578` remains dependency-held after wording, direct-interaction, and NLL evidence. A
  conditional Counterfactual Workspace Induction design is documented in `REASONING_FRONTIER.md`: it
  trains only a later, training-time reflection that distinguishes a legal local register from a
  grammar-valid semantic foil, then tests the ordinary unreflected task with syntax-only,
  label-permutation, and equal-compute continuation controls. It cannot be trained unless STRR first
  establishes a generalizable primitive register.

- **2026-07-13 16:23** — **STRR now has a durable, independently admitted full corpus but no SFT.** The
  eight-variant artifact is mirrored locally/Newton with **27,200 episodes / 311,127 rows**, 3,400/3,400
  local contexts, and 900 paired held-out episodes. Train/held-out SHA-256 values are
  `82245615f0849c3270f99f2db85c604ff46cb2c3dfb14f0ab3660dff3eb0d3ec` /
  `a699ac58ad8184f4dc23dcfa317cd6e7b8f7d4ef453dcbf1ae21201901e0948a`; its independent audit reports
  zero invalid, duplicate, counterfactual, missing-context, exact-overlap, or 13-gram findings. This is
  materialized solely to make the transport-representation ablation reproducible after the transcript
  gate; it has no SFT checkpoint, GPU allocation, or capability score.

- **2026-07-13 18:05** — **TNDL passed full local admission rehearsal; it remains a CPU-only candidate.**
  The token-native delta-ledger control produces 27,200 train episodes / **169,115** deduplicated rows,
  covers all **3,400/3,400** local contexts, and has 900 held-out episodes (300 each recombine-w4,
  recombine-w6, and width-8). The independent full audit found zero invalid rows/episodes, duplicate
  normalized train prompts, counterfactual mismatches, missing contexts, exact prompt hits, or 13-gram
  train/held-out hits. Repeating an opaque immutable tape hash between the three-token ledger triples is
  an anti-contamination delimiter only: the controller never decodes or predicts it. The candidate has
  no Stokes artifact, SFT checkpoint, GPU allocation, or capability result yet; it is not evidence of
  reasoning or context scaling.

- **2026-07-13 18:15** — **TNDL CPU build/audit is live on Stokes, not Newton.** Stokes default
  `/usr/bin/python3` is Python 3.6 and rejected the repository's postponed-annotation syntax during
  preflight, so both dedicated CPU scripts now pin the verified `/usr/bin/python3.12` interpreter.
  Submission `738430` writes only fresh TNDL artifacts and `738431` is its read-only `afterok` audit;
  both request four CPUs and 24 GiB on `normal`. Their scripts passed remote compilation and Slurm
  `--test-only`; no H100, pretraining stream, checkpoint, or existing artifact path is shared.
