# AGENT RUNBOOK — Shohin autonomous custody

> **If you are a new agent taking over, read THIS FILE FIRST, top to bottom.** It is the single
> source of truth for the live training run and the standing directive. Everything you need to keep
> the model alive, hit the next milestone, and not break anything is here. Other docs
> (`MASTER_PLAN.md`, `DIVERGENCE_DIAGNOSIS.md`, `DATA.md`) are background/history; this file is the
> operational plan of record.
>
> **Last updated:** 2026-07-12 ~14:50 EDT (`685084` remains healthy through 177.66k; fresh composition interview confirms raw 170k has no general execution or compact-state transfer; future language builds require hash-bound scan approvals and TACO is scaling behind all-test replay). Keep the "LIVE STATE" section current
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

What that means operationally, and is now settled — **do not relitigate:**
- **Single H100 flagship.** The current user directive is to maximize one H100. The 8xH100 path is
  unavailable and the previously validated 2xH100 path is not the active flagship lane. Do not move
  the protected run to multi-GPU without a new explicit user decision.
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

| Item | Value (as of 2026-07-12 ~14:50 EDT) |
|---|---|
| **60k pretrain job** | `680149`, name `shohin-flagship`, node **evc22**, **DONE** (`[done] 60000 steps in 112203s`) |
| **Extended pretrain job** | `683715` completed cleanly. Current active continuation is **`685084`** on **evc22**: one H100, `BS=32 ACC=8 CKPT=250`, exact 524,288-token updates, and the proven default compile path. It resumed `ckpt_0141500.pt -> step 141501`; the prior dual-GPU successor remains canceled per user instruction. |
| Extended pretrain status | **`685084` is healthy through step 177,660** at ~**154.29k tok/s**. Loss remains in the normal ~1.2-2.4 band and gnorm is normally 0.07-0.32; isolated outlier guards recover on their next steps. `ckpt_0170000.pt` is preserved on Newton as `best_step170000.pt`, and the full local DR copy `train/flagship_out/ckpt_0170000.pt` matches both at md5 **`7ad139b6b9b537a5a3e65978f8296419`**. Next local DR target is 180k. Do not integrate CUDA graphs into live training: the clean whole-update canary gained only ~1.8% while removing the flagship's established guard/observability path. |
| **SFT feedback job** | `681000`, name `shohin-sft`, **DONE**; wrote baseline `train/sft_out/sft_ep3.pt`. Isolated v2 pilot `685708` completed one epoch from `best_step120000.pt` to `train/sft_v2_120k/sft_ep1.pt`. It is a narrow arithmetic-format ablation, not a promoted broad-reasoning recipe. V4 `686323` was canceled before its first step after stale 40/35/15/10 weights; v4 `686324` was canceled before any artifact after a code-boundary audit found 461/3,542 legacy BPE prompt-prefix mismatches. Both are invalid and preserved. Corrected v4 pilot **`686326`** completed one epoch in 1,218s to `train/sft_v4_168750_r3/sft_ep1.pt`, with audited 40/47/8/5 weights and inference-aligned prompt/completion token construction. It is an unevaluated candidate, not promoted. |
| **Eval board job** | Corrected CUDA-only v2 board **`686277` completed**: GSM8K maj@4 **6/100**, pass@1 **14/100**, MATH-500 **6/100**, HumanEval **6/164**, MBPP **0/100**. The v2 pilot is **rejected for promotion**. RG held-out `686278` is **90/800 = 11.25%** and in-training `686279` is **98/800 = 12.25%**: it learned a few routines that transfer but remains zero on most logic/transformation/cipher/geometry families. The corrected raw-base board **`686315`** pinned `best_step168750.pt` and completed: GSM8K maj@4 **5/100**, pass@1 **2/100**, MATH-500 **2/100**, HumanEval **7/164**, MBPP **0/100**. `686316` direct adaptive interaction was **1/6 initial, 1/6 after explicit self-review, 1/6 with a verified intermediate fact**; only the simple syllogism was correct. V4 r3 public board **`686336` completed and is rejected**: GSM8K maj@4 **5/100**, pass@1 **14/100**, MATH-500 **1/100**, HumanEval **2/164**, MBPP **0/100**. Its corrected held-out procedural result `686337` is **209/800 = 26.125%**, above V2's 90/800 on the same evaluator, but the later raw base means this is a useful diagnostic signal rather than clean data-only attribution. V4 remains a generator/verifier candidate, not a broad promotion. |
| **V5 primitive board** | **`686401` completed; V5 is rejected for broad promotion.** From the raw-168.75k base it scored GSM8K maj@4 **10/100**, greedy **9/100**, MATH-500 **3/100**, HumanEval **2/164**, and MBPP **0/100**. It shows narrow arithmetic-format transfer but regresses code from raw's 7/164 HumanEval and does not establish broad math, code, or instruction-following transfer. |
| **V6 contract SFT** | **`686413` completed cleanly** to `train/sft_v6_contracts_168750_r2/sft_ep1.pt` (4,535 updates, 1,388s). Its fresh 245-case contract holdout `686414` rises from raw **20/245 = 8.16%** to **142/245 = 57.96%**, especially review 28/35, scaffold 34/35, and reuse 34/35. This is deliberately not a latent-reasoning claim: independent deep audit `686415` is only **4/8 initial, 1/8 review, 1/8 scaffold, 0/8 compact reuse** and shows invalid compact calculations. Matched matrix `686438` completed at Q/A **4/48 -> 17/48**, direct **5/48 -> 10/48**, CoT **0/48 -> 21/48**, one-shot **7/48 -> 7/48**; it establishes contract transfer but not compact-state reasoning. No public board is justified. |
| **V7 typed-state SFT** | The fresh solver-verified state corpus has **315,000 train / 10,500 held-out** rows across write, repair, and reuse contracts; independent audit reports 0 malformed rows, duplicate questions, exact held-out prompts, or 13-gram held-out overlap. Raw pinned baseline `686456` completed its 420 rows before a duplicate-output exit: **21/420 = 5.0% answer accuracy and 0/280 valid typed states**. V7 `686467` completed cleanly from `best_step168750.pt`. Its exact 420-prompt holdout `686471` reaches **307/420 = 73.10% answers** and **169/280 = 60.36% exact states**: repair 128/140 answers and 132/140 states, reuse 140/140 answers, but write only 39/140 answers and 37/140 states. The independent eight-case interview `686484` is **1/8 initial, 1/8 review, 1/8 scaffold, and 0/8 compact reuse**, with wrong arithmetic even after a verified fact and malformed/unrelated state text. **Reject V7 as a general-reasoning or latent-compaction candidate.** Keep it isolated as a constructed-contract diagnostic; its hash-matched state/deep artifacts are `1f9fe0b2993d1a9dafc98cd2d7943887` and `c4963fae52d5ac9c38614e77f93f98c8`. |
| **2-H100 speed canary** | `681040`, name `shohin-ddp2-canary`, **COMPLETED cleanly** on evc42: resumed from `ckpt_0060000.pt`, `world=2`, loss in band, no DDP hang, ended at `61050` in 2093s with ~262k tok/s (~1.76x the 1-GPU ~149k tok/s). This validates the 2-H100 path. Do not confuse idle `evc6`/`evc16` with H100 capacity: they are V100 nodes and the trainer is bf16/H100-oriented. `evc105` is idle 4x H200 NVL, but Slurm rejects this account on `short`/`ucfit`, so it is not usable unless the user's allocation changes. |
| 60k final loss | final logged band ~1.5-1.7; last logged step 59990 loss 1.6989, lr 0.0005 |
| 60k skips | **45 total**, stable/healthy |
| **Corpus-expansion job** | `680324` — **✅ DONE** (finished ~12:10) |
| finemath3 output | `artifacts/shards/finemath3/` — **✅ COMPLETE: 125 shards, exactly 25.0B tokens** (`manifest.json` present, 22 GB; 8,575 contaminated docs dropped vs evalgrams). **Included in the 300k relaunch SHARDS.** |
| SFT mix (Newton) | `artifacts/sft/sft_mix_core.jsonl` — **97,439 examples**, rebuilt 2026-07-08 with hard eval filtering. Audit: 0 malformed rows, 0 duplicate questions, 0 exact eval-prompt hits; builder dropped **206 exact eval-prompt overlaps** and **741 eval 13-gram overlaps** before writing. md5 `53ed91368b4c238dc18a1ab1699e4158`; report md5 `21459b382767801e205f3f625ce106cd`. |
| Local teacher distillers | Nemotron screen run completed at **1,781 rows** but provider health was poor (`kept=19`, `err=52506` in the screen run). Bounded probes after that were also unhealthy: Nemotron `limit=5` kept 0 with 3 provider errors; GLM `limit=3` kept 0 with 3 provider errors. **Leave Nemotron and GLM paused until provider health clears; do not blindly respawn.** HY3 bulk process died/stalled after ~25.2k rows; `conc=2` and `conc=1` restarts exited without appending, even though a tiny direct Hermes/probe call completed cleanly. Claude/minimax snapshots are present. GLM remains the preferred strongest open-weight teacher when available. |
| **Verified-data expansion** | **Active, CPU-only, isolated from pretrain.** `openmath_pt` is complete at **5,000,000,144 tokens / 50 shards** and remains future-relaunch-only. FineWeb-Edu `686298` is **complete and admitted only for a future natural relaunch**: **4,599,762,693 tokens / 46 shards**, 4,139,127 docs kept from 9,672,101 seen; 5,531,139 below-quality docs and 1,816 eval-contaminated docs were rejected. Manifest md5 `05739fa1cb31a45e1c496909f9461fa1`; full-shard entropy/top-token scan `686340` found no outlier or byte-fallback shard. DCLM-Baseline pilot `686342` and full scan `686360` are **complete and accepted as a pilot only**: **5,000,000,487 tokens / 50 shards**, 3,506,817 kept of 4,203,263 seen, with 1,820 eval-contaminated rows rejected and no entropy or byte-fallback outlier. CPU-only replacement build **`686470`** now targets 25B tokens; it must receive its own full scan and replaces rather than supplements the 5B pilot at a future handoff. The solver-verified **primitives v1** ablation data is complete: **210,000 train / 3,500 held-out** rows across arithmetic, base conversion, state updates, sort/dedup, string insertion, syllogism, and correction; 0 malformed rows, 0 duplicate normalized questions, and no train/held-out prompt overlap. `rg_v4` has **374,659 valid traces**, 0 malformed rows, and 0 normalized-question duplicates across 25 answer-checked families. CodeContests `686291` completed **3,000** train-only Python examples; its completion-form derivative has **3,593** deduplicated examples. Frozen **v4** has **643,595 clean rows**: math 240,297 / procedural 374,659 / code 3,542 / teacher 25,097. Exact SFT packing `686312` passed at **62,926** sequences. The original 15% code target would replay code 7.7x and teacher 3.1x, so v4 pilot weights are revised to **40/47/8/5** (math/procedural/code/teacher; code ~4.1x, teacher ~1.6x). `686317` is independently scaling more verified CodeContests data for a later mix; it does not alter v4. No new output may enter a frozen SFT mix before its final audit. |
| **25B language replacements** | FineWeb-Edu pilot is accepted only as evidence; CPU-only **`686530`** is building `fineweb_edu_25b` and DCLM **`686529`** is building `dclm_baseline_25b`. Earlier partials from `686526`/`686470` were renamed and preserved, then rejected before admission because they used a stale-pickle-only decontamination path. New builds augment the pickle with direct n-grams from every current eval JSONL and record those counts in each manifest. Dependency-held scans `686538`/`686539` now write machine-readable `artifacts/shard_scans/*.json` reports. Before language relaunch, each report must be reviewed and converted to a hash-bound `*.approved.json` record with matching manifest/report hashes, >=24.5B manifest tokens, and <=1% byte fallback; both relaunch scripts refuse to run without them. Never mix in the 5B pilots. |
| **TACO algorithmic-code gate** | Full pilot audit **`686514` passed**: every one of 250 initially accepted Python programs matched its source record and passed **all** bounded supplied stdin/stdout cases; 0 execution/missing-case/source-unmatched drops, 0 malformed rows, duplicate prompts, exact eval hits, or 13-gram hits. Hash-matched local/Newton full derivative md5 `a33b5e0b5287bdcaec2fac550a13d5cb`; report md5 `c42b6e5cf8408c0d1cccd5f73b6a319e`. It remains an `algorithmic_code` source, deliberately separate from HumanEval/MBPP completion code. Prefix-order chain `686552 -> 686553` is preserved as a **non-admissible control** because it started before deterministic stream shuffling landed. Only shuffled chain `686554 -> 686555` can become a training candidate: it selects 3,000 examples from 5,000 seen with a 10k-row seeded shuffle buffer, then replays all supplied tests. On success, **`686560`** freezes only that full derivative through decontamination/deduplication into an `algorithmic_code` candidate (>=1,000 rows required) and **`686561`** measures its true 2,048-token packing capacity. No SFT may be submitted before those reports are reviewed. |
| **Direct capability audit** | Raw 168k/170k evidence remains negative: raw scored **4/48 Q/A**, **4/48 plain instruction**, **0/48 CoT**, **5/48 one-shot**; the pinned 170k compact-state interview `686370` was **1/8 initial, 0/8 review, 1/8 scaffold, 0/8 compact-state reuse**. A fresh local-MPS seven-case re-interview reproduced **1/7 initial, 0/7 review, 1/7 verified-fact use, 0/7 compact reuse**. A separate eight-case composition interview then independently reproduced **1/8 initial, 0/8 review, 1/8 verified-fact use, 0/8 valid `state=` emissions, and 0/8 valid-state-and-reuse**. Only the simple logic constraint passed; product/reject arithmetic, base conversion, sequential state, sort/string transformation, and two executable Python contracts all failed. The fresh typed-state baseline `686456` is similarly negative: **21/420 answers and 0/280 exact emitted states**; by contract it is 0/140 write, 6/140 repair, and 15/140 reuse answers. V5 primitive SFT has a real but narrow direct signal: held-out primitive gate **`686368` = 272/700 (38.86%)** vs raw **0/700**, with 100/100 syllogisms, 88/100 string insertion, 47/100 correction, but only 12/100 arithmetic, 4/100 base conversion, 2/100 sort, and 19/100 state updates. Its verbatim interview `686388` is **2/8 initial, 2/8 review, 1/8 scaffold, 3/8 compact-state reuse**; the new reuse successes are trained primitive-like operations only. Matched matrix `686389` does show Q/A **4/48 -> 17/48** and CoT **0/48 -> 11/48**, concentrated in arithmetic/sort/state, while direct instruction is only **5/48 -> 6/48** and one-shot regresses **7/48 -> 5/48**. Full evidence: `CAPABILITY_DIAGNOSIS.md`, `TARGETS_AND_GAPS.md`, `artifacts/eval_history/deep_interaction_raw170k_r2_686370.json` (md5 `1979bcc79cb18830cb3080a7cab85e82`), refreshed `manual_capability_raw170k_refresh_20260712_mps.json` (md5 `9cd3216365b5292851e298bee4a1aeef`), composition artifact `generalization_interview_raw170k_20260712_mps.json` (md5 `d9ad30fad6c00958ad9d6908ca14c38a`), `sft_v5_primitives_168750_rg_heldout_686368.json` (md5 `4b6fe30e6b75ca7e409a152286f0ff8e`), `sft_v5_primitives_168750_deep_interaction_686388.json` (md5 `19ae7208d981f22abbfe8f9b48523c0c`), and remote state artifact `artifacts/eval_history/raw168750_primitives_v3_state_p20.json`. |
| **Post-board transcript gate** | Fresh operator-run probe `686425` compared raw 168.75k with V5 on seven new cases and preserved every turn. Raw: **1/7 initial, 0/7 review, 1/7 verified-fact, 0/7 state reuse**. V5: **3/7, 3/7, 2/7, 3/7**, limited to arithmetic, sorting, and logic; it still fails base conversion, sequential state update, string insertion, and syntax-valid Python. V5 almost never emitted the requested `state=` line, so its reuse score is **not** latent context compaction. Canonical local/Newton hash-matched artifact: `artifacts/eval_history/manual_capability_raw168750_vs_sft_v5_20260712_JOBID.json`, md5 `28dd0b15de2af16a10a2012f630072a1`. |
| **Verifier selection gate** | Corrected evaluator job `686437` reports held-out GSM8K first-pass **7/100**, oracle@16 **36/100**, verifier@16 **9/100**. Candidate sampling provides real headroom, but the 30-step verifier captures only two points of it. Improve generation and verifier ranking separately; do not report best-of-16 as current pass@1. |
| **Verifier data-volume gate** | The clean first cross-family verifier derivative has 1,201 positive + 1,201 negative examples and only **737 packed 512-token sequences**, or about **92 updates at two epochs**. It is too small and high-variance for a meaningful SFT, so no verifier SFT was submitted. Normalized verifier-prompt dedup now blocks punctuation-only duplicates (`bec11a4`). The 28-family train bank is actually 11,200 rows (400/family). Active isolated H100 rollout `686536` correctly covers its first 10,000 rows/25 sorted families and remains useful, but cannot by itself become training data. New tail rollout **`686564`** skips exactly 10,000 valid rows and covers the missing 1,200/three families; `686565` combines and globally deduplicates both outputs, then `686566` packs the complete 28-family derivative. The prior pending 25-family data/packing jobs `686540`/`686541` were canceled before start. `sft_verifier.sbatch` now refuses to launch unless the packing report is hash-bound to the current JSONL, has both labels, uses the requested length, and reaches **>=3,000 packed 512-token sequences**; that gate is meant to buy several hundred updates rather than repeat the 92-update failure. |
| **Recurrence ablation** | Mame `n_loop=1` job `686301` and `n_loop=2` job `686302` both completed cleanly for 800 matched updates. `n_loop=2` is mechanically stable but not promoted: final logged loss was essentially tied (**2.4890 vs 2.4899**) while time rose **886s -> 1466s** and steady throughput fell **472.7k -> 286.0k tok/s**. No capability gate was run, so recurrence stays off the flagship. |
| **Future handoff data stream** | **Fixed forward-only, never applied to active `685084`.** Prior checkpoints did not serialize `ShardLoader` state; every resumed job could recreate its stream from the same `DSEED=777`. `train.py` now records `data_stream_generation`/seed and resumes with a deterministic distinct stream generation. Tiny-train smoke verified generation **0 -> 1** and checkpoint metadata. This prevents repeated prefixes at the next natural handoff but does not claim exact cursor restoration for old chunks. |
| Preserved checkpoints (cluster) | Includes prior milestones plus **`best_step170000.pt`** copied from numbered `ckpt_0170000.pt`; both md5 **`7ad139b6b9b537a5a3e65978f8296419`**. |
| **Local DR backup (Mac)** | Includes prior milestones plus **`train/flagship_out/ckpt_0170000.pt`**, full+optimizer, locally and remotely md5 **`7ad139b6b9b537a5a3e65978f8296419`**. Next DR target: 180k. |
| **Large artifact transfer policy** | For big checkpoints/shards/uploads, prefer VPS-to-VPS or Newton-to-VPS staging when credentials/hosts are available; the VPS links have ~20 Gbit internet and should beat Mac↔Newton transfers. Still use `.part` files and md5/sha256 on both ends before trusting or deleting anything. |

**Checkpoints preserved so far:** every 10k through 50k; 60k is model-only because the trainer writes
`ckpt_final.pt` without optimizer state. `ckpt_0060000.pt` is local and cluster hash-matched. The 300k
extension resumes from `ckpt_0060000.pt` with fresh optimizer rewarmup, so no stale 59k momentum is used.
`ckpt_0059000.pt` is the local full+optimizer emergency fallback if a fresh-optimizer resume proves bad.

**Next actions in order:** (1) Watch `685084`: retain the normal ~154k tok/s band and expected 250-step
checkpoints, preserve/download 180k, and never interrupt a recovered isolated gnorm skip. (2) Retain V7
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

*Keep this file honest. When you hit a milestone, do the work, then come back and update §1 (LIVE
STATE) and any step that changed. A future agent — maybe you after a context reset — is relying on it.*
