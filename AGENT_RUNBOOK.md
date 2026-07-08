# AGENT RUNBOOK — Shohin autonomous custody

> **If you are a new agent taking over, read THIS FILE FIRST, top to bottom.** It is the single
> source of truth for the live training run and the standing directive. Everything you need to keep
> the model alive, hit the next milestone, and not break anything is here. Other docs
> (`MASTER_PLAN.md`, `DIVERGENCE_DIAGNOSIS.md`, `DATA.md`) are background/history; this file is the
> operational plan of record.
>
> **Last updated:** 2026-07-08 ~00:55 EDT (2-H100 canary succeeded; resumable job chain active). Keep the "LIVE STATE" section current
> every milestone — update it, don't let it rot.

---

## 0. The mission and the governing directive

**Mission:** Ship the best sub-200M-parameter (we're at ~135M) verifiable-reasoning language model of
2026 — math / code / logic. Concretely: **beat MobileLLM-R1-140M** on GSM8K / MATH-500 / HumanEval /
MBPP / logic. Reasoning specialist, verifiable-first. Data must be decontaminated, execution/answer-
verified, concise-CoT.

**Governing directive (verbatim, from the user — this overrides earlier plans):**

> "Forget the cluster that is not possible. So put all your effort into making our pretraining as
> perfect as it can be on the single GPU. Train it for weeks I dont mind. The only thing that matters
> is SoTA."

What that means operationally, and is now settled — **do not relitigate:**
- **No 8×H100.** The 8×H100 (`highgpu`, evc101–105) is permanently OFF the table (other users,
  no access). A measured **2×H100 single-node DDP path is now allowed** after the successful canary
  `681040`, but only with preserved global batch semantics (`NG=2 BS=16 ACC=8`) and checkpointed
  handoffs. Do not attempt broader multi-node/multi-GPU changes.
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

| Item | Value (as of 2026-07-08 ~00:55 EDT) |
|---|---|
| **60k pretrain job** | `680149`, name `shohin-flagship`, node **evc22**, **DONE** (`[done] 60000 steps in 112203s`) |
| **Extended pretrain job** | 1-GPU job `680992` was stopped at the 2-GPU transition after preserving `ckpt_0062000.pt`; current active continuation is short backfill job **`681083`**, name `shohin-flagship`, node **evc32**, RUNNING from `ckpt_0062000.pt` |
| Extended pretrain status | `681083` resumed at **step 62001** from full optimizer checkpoint `ckpt_0062000.pt`; latest seen **step 62030**, loss 1.7218, lr 0.0050. Follow-on queue is dependency-ordered: `681080` 1-GPU 2h chunk after `681083`, then `681078` 2-GPU 2h chunk after `681080`. |
| **SFT feedback job** | `681000`, name `shohin-sft`, node **evc43**, **DONE**; wrote `train/sft_out/sft_ep3.pt` |
| **Eval board job** | `681030`, name `shohin-eval`, **COMPLETED** on `sft_ep3.pt` (`N=100`, `K=1`): GSM8K 6/100, MATH500 0/100, HumanEval 4/164, MBPP 0/100. Treat as diagnostic/weak SFT, not a recipe win. |
| **2-H100 speed canary** | `681040`, name `shohin-ddp2-canary`, **COMPLETED cleanly** on evc42: resumed from `ckpt_0060000.pt`, `world=2`, loss in band, no DDP hang, ended at `61050` in 2093s with ~262k tok/s (~1.76x the 1-GPU ~149k tok/s). This validates the 2-H100 path. |
| 60k final loss | final logged band ~1.5-1.7; last logged step 59990 loss 1.6989, lr 0.0005 |
| 60k skips | **45 total**, stable/healthy |
| **Corpus-expansion job** | `680324` — **✅ DONE** (finished ~12:10) |
| finemath3 output | `artifacts/shards/finemath3/` — **✅ COMPLETE: 125 shards, exactly 25.0B tokens** (`manifest.json` present, 22 GB; 8,575 contaminated docs dropped vs evalgrams). **Included in the 300k relaunch SHARDS.** |
| SFT mix (Newton) | `artifacts/sft/sft_mix_core.jsonl` — **85,593 examples** at launch (OpenMath + rgym + code + latest verified teacher traces) |
| Local teacher distillers | HY3 and nemotron processes still alive and writing (`hy3_reasoning.jsonl` 15.6k+ rows; `hy3_reasoning_nemotron.jsonl` 1.07k+ rows). GLM remains paused after raw NVIDIA HTTP 429. |
| Preserved checkpoints (cluster) | `flagship_out/best_step{10000,12000,14000,16000,20000,30000,40000,50000}.pt` (+ early 4k/5k/6k) plus **`best_step60000.model.pt`**, numbered **`ckpt_0060000.pt`**, and **`best_step62000_pre2gpu.pt`** (`md5 e4f3de659effac5c6875c6ae17d6b544`) |
| **Local DR backup (Mac)** | **Post-60k downloaded/verified:** `train/flagship_out/ckpt_0061000.pt` (1.0 GB, full+optimizer extension checkpoint, md5 `28a18ebd7efc67cbbb72db6505493248`); `ckpt_0060000.pt` and hardlink `best_step60000.model.pt` (model-only 60k, md5 `d2fdf867bd49cf517b62364e152bffde`); `ckpt_0059000.pt` (full+optimizer fallback, md5 `0038df81be145cf4a4b0644e2dce284a`); `train/sft_out/sft_ep3.pt` (md5 `dda39ab36aa73bd6284b94d9fbf252e5`). Older full checkpoint `ckpt_0050000.pt` also remains local. Refresh again at 70k or after a promoted 2-GPU checkpoint. |
| **Large artifact transfer policy** | For big checkpoints/shards/uploads, prefer VPS-to-VPS or Newton-to-VPS staging when credentials/hosts are available; the VPS links have ~20 Gbit internet and should beat Mac↔Newton transfers. Still use `.part` files and md5/sha256 on both ends before trusting or deleting anything. |

**Checkpoints preserved so far:** every 10k through 50k; 60k is model-only because the trainer writes
`ckpt_final.pt` without optimizer state. `ckpt_0060000.pt` is local and cluster hash-matched. The 300k
extension resumes from `ckpt_0060000.pt` with fresh optimizer rewarmup, so no stale 59k momentum is used.
`ckpt_0059000.pt` is the local full+optimizer emergency fallback if a fresh-optimizer resume proves bad.

**Next actions in order:** (1) monitor `681083` through its 30-minute backfill window; it should keep
stepping from `ckpt_0062000.pt` and save every 100 steps. (2) Confirm `681080` starts after `681083`
and resumes from the newest checkpoint; it is a 1-GPU 2h chunk. (3) Confirm `681078` starts after
`681080`; it is the 2-H100 2h chunk using `train/jobs/flagship2.sbatch`, `NG=2 BS=16 ACC=8`, and should
show `world=2` plus ~260k tok/s. (4) If 2-GPU chunks keep scheduling cleanly, continue with short
checkpointed 2-GPU chunks or promote to longer 2-GPU walltimes when priority allows. (5) Refresh local
DR backup at the next useful numbered checkpoint (70k or after a stable promoted 2-GPU checkpoint).
(6) Do not rerun the same SFT recipe blindly -- the 60k SFT board is weak, so the next SFT should be a
measured data/prompt/format variant after inspecting generations.

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
- **`highgpu` / 8×H100 = no access.** Single GPU only.

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
| `hy3_reasoning*.jsonl` | ~7k and growing | Live verified teacher fleet output (hy3 / Nemotron / GLM / Claude / minimax). **Do not train directly from live files**; snapshot via `build_sft_mix.py`. |
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
(but we run single-GPU):
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
SFT), `eval_all.sbatch` (full board). All have GPU-free-wait + retry; flagship also does bad-node
exclusion + auto-requeue. `eval_all.sbatch` still lacks a CUDA-busy retry loop — **TODO: harden it**
(low priority; rerun manually if it hits the race).

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
   - **Overlap:** FineMath-4+ ⊆ FineMath-3+ (score≥4 ⊂ score≥3) — including both dirs double-weights the
     best math. Decide: deliberate upweight vs dedup.
   - **Mix ratio:** interleaved loader round-robins across shard *dirs*; adding finemath3 → 3 math : 1
     code = 75/25. Set intentionally (half the eval board is code). Could split code into more dirs or
     weight by duplicating dir entries.

**Time-box note:** the hy3 teacher is free **this week** — prioritize distillation volume while it lasts.

---

*Keep this file honest. When you hit a milestone, do the work, then come back and update §1 (LIVE
STATE) and any step that changed. A future agent — maybe you after a context reset — is relying on it.*
