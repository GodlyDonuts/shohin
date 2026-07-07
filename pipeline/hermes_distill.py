#!/usr/bin/env python
"""Rejection-sampled reasoning-trace distillation via the Hermes CLI (free `tencent/hy3:free`).

Our thesis (DATA.md): the decisive lever is SHORT-CoT distillation from a strong teacher, keeping
ONLY correct (verifiable) traces. This harness does exactly that, over a problem bank with known
answers:

  problem (+gold answer)  ->  hermes -z <prompt> -m hy3  ->  extract answer  ->  verify vs gold
                                                                                    |
                                                    keep ONLY if correct (rejection sampling)  -> JSONL

Output rows match the curated SFT format ({question, response, answer, source}) so they drop straight
into the SFT mix. Parallel (ThreadPool over subprocesses), resumable (skips questions already emitted),
and it logs kept/wrong/error counts + yield so quality is auditable.

  python hermes_distill.py --problems gsm8k_train.jsonl --q-field question --gold-field answer \
      --answer-type gsm8k --out ../artifacts/sft/hy3_gsm8k.jsonl --concurrency 16 --limit 2000

Teacher output is re-tokenized with our own 32k vocab at train time, so the teacher's vocab is
irrelevant (trace distillation, not logit-KD).
"""
import argparse, hashlib, json, os, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- answer extraction (mirrors train/eval_suite.py so verification == eval scoring) --------------

def _clean_num(s):
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    m = re.search(r"-?\d+(?:/\d+)?(?:\.\d+)?", s)
    return m.group(0) if m else None


def extract_gsm8k(text):
    m = re.findall(r"answer is\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return _clean_num(m[-1])
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return _clean_num(nums[-1]) if nums else None


def extract_boxed(text):
    i = text.rfind(r"\boxed")
    if i >= 0:
        j = text.find("{", i)
        if j >= 0:
            depth = 0
            for k in range(j, len(text)):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[j + 1:k].strip()
    m = re.findall(r"answer is\s*\$?\s*([^\n.$]+)", text)
    return m[-1].strip() if m else None


def _to_float(x):
    try:
        if "/" in str(x):
            a, b = str(x).split("/")
            return float(a) / float(b)
        return float(x)
    except Exception:
        return None


def _norm_txt(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _norm_math(s):
    """Normalize a math answer for comparison: \\frac{a}{b}->(a)/(b), strip LaTeX cruft/units."""
    s = str(s)
    s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)   # \frac{a}{b}
    s = re.sub(r"\\text\s*\{[^{}]*\}", "", s)
    s = re.sub(r"\\mbox\s*\{[^{}]*\}", "", s)
    s = re.sub(r"\\(left|right|displaystyle|mathrm|,|;|:|!|\s)", "", s)
    s = s.replace("^\\circ", "").replace("\\%", "").replace("\\$", "")
    s = s.replace("\\", "").replace("$", "").replace(" ", "").rstrip(".").strip().lower()
    return s


def _eval_math(s):
    """Best-effort numeric value of a (possibly LaTeX) math answer; None if not numeric."""
    n = _norm_math(s)
    v = _to_float(n)
    if v is not None:
        return v
    m = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?/\(?(-?\d+(?:\.\d+)?)\)?", n)   # a/b or (a)/(b)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except Exception:
            return None
    return None


def verify(gen_text, gold, answer_type):
    """Return (is_correct, extracted_answer). Types: gsm8k (numeric), boxed (math),
    mc (multiple-choice letter A-E), exact (normalized text, e.g. yes/no)."""
    if answer_type == "gsm8k":
        pred = extract_gsm8k(gen_text)
        if pred is None:
            return False, None
        pf, gf = _to_float(pred), _to_float(_clean_num(str(gold)))
        if pf is not None and gf is not None:
            return abs(pf - gf) < 1e-4, pred
        return str(pred).strip() == str(gold).strip(), pred
    if answer_type == "boxed":
        pred = extract_boxed(gen_text)
        if pred is None:
            return False, None
        gb = extract_boxed(str(gold)) or str(gold)
        if _norm_math(pred) == _norm_math(gb):                 # LaTeX-aware string match
            return True, pred
        pv, gv = _eval_math(pred), _eval_math(gb)              # numeric equivalence (handles fractions)
        if pv is not None and gv is not None:
            return abs(pv - gv) < 1e-4, pred
        return False, pred
    if answer_type == "mc":
        m = re.findall(r"answer is\s*[:\-]?\s*\(?([A-Ea-e])\b", gen_text)
        pred = m[-1].upper() if m else None
        if pred is None:                                   # fallback: last standalone A-E
            m2 = re.findall(r"\b([A-E])\b", gen_text)
            pred = m2[-1] if m2 else None
        if pred is None:
            return False, None
        return pred == str(gold).strip().upper(), pred
    if answer_type == "exact":
        m = re.findall(r"answer is\s*[:\-]?\s*([^\n.]+)", gen_text)
        pred = m[-1].strip() if m else None
        if pred is None:
            return False, None
        return _norm_txt(pred) == _norm_txt(gold), pred
    raise ValueError(f"unknown answer_type {answer_type}")


PROMPT_TMPL = (
    "Solve this problem with concise, correct step-by-step reasoning. Keep the steps brief and to the "
    "point (no rambling). End with a line exactly in the form \"The answer is X.\" where X is the final "
    "answer.\n\nProblem: {q}"
)


def call_hermes(prompt, model, timeout):
    try:
        r = subprocess.run(["hermes", "-z", prompt, "-m", model],
                           capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "").strip()
        return out if out else None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


import urllib.request, urllib.error

# provider -> (base_url, api-key env var). All OpenAI-compatible /chat/completions.
BACKENDS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "nvidia":     ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
}


def call_openai_compat(prompt, model, timeout, base_url, api_key,
                       temperature=0.4, retries=4, max_tokens=1200):
    """OpenAI-compatible HTTP call (no per-call process spawn -> high concurrency on a small box).
    Uses the `content` field (we want a CONCISE trace, not a long <think>; max_tokens bounds it, and
    reasoning models keep their long CoT in `reasoning_content`, which we deliberately ignore).
    Backs off on 429/503; returns None on hard failure, "__RATELIMIT__" if 429 retries exhaust."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({"model": model, "temperature": temperature, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for att in range(retries):
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            out = (d["choices"][0]["message"].get("content") or "").strip()
            return out if out else None
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):                  # rate-limited / temporarily unavailable
                time.sleep(2 * (att + 1) + (att * att))
                continue
            return None
        except Exception:
            return None
    return "__RATELIMIT__"


def qhash(q):
    return hashlib.sha1(q.strip().encode("utf-8", "ignore")).hexdigest()[:16]


def load_grams(path):
    if not path or not os.path.exists(path):
        return None, None
    import pickle
    d = pickle.load(open(path, "rb"))
    return d["grams"], d["n"]


def contaminated(text, grams, n):
    if grams is None:
        return False
    w = re.findall(r"\w+", text.lower())
    return any(" ".join(w[i:i + n]) in grams for i in range(len(w) - n + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True, help="JSONL problem bank (train split; has gold answer)")
    ap.add_argument("--q-field", default="question")
    ap.add_argument("--gold-field", default="answer")
    ap.add_argument("--answer-type", choices=["gsm8k", "boxed"], default="gsm8k")
    ap.add_argument("--gold-is-gsm8k-format", action="store_true",
                    help="gold field is the raw GSM8K '#### N' solution; extract N from it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="tencent/hy3:free")
    ap.add_argument("--backend", choices=["hermes", "openrouter", "nvidia"], default="hermes",
                    help="hermes = CLI subprocess (~0.5/s); openrouter/nvidia = direct OpenAI-compatible "
                         "HTTP (reads OPENROUTER_API_KEY / NVIDIA_API_KEY env; high concurrency)")
    ap.add_argument("--max-tokens", type=int, default=1200, help="cap teacher output -> concise traces")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--timeout", type=int, default=90, help="per-call seconds")
    ap.add_argument("--max-resp-chars", type=int, default=2000, help="reject overly long traces")
    ap.add_argument("--decontam-grams", default=None)
    ap.add_argument("--source", default="hy3_distill")
    a = ap.parse_args()

    grams, gram_n = load_grams(a.decontam_grams)
    base_url, api_key = None, ""
    if a.backend in BACKENDS:
        base_url, key_env = BACKENDS[a.backend]
        api_key = os.environ.get(key_env, "")
        if not api_key:
            raise SystemExit(f"backend={a.backend} but {key_env} not set in env")

    # resume: collect already-emitted question hashes
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(qhash(json.loads(line)["question"]))
            except Exception:
                pass
    if done:
        print(f"[resume] {len(done)} already done in {a.out}", flush=True)

    probs = []
    for line in open(a.problems):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        q = str(row.get(a.q_field, "") or row.get("question", "")).strip()
        gold = row.get(a.gold_field, row.get("gold", ""))
        if a.gold_is_gsm8k_format:
            m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", str(gold))
            gold = _clean_num(m.group(1)) if m else None
        # a normalized problem bank may carry a prebuilt teacher prompt + per-row answer_type
        atype = row.get("answer_type") or a.answer_type
        prompt = row.get("prompt") or PROMPT_TMPL.format(q=q)
        if not q or gold is None or gold == "":
            continue
        if qhash(q) in done:
            continue
        probs.append({"q": q, "prompt": prompt, "gold": gold, "atype": atype})
    if a.limit:
        probs = probs[:a.limit]
    print(f"[distill] {len(probs)} problems to attempt | model={a.model} conc={a.concurrency}", flush=True)

    lock = threading.Lock()
    stats = dict(attempted=0, kept=0, wrong=0, err=0, contam=0, toolong=0)
    t0 = time.time()
    fout = open(a.out, "a")

    def work(item):
        q, gold, prompt, atype = item["q"], item["gold"], item["prompt"], item["atype"]
        if a.backend in BACKENDS:
            gen = call_openai_compat(prompt, a.model, a.timeout, base_url, api_key,
                                     max_tokens=a.max_tokens)
        else:
            gen = call_hermes(prompt, a.model, a.timeout)
        with lock:
            stats["attempted"] += 1
            n = stats["attempted"]
        if gen is None or gen == "__RATELIMIT__":
            with lock:
                stats["err"] += 1
                if gen == "__RATELIMIT__":
                    stats["ratelimit"] = stats.get("ratelimit", 0) + 1
            return
        if len(gen) > a.max_resp_chars:
            with lock:
                stats["toolong"] += 1
            return
        ok, pred = verify(gen, gold, atype)
        if not ok:
            with lock:
                stats["wrong"] += 1
            return
        if contaminated(q + " " + gen, grams, gram_n):
            with lock:
                stats["contam"] += 1
            return
        row = dict(question=q, response=gen, answer=str(gold), source=a.source)
        with lock:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            stats["kept"] += 1
            if n % 25 == 0 or n <= 5:
                dt = time.time() - t0
                rate = n / dt if dt else 0
                y = 100 * stats["kept"] / max(1, stats["attempted"])
                print(f"[{n}/{len(probs)}] kept={stats['kept']} wrong={stats['wrong']} "
                      f"err={stats['err']} yield={y:.0f}% {rate:.1f}/s", flush=True)

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(work, it) for it in probs]
        for _ in as_completed(futs):
            pass
    fout.close()
    dt = time.time() - t0
    print(f"[done] attempted={stats['attempted']} kept={stats['kept']} wrong={stats['wrong']} "
          f"err={stats['err']} contam={stats['contam']} toolong={stats['toolong']} "
          f"yield={100*stats['kept']/max(1,stats['attempted']):.1f}% in {dt:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
