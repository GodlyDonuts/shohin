#!/usr/bin/env python
"""Shohin code-reasoning eval — HumanEval & MBPP with real test execution.

Generates a solution, assembles the full program with its official tests, and runs it in a
sandboxed subprocess (timeout, isolated temp file). Scores pass@1 (greedy) or pass@k (any of k
samples passes). Executes UNTRUSTED generated code — kept to a subprocess with a hard timeout;
run only on a machine you're comfortable executing model output on.

  python eval_code.py --ckpt flagship_out/best_step10000.model.pt --tokenizer ../artifacts/shohin-tok-32k.json \\
      --task humaneval --data ../artifacts/evals/humaneval_full.jsonl --n 20 --k 1
"""
import argparse, json, os, re, subprocess, sys, tempfile
import torch
from tokenizers import Tokenizer
from model import GPT, GPTConfig

# HumanEval completes ONE function body -> cut anything that starts a new top-level construct.
HE_STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n@", "\nassert ", "\n#", "\nQuestion:"]
# MBPP writes a whole solution (may define helper functions) -> only cut at test/prompt boundaries.
MBPP_STOPS = ["\nassert ", "\n[DONE]", "\nQuestion:", "\nif __name__", "\n>>>", "\nprint("]


@torch.no_grad()
def gen_code(model, tok, prompt, device, max_new=320, temp=0.0, top_k=40):
    cap = model.cfg.seq_len
    ids = tok.encode(prompt).ids[-cap:]
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=("cuda" in str(device)))
    with ac:
        logits, cache = model(torch.tensor([ids], device=device), return_cache=True, pos=0)
    pos, gen = len(ids), []
    for _ in range(max_new):
        lg = logits[0, -1]
        if temp and temp > 0:
            lg = lg / temp
            if top_k:
                v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                lg = lg.masked_fill(lg < v[-1], float("-inf"))
            nxt = int(torch.multinomial(torch.softmax(lg.float(), -1), 1))
        else:
            nxt = int(lg.argmax())
        gen.append(nxt)
        if pos >= cap:
            break
        with ac:
            logits, cache = model(torch.tensor([[nxt]], device=device), cache=cache, pos=pos, return_cache=True)
        pos += 1
    return tok.decode(gen)


def truncate(completion, stops):
    for s in stops:
        i = completion.find(s)
        if i != -1:
            completion = completion[:i]
    return completion


def run_program(program, timeout=8):
    """True iff the program runs to completion (all asserts pass) with exit code 0."""
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        os.write(fd, program.encode()); os.close(fd)
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout,
                           text=True, cwd=tempfile.gettempdir())
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except OSError: pass


def he_program(row, completion):
    return row["prompt"] + truncate(completion, HE_STOPS) + "\n\n" + row["test"] + f"\ncheck({row['entry_point']})\n"


def he_instruction_prompt(row):
    return (f"Question: {row['prompt']}\n"
            "Write only the complete Python function that solves this task.\nAnswer:")


def he_instruction_program(row, completion):
    code = truncate(completion, HE_STOPS).strip()
    # Instruction-tuned models commonly emit the full function; retain the
    # standard continuation behavior only when they emit a body fragment.
    program = code if re.search(r"(?m)^\s*def\s+", code) else row["prompt"] + code
    return program + "\n\n" + row["test"] + f"\ncheck({row['entry_point']})\n"


def mbpp_prompt(row):
    tests = "\n".join(row["test_list"])
    return (f"You are an expert Python programmer. Write a function for this task:\n{row['text']}\n"
            f"Your function must pass these tests:\n{tests}\n[BEGIN]\n")


def mbpp_program(row, completion):
    code = truncate(completion, MBPP_STOPS)
    setup = row.get("test_setup_code", "") or ""
    tests = "\n".join(row["test_list"])
    return code + "\n" + setup + "\n" + tests + "\n"


def mbpp_instruction_prompt(row):
    return f"Question: {row['text']}\nWrite only Python code.\nAnswer:"


def solve_pass(model, tok, row, device, task, k, temp, mkprompt, mkprog):
    prompt = mkprompt(row)
    for _ in range(max(1, k)):
        comp = gen_code(model, tok, prompt, device, temp=(0.0 if k <= 1 else temp))
        if run_program(mkprog(row, comp)):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--task", default="humaneval", choices=["humaneval", "mbpp"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--prompt-style", choices=["completion", "instruction"], default="completion",
                    help="completion is the official default; instruction is an SFT-contract diagnostic")
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[eval] device={device} task={a.task} k={a.k} prompt_style={a.prompt_style}", file=sys.stderr)
    ck = torch.load(a.ckpt, map_location="cpu")
    model = GPT(GPTConfig(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])
    tok = Tokenizer.from_file(a.tokenizer)

    if a.task == "humaneval":
        mkprompt, mkprog = ((lambda r: r["prompt"]), he_program) if a.prompt_style == "completion" \
            else (he_instruction_prompt, he_instruction_program)
    else:
        mkprompt, mkprog = (mbpp_prompt, mbpp_program) if a.prompt_style == "completion" \
            else (mbpp_instruction_prompt, mbpp_program)
    rows = [json.loads(l) for l in open(a.data)][:a.n]
    passed = 0
    for i, r in enumerate(rows):
        ok = solve_pass(model, tok, r, device, a.task, a.k, a.temp, mkprompt, mkprog)
        passed += ok
        if i < 5:
            print(f"[{i}] {r.get('task_id','?')} pass={ok}", file=sys.stderr)
    acc = passed / max(len(rows), 1)
    tag = f"pass@{a.k}" if a.k > 1 else "pass@1"
    print(f"{a.task}  {tag}  ckpt={a.ckpt}  step={ck.get('step')}  {passed}/{len(rows)} = {acc:.1%}")


if __name__ == "__main__":
    main()
