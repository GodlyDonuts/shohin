#!/usr/bin/env python
"""Shohin SFT — completion-masked fine-tuning on (question, reasoning) pairs.

Loss is computed ONLY on the answer tokens (prompt tokens are masked with ignore_index=-1), so the
model learns to *produce* step-by-step reasoning, not to model the questions. Examples are packed to
seq_len for efficiency (single-GPU friendly). The prompt format matches eval_suite.py
("Question: ...\\nAnswer: ...") so the learned behavior transfers directly to the benchmarks.

  python sft.py --init flagship_out/best_step10000.model.pt --data ../artifacts/sft/math.jsonl \\
      --tokenizer ../artifacts/shohin-tok-32k.json --epochs 3 --out sft_out
"""
import argparse, glob, json, math, os, time
import numpy as np
import torch
from tokenizers import Tokenizer
from model import GPT, GPTConfig
from muon import Muon, split_params


def build_packed(data_paths, tok, seq_len, q_fields, r_fields, eos_id, max_examples=0,
                 group_field=None, prompt_override_field=None):
    """Tokenize (question, response) rows -> packed sequences with a completion-only loss mask.
    Returns (X[int64 N,seq_len], Y[int64 N,seq_len]) where Y is -1 on prompt/pad (ignored)."""
    grouped_buffers = {}            # group -> (token ids, loss-mask)
    n_ex = n_tok = n_ans = 0
    for p in data_paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = next((r[f] for f in q_fields if r.get(f)), None)
            a = next((r[f] for f in r_fields if r.get(f)), None)
            if not q or not a:
                continue
            group = str(r.get(group_field) or "default") if group_field else "default"
            buf_x, buf_m = grouped_buffers.setdefault(group, ([], []))
            prompt_override = str(r.get(prompt_override_field) or "") if prompt_override_field else ""
            prompt = prompt_override or f"Question: {q}\nAnswer:"
            # Completion-form code must retain the indentation beginning its
            # function body. Standard answer-form SFT keeps its established
            # trimmed response behavior.
            answer = str(a).rstrip() if prompt_override else str(a).strip()
            sep = "" if prompt_override or prompt.endswith((" ", "\n", "\t")) else " "
            full = prompt + sep + answer
            pids = tok.encode(prompt).ids
            fids = tok.encode(full).ids
            if len(fids) >= seq_len:            # skip pathologically long examples
                continue
            fids = fids + [eos_id]
            mask = [0] * len(pids) + [1] * (len(fids) - len(pids))  # train only on the answer + eos
            buf_x.extend(fids)
            buf_m.extend(mask)
            n_ex += 1
            n_tok += len(fids)
            n_ans += sum(mask)
            if max_examples and n_ex >= max_examples:
                break
    # slice into seq_len+1 windows; target = next token, -1 where the next token is masked
    X, Y, groups = [], [], []
    step = seq_len
    for group, (buf_x, buf_m) in grouped_buffers.items():
        for i in range(0, len(buf_x) - seq_len - 1, step):
            xi = buf_x[i:i + seq_len]
            yi = [buf_x[j + 1] if buf_m[j + 1] else -1 for j in range(i, i + seq_len)]
            X.append(xi); Y.append(yi); groups.append(group)
    print(f"[sft-data] {n_ex:,} examples, {n_tok:,} tokens ({n_ans:,} answer tokens = "
          f"{100*n_ans/max(n_tok,1):.0f}% trained), {len(X):,} packed seqs of {seq_len}", flush=True)
    if group_field:
        counts = {group: groups.count(group) for group in sorted(set(groups))}
        print(f"[sft-data] packed groups={counts}", flush=True)
    return np.array(X, dtype=np.int64), np.array(Y, dtype=np.int64), np.array(groups, dtype=object)


def parse_sample_weights(items):
    weights = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"invalid --sample-weights item {item!r}; expected group=weight")
        weight = float(value)
        if weight <= 0:
            raise ValueError(f"sample weight must be positive: {item!r}")
        if key in weights:
            raise ValueError(f"duplicate sample-weight group: {key}")
        weights[key] = weight
    return weights


def weighted_epoch_order(rng, groups, batch_size, weights):
    """Sample packed sequences by immutable group labels without duplicating data files."""
    group_to_indices = {group: np.flatnonzero(groups == group) for group in sorted(set(groups))}
    missing = sorted(set(weights) - set(group_to_indices))
    if missing:
        raise ValueError(f"sample weights name absent groups: {', '.join(missing)}")
    names = list(weights)
    probs = np.array([weights[name] for name in names], dtype=np.float64)
    probs /= probs.sum()
    count = (len(groups) // batch_size) * batch_size
    chosen = rng.choice(len(names), size=count, p=probs)
    order = np.empty(count, dtype=np.int64)
    requested = {}
    for i, name in enumerate(names):
        slots = np.flatnonzero(chosen == i)
        requested[name] = int(len(slots))
        if len(slots):
            order[slots] = rng.choice(group_to_indices[name], size=len(slots), replace=True)
    return order, requested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="pretrained checkpoint to fine-tune from")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--q-fields", nargs="+", default=["question", "problem", "prompt", "instruction"])
    ap.add_argument("--r-fields", nargs="+", default=["response", "answer", "solution", "completion", "output"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-muon", type=float, default=2e-3)   # gentler than pretrain (fine-tune)
    ap.add_argument("--lr-adam", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--pack-len", type=int, default=0, help="pack sequence length (0 = model seq_len); shorter = less memory")
    ap.add_argument("--group-field", default=None,
                    help="optional immutable row field used to keep packed sequences by group")
    ap.add_argument("--prompt-override-field", default=None,
                    help="optional row field containing an exact completion prompt (for code completion SFT)")
    ap.add_argument("--sample-weights", nargs="*", default=[], metavar="GROUP=WEIGHT",
                    help="weighted per-epoch sampling over --group-field values; examples are sampled with replacement")
    ap.add_argument("--eos", default="<|endoftext|>")
    ap.add_argument("--out", default="sft_out")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")
    os.makedirs(a.out, exist_ok=True)
    tok = Tokenizer.from_file(a.tokenizer)
    eos_id = tok.token_to_id(a.eos)

    ck = torch.load(a.init, map_location="cpu")
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    print(f"[sft] init from {a.init} (step {ck.get('step')}), params {model.num_params()/1e6:.1f}M, device {device}", flush=True)

    paths = []
    for d in a.data:
        paths += sorted(glob.glob(d)) if any(c in d for c in "*?[") else [d]
    pack_len = a.pack_len or cfg.seq_len
    X, Y, groups = build_packed(paths, tok, pack_len, a.q_fields, a.r_fields, eos_id,
                                a.max_examples, group_field=a.group_field,
                                prompt_override_field=a.prompt_override_field)
    N = len(X)
    if N == 0:
        print("[sft] no packed sequences — check data/fields"); return

    weights = parse_sample_weights(a.sample_weights)
    if weights and not a.group_field:
        raise ValueError("--sample-weights requires --group-field")
    raw = model
    if a.compile:
        model = torch.compile(model)
    muon_p, adam_p = split_params(raw)
    opt_muon = Muon(muon_p, lr=a.lr_muon)
    opt_adam = torch.optim.AdamW(adam_p, lr=a.lr_adam, betas=(0.9, 0.95), weight_decay=0.0)
    total_steps = a.epochs * math.ceil(N / a.batch_size)

    def lr_at(step):
        if step < a.warmup:
            return step / max(1, a.warmup)
        r = (step - a.warmup) / max(1, total_steps - a.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * r))   # cosine decay to 0.1

    rng = np.random.default_rng(1337)
    t0, step = time.time(), 0
    for ep in range(a.epochs):
        if weights:
            order, requested = weighted_epoch_order(rng, groups, a.batch_size, weights)
            print(f"[sft-data] epoch {ep} weighted samples={requested}", flush=True)
        else:
            order = rng.permutation(N)
        for bi in range(0, len(order) - a.batch_size + 1, a.batch_size):
            idx = order[bi:bi + a.batch_size]
            x = torch.from_numpy(X[idx]).to(device)
            y = torch.from_numpy(Y[idx]).to(device)
            sc = lr_at(step)
            for g in opt_muon.param_groups:
                g["lr"] = a.lr_muon * sc
            for g in opt_adam.param_groups:
                g["lr"] = a.lr_adam * sc
            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=("cuda" in str(device))):
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), a.clip)
            opt_muon.step()
            opt_adam.step()
            if step % a.log_every == 0:
                print(f"epoch {ep} step {step}/{total_steps} loss {loss.item():.4f} "
                      f"lr {a.lr_muon*sc:.5f} {time.time()-t0:.0f}s", flush=True)
            step += 1
        torch.save(dict(model=raw.state_dict(), cfg=cfg.__dict__, step=f"sft_ep{ep+1}"),
                   os.path.join(a.out, f"sft_ep{ep+1}.pt"))
        print(f"[sft] saved epoch {ep+1}", flush=True)
    print(f"[sft] done {total_steps} steps in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
