"""Muon optimizer (Keller Jordan / modded-nanoGPT) — momentum SGD with Newton-Schulz
orthogonalization of the update, for 2D hidden matrices. Embeddings / head / norms use AdamW.
"""
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Orthogonalize G via a quintic Newton-Schulz iteration (bf16, coeffs from modded-nanoGPT)."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            mom, nes, ns, lr = grp["momentum"], grp["nesterov"], grp["ns_steps"], grp["lr"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if nes else buf
                g = zeropower_via_newtonschulz5(g, ns)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5   # modded-nanoGPT update scaling
                p.add_(g.type_as(p), alpha=-lr * scale)


def split_params(model):
    """Muon for 2D block matrices; AdamW for embeddings / lm_head / norms / 1D."""
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "tok" not in name and "head" not in name:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw
