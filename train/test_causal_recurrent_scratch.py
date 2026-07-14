#!/usr/bin/env python3
"""CPU contracts for the frozen-base causal recurrent scratch adapter."""

import torch

from causal_recurrent_scratch import CausalRecurrentScratch
from model import GPT, GPTConfig


def tiny_model():
    torch.manual_seed(17)
    cfg = GPTConfig(
        vocab_size=64, n_layer=4, n_head=4, n_kv_head=2,
        d_model=32, d_ff=64, seq_len=32, zloss=0.0,
    )
    return GPT(cfg).eval()


def main():
    base = tiny_model()
    adapter = CausalRecurrentScratch(base, layer=1, slots=3, width=12).eval()
    ids = torch.tensor([
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
    ], dtype=torch.long)

    # A disabled state path and the initialized zero gate are exactly the base.
    base_logits, _ = base(ids)
    disabled = adapter.forward_ids(ids, prompt_len=4, steps=3, state_mode="disabled")
    zero_gate = adapter.forward_ids(ids, prompt_len=4, steps=3)
    torch.testing.assert_close(disabled, base_logits, atol=0.0, rtol=0.0)
    torch.testing.assert_close(zero_gate, base_logits, atol=0.0, rtol=0.0)

    assert all(not parameter.requires_grad for parameter in adapter.model.parameters())
    assert adapter.adapter_num_params() > 0
    assert adapter.adapter_num_params() < base.num_params() // 2

    # Prompt-only causality: changing teacher-forced answer IDs cannot alter the
    # scratch state constructed from the fixed prompt prefix.
    adapter.readout_gate.data.fill_(0.7)
    changed = ids.clone()
    changed[:, 4:] = torch.tensor([[20, 21], [22, 23]])
    _, state_a = adapter.forward_ids(ids, prompt_len=4, steps=3, return_state=True)
    _, state_b = adapter.forward_ids(changed, prompt_len=4, steps=3, return_state=True)
    torch.testing.assert_close(state_a, state_b, atol=1e-6, rtol=1e-6)

    # State interventions share one decoder and are behaviorally distinct.
    normal = adapter.forward_ids(ids, prompt_len=4, steps=3, state_mode="normal")
    zero = adapter.forward_ids(ids, prompt_len=4, steps=3, state_mode="zero")
    shuffled = adapter.forward_ids(ids, prompt_len=4, steps=3, state_mode="shuffled")
    override = adapter.forward_ids(
        ids, prompt_len=4, steps=3, state_mode="override",
        state_override=torch.zeros_like(state_a),
    )
    assert not torch.equal(normal, disabled)
    torch.testing.assert_close(zero, override, atol=1e-6, rtol=1e-6)
    assert not torch.equal(normal, shuffled)

    lower = adapter._lower(ids)
    recurrent, trajectory = adapter.compute_scratch(
        lower[:, :4].detach(), 3, recurrent=True, return_trajectory=True,
    )
    reset, reset_trajectory = adapter.compute_scratch(
        lower[:, :4].detach(), 3, recurrent=False, return_trajectory=True,
    )
    assert len(trajectory) == 3
    torch.testing.assert_close(recurrent, trajectory[-1])
    for reset_step in reset_trajectory:
        torch.testing.assert_close(reset, reset_step, atol=1e-6, rtol=1e-6)
    assert not torch.equal(recurrent, reset)

    # The selective verbalizable-workspace readout is finite, sparse by
    # construction, and still starts at the exact disabled model.
    workspace = CausalRecurrentScratch(
        tiny_model(), layer=2, slots=2, width=10,
        workspace_topk=4, workspace_temperature=0.2,
    ).eval()
    workspace_base, _ = workspace.model(ids)
    workspace_zero_gate = workspace.forward_ids(ids, prompt_len=4, steps=2)
    torch.testing.assert_close(workspace_zero_gate, workspace_base, atol=0.0, rtol=0.0)
    workspace.readout_gate.data.fill_(0.5)
    workspace_logits = workspace.forward_ids(ids, prompt_len=4, steps=2)
    assert torch.isfinite(workspace_logits).all()
    assert not torch.equal(workspace_logits, workspace_base)

    # Only adapter parameters receive gradients under answer supervision.
    prompts, answers = ids[:, :4], ids[:, 4:]
    _, loss, state, targets = adapter.supervised_loss(prompts, answers, eos_id=0, steps=3)
    assert torch.isfinite(loss) and state.shape == (2, 3, 12)
    assert targets.shape == ids.shape
    loss.backward()
    assert any(parameter.grad is not None for parameter in adapter.adapter_parameters())
    assert all(parameter.grad is None for parameter in adapter.model.parameters())

    # Invalid controls fail closed.
    try:
        adapter.forward_ids(ids[:1], prompt_len=4, steps=2, state_mode="shuffled")
    except ValueError:
        pass
    else:
        raise AssertionError("single-example shuffled control must fail")

    print("causal recurrent scratch tests passed")


if __name__ == "__main__":
    main()
