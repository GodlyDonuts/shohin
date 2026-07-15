import unittest

import torch

from counterfactual_cursor_action import (
    EXECUTE,
    HALT,
    HALT_PENDING,
    SELECT,
    CursorQSidecar,
    DecodeState,
    EventTokenManifest,
    FrozenBaseCursorSelector,
    advance_state,
    centered_cursor_bits,
    initial_state,
    teacher_forced_state_grid,
)
from model import GPT, GPTConfig


class CounterfactualCursorActionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.cfg = GPTConfig(
            vocab_size=64, n_layer=2, n_head=2, n_kv_head=1,
            d_model=16, d_ff=32, seq_len=32, qk_norm=True,
        )
        self.manifest = EventTokenManifest(
            operation_ids=(10, 11, 12, 13), commit_id=20, done_id=21, eos_id=0,
        )

    def test_strict_old_state_load_and_zero_init_parity(self):
        base = GPT(self.cfg).eval()
        old_state = {key: value.clone() for key, value in base.state_dict().items()}
        reloaded = GPT(self.cfg).eval()
        reloaded.load_state_dict(old_state, strict=True)
        selector = FrozenBaseCursorSelector(reloaded)
        ids = torch.randint(0, self.cfg.vocab_size, (3, 9))
        cursor = torch.tensor([0, 2, 4])
        expected, _ = reloaded(ids)
        observed, _ = selector(ids, cursor)
        self.assertTrue(torch.equal(expected, observed))
        self.assertEqual(selector.sidecar.metadata()["parameters"], 24)
        self.assertEqual(CursorQSidecar(64).metadata()["parameters"], 192)

    def test_centered_bits_and_phase_gating(self):
        cursor = torch.arange(5, dtype=torch.long)
        bits = centered_cursor_bits(cursor)
        self.assertEqual(bits.tolist(), [
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ])
        sidecar = CursorQSidecar(64)
        with torch.no_grad():
            sidecar.projection.weight.fill_(1.0)
        grid = cursor[:, None].expand(-1, 2)
        mask = torch.tensor([[True, False]]).expand(5, -1)
        delta = sidecar(grid, mask)
        self.assertTrue(torch.equal(delta[:, 1], torch.zeros_like(delta[:, 1])))
        self.assertFalse(torch.equal(delta[:, 0], torch.zeros_like(delta[:, 0])))

    def test_event_fsm_and_batch_independence(self):
        state = DecodeState(
            cursor=torch.tensor([0, 4, 4, 2]),
            phase=torch.tensor([SELECT, SELECT, HALT_PENDING, EXECUTE]),
        )
        tokens = torch.tensor([10, 10, 0, 20])
        next_state, invalid = advance_state(state, tokens, self.manifest)
        self.assertEqual(next_state.cursor.tolist(), [1, 4, 4, 2])
        self.assertEqual(next_state.phase.tolist(), [EXECUTE, SELECT, HALT, SELECT])
        self.assertEqual(invalid.tolist(), [False, True, False, False])

        premature, invalid = advance_state(initial_state(1), torch.tensor([21]), self.manifest)
        self.assertEqual(premature.phase.item(), SELECT)
        self.assertTrue(invalid.item())
        pending, invalid = advance_state(
            DecodeState(torch.tensor([4]), torch.tensor([SELECT])),
            torch.tensor([21]), self.manifest,
        )
        self.assertEqual(pending.phase.item(), HALT_PENDING)
        self.assertFalse(invalid.item())

    def test_teacher_forced_alignment_ignores_prompt_events(self):
        ids = torch.tensor([[10, 20, 7, 10, 9, 20, 11, 20, 21]])
        cursor, select, violations = teacher_forced_state_grid(
            ids, torch.tensor([2]), self.manifest,
        )
        self.assertEqual(cursor.tolist(), [[0, 0, 0, 1, 1, 1, 2, 2, 2]])
        self.assertEqual(select.tolist(), [[False, False, True, False, False, True, False, True, True]])
        self.assertTrue(violations[0, 8].item())

    def test_only_sidecar_receives_gradients(self):
        model = FrozenBaseCursorSelector(GPT(self.cfg))
        ids = torch.randint(0, self.cfg.vocab_size, (2, 8))
        targets = torch.full_like(ids, -1)
        targets[:, -1] = torch.tensor([10, 11])
        _, loss = model(ids, torch.tensor([0, 1]), targets=targets)
        loss.backward()
        self.assertIsNotNone(model.sidecar.projection.weight.grad)
        self.assertTrue(torch.isfinite(model.sidecar.projection.weight.grad).all())
        self.assertTrue(all(parameter.grad is None for parameter in model.base.parameters()))

    def test_cached_decode_matches_full_replay(self):
        model = GPT(self.cfg).eval()
        sidecar = CursorQSidecar(self.cfg.d_model // self.cfg.n_head)
        with torch.no_grad():
            sidecar.projection.weight.normal_(0.0, 0.1)
        prompt = torch.tensor([[5, 6, 7]])
        prompt_cursor = torch.zeros_like(prompt)
        prompt_select = torch.tensor([[False, False, True]])
        prompt_delta = sidecar(prompt_cursor, prompt_select)
        logits, cache = model(
            prompt, return_cache=True, q_delta=prompt_delta,
            q_delta_layer=-1, q_delta_head=0,
        )
        state = initial_state(1)
        full = prompt.clone()
        prompt_last = torch.tensor([2])
        for token in (10, 9, 20, 11):
            emitted = torch.tensor([token])
            state, _ = advance_state(state, emitted, self.manifest)
            active = state.phase.eq(SELECT)
            delta = sidecar(state.cursor[:, None], active[:, None])
            logits, cache = model(
                emitted[:, None], cache=cache, pos=full.shape[1], return_cache=True,
                q_delta=delta, q_delta_layer=-1, q_delta_head=0,
            )
            full = torch.cat((full, emitted[:, None]), dim=1)
            cursor_grid, select_grid, _ = teacher_forced_state_grid(full, prompt_last, self.manifest)
            replay_delta = sidecar(cursor_grid, select_grid)
            replay, _ = model(
                full, q_delta=replay_delta, q_delta_layer=-1, q_delta_head=0,
            )
            self.assertTrue(torch.allclose(logits[:, -1], replay[:, -1], atol=2e-5, rtol=2e-5))


if __name__ == "__main__":
    unittest.main()
