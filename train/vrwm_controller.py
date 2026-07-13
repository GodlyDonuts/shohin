"""Closed-loop controller for VRWM inference and evaluation.

The controller deliberately has a narrow role: it supplies a previously stored
next instruction and returns the model's parsed working memory unchanged.  It
does not execute a correction, select a better candidate, or synthesize the
final answer, so a successful rollout is evidence that the model generated all
intermediate state transitions itself.
"""
from vrwm_protocol import apply_operation, parse_memory, repair_prompt, transition_prompt


def rollout_episode(episode, ask, prompt_style="default", self_repair=False):
    """Run model-produced state transitions, optionally with one model-only repair pass."""
    memory = dict(episode["initial_memory"])
    rows = []
    for index, operation in enumerate(episode["operations"]):
        prompt = transition_prompt(memory, operation, style=prompt_style)
        draft_response = ask(prompt)
        draft = parse_memory(draft_response)
        repair_response = ""
        predicted = draft
        if self_repair and draft is not None:
            repair_response = ask(repair_prompt(memory, operation, draft, style=prompt_style))
            predicted = parse_memory(repair_response)
        expected = apply_operation(memory, operation)
        ok = predicted == expected
        rows.append({
            "index": index,
            "prompt": prompt,
            "response": repair_response or draft_response,
            "draft_response": draft_response,
            "draft_memory": draft,
            "repair_response": repair_response,
            "input_memory": memory,
            "expected_memory": expected,
            "predicted_memory": predicted,
            "correct": ok,
        })
        if not ok:
            return {"success": False, "memory": memory, "rows": rows}
        memory = predicted
    return {"success": True, "memory": memory, "rows": rows}
