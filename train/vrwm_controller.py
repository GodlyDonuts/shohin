"""Closed-loop controller for VRWM inference and evaluation.

The controller deliberately has a narrow role: it supplies a previously stored
next instruction and returns the model's parsed working memory unchanged.  It
does not execute a correction, select a better candidate, or synthesize the
final answer, so a successful rollout is evidence that the model generated all
intermediate state transitions itself.
"""
from vrwm_protocol import apply_operation, parse_memory, transition_prompt


def rollout_episode(episode, ask):
    """Run a model-provided ``ask(prompt)`` across an episode's program."""
    memory = dict(episode["initial_memory"])
    rows = []
    for index, operation in enumerate(episode["operations"]):
        prompt = transition_prompt(memory, operation)
        response = ask(prompt)
        predicted = parse_memory(response)
        expected = apply_operation(memory, operation)
        ok = predicted == expected
        rows.append({
            "index": index,
            "prompt": prompt,
            "response": response,
            "input_memory": memory,
            "expected_memory": expected,
            "predicted_memory": predicted,
            "correct": ok,
        })
        if not ok:
            return {"success": False, "memory": memory, "rows": rows}
        memory = predicted
    return {"success": True, "memory": memory, "rows": rows}
