"""CBC controller checks: source deletion, interchange, and no repair path."""
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from bisimulation_compiler_controller import evaluate_pair
from bisimulation_compiler_protocol import canonical_state, delta_prompt, sum_query_prompt, update_prompt
from generate_counterfactual_bisimulation_v1 import HELDOUT_DOMAINS, make_episode


episode = make_episode("controller-smoke", "heldout", HELDOUT_DOMAINS[0], 3, random.Random(17), True)
keys = tuple(episode["keys"])
responses = {}
for world in (episode["normal"], episode["counterfactual"]):
    for prompt in world["compile_prompts"].values():
        responses[prompt] = world["initial_state"]
    for step in world["steps"]:
        responses[step["update_prompt"]] = step["after_state"]
        responses[step["delta_prompt"]] = step["delta"]
    responses[world["query"]["prompt"]] = "answer={}".format(world["query"]["answer"])

counter_terminal = episode["counterfactual"]["steps"][-1]["after_state"]
cross_world_prompt = sum_query_prompt(counter_terminal, keys, episode["reference"], episode["style"])
responses[cross_world_prompt] = "answer={}".format(episode["counterfactual"]["query"]["answer"])

def ask(prompt):
    return responses[prompt]


result = evaluate_pair(episode, ask)
assert result["normal"]["primary"]["final_correct"]
assert result["counterfactual"]["primary"]["final_correct"]
assert all(row["inverse_delta_correct"] for row in result["normal"]["primary"]["rows"])
assert result["same_world_interchange_success"]
assert result["counterfactual_interchange_success"]
assert result["cross_world_interchange"]["uses_counterfactual_answer"]
assert result["cross_world_interchange"]["rejects_normal_answer"]
assert result["cross_world_counterfactual_success"]

broken = dict(responses)
first_prompt = episode["normal"]["steps"][0]["update_prompt"]
broken[first_prompt] = canonical_state({keys[0]: 0, keys[1]: 0}, keys)
failed = evaluate_pair(episode, lambda prompt: broken[prompt])
assert not failed["normal"]["primary"]["state_closed_loop"]
assert failed["normal"]["primary"]["final_response"] == ""
print("bisimulation compiler controller checks: passed")
