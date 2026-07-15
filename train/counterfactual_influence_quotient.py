"""Nonlinear intervention signatures for text-to-operator identification.

The mechanism does not ask a classifier to name an operation. It perturbs
visible quantities and entity roles in a problem, measures the resulting
change at future transformer positions, and compares that causal signature to
the signatures of canonical operator hypotheses. Structured task labels are
used only by evaluators after a prediction has been made.
"""

from __future__ import annotations

import hashlib
import random
import re

import torch
import torch.nn.functional as F


NUMERIC_OPCODES = (
    "add_0", "add_1", "sub_0", "sub_1", "move_0_1", "move_1_0",
)
STRUCTURAL_OPCODES = ("merge_0_1", "merge_1_0", "swap")
QUERY_CODES = ("read_0", "read_1", "sum", "difference_0_1", "difference_1_0")
INTEGER = re.compile(r"(?<![\w-])\d+(?![\w-])")


def _event_line_indices(lines):
    return tuple(
        index for index, line in enumerate(lines)
        if line.lstrip().startswith(("Step ", "Event ")) and ":" in line
    )


def _query_line_index(lines):
    events = set(_event_line_indices(lines))
    answer = max(
        index for index, line in enumerate(lines)
        if line.strip() in {"Answer:", "Result:"}
    )
    candidates = [
        index for index in range(answer)
        if index not in events and lines[index].strip()
    ]
    if len(candidates) < 2:
        raise ValueError("question lacks separate introduction and query lines")
    return candidates[-1]


def _replace_line(question, index, replacement):
    lines = question.splitlines()
    lines[int(index)] = replacement
    return "\n".join(lines)


def _same_width_perturbation(value):
    value = int(value)
    lower = 0 if value == 0 else 10 ** (len(str(value)) - 1)
    upper = 10 ** len(str(value)) - 1
    if value < upper:
        return value + 1
    if value > lower:
        return value - 1
    raise ValueError("cannot perturb singleton numeric width")


def _replace_one_integer(text, old, new, *, after=0):
    old = int(old)
    matches = [
        match for match in INTEGER.finditer(text, int(after))
        if int(match.group()) == old
    ]
    if len(matches) != 1:
        raise ValueError("expected one visible integer {}, found {}".format(old, len(matches)))
    match = matches[0]
    return text[:match.start()] + str(int(new)) + text[match.end():]


def _swap_literals(text, left, right):
    left, right = str(left), str(right)
    marker_left = "__CIQ_LEFT_{}__".format(hashlib.sha256(left.encode()).hexdigest()[:12])
    marker_right = "__CIQ_RIGHT_{}__".format(hashlib.sha256(right.encode()).hexdigest()[:12])
    pattern_left = re.compile(r"(?<!\w){}(?!\w)".format(re.escape(left)), re.IGNORECASE)
    pattern_right = re.compile(r"(?<!\w){}(?!\w)".format(re.escape(right)), re.IGNORECASE)
    swapped, count_left = pattern_left.subn(marker_left, text)
    swapped, count_right = pattern_right.subn(marker_right, swapped)
    if not count_left or not count_right:
        raise ValueError("entity swap requires both visible identifiers")
    return swapped.replace(marker_left, right).replace(marker_right, left)


def event_candidates(line, keys):
    """Enumerate legal canonical event rewrites from visible text only."""
    if len(keys) != 2 or ":" not in line:
        raise ValueError("event candidates require two identifiers and a line prefix")
    prefix, body = line.split(":", 1)
    values = [int(match.group()) for match in INTEGER.finditer(body)]
    key0, key1 = map(str, keys)
    if len(values) == 1:
        value = values[0]
        return value, {
            "add_0": f"{prefix}: Add {value} to {key0}.",
            "add_1": f"{prefix}: Add {value} to {key1}.",
            "sub_0": f"{prefix}: Subtract {value} from {key0}.",
            "sub_1": f"{prefix}: Subtract {value} from {key1}.",
            "move_0_1": f"{prefix}: Move {value} from {key0} to {key1}.",
            "move_1_0": f"{prefix}: Move {value} from {key1} to {key0}.",
        }
    if values:
        raise ValueError("event line has multiple visible values")
    return 0, {
        "merge_0_1": f"{prefix}: Add all of {key0} to {key1}.",
        "merge_1_0": f"{prefix}: Add all of {key1} to {key0}.",
        "swap": f"{prefix}: Swap {key0} with {key1}.",
    }


def query_candidates(line, keys):
    """Enumerate canonical future readouts without using a query label."""
    if len(keys) != 2 or ":" not in line:
        raise ValueError("query candidates require two identifiers and a line prefix")
    prefix = line.split(":", 1)[0]
    key0, key1 = map(str, keys)
    return {
        "read_0": f"{prefix}: What is the final {key0} total?",
        "read_1": f"{prefix}: What is the final {key1} total?",
        "sum": f"{prefix}: What is the sum of the final {key0} and {key1} totals?",
        "difference_0_1": f"{prefix}: How many more are in {key0} than in {key1}?",
        "difference_1_0": f"{prefix}: How many more are in {key1} than in {key0}?",
    }


def _perturb_initial(question, key, value):
    lines = question.splitlines()
    events = _event_line_indices(lines)
    if not events:
        raise ValueError("question has no event lines")
    intro = next(index for index in range(events[0]) if lines[index].strip())
    match = re.search(r"(?<!\w){}(?!\w)".format(re.escape(str(key))), lines[intro], re.IGNORECASE)
    if match is None:
        raise ValueError("initial identifier is not visible")
    following = next(INTEGER.finditer(lines[intro], match.end()), None)
    if following is None or int(following.group()) != int(value):
        raise ValueError("identifier is not followed by its bound initial value")
    replacement = (
        lines[intro][:following.start()] + str(_same_width_perturbation(value))
        + lines[intro][following.end():]
    )
    lines[intro] = replacement
    return "\n".join(lines)


def operation_intervention_bundle(row, operation_index, candidate_opcode=None):
    """Return one baseline plus matched interventions for an event hypothesis."""
    question = str(row["question"])
    keys = tuple(map(str, row["keys"]))
    initial = row["initial"]
    lines = question.splitlines()
    events = _event_line_indices(lines)
    line_index = events[int(operation_index)]
    value, candidates = event_candidates(lines[line_index], keys)
    if candidate_opcode is not None:
        baseline = _replace_line(question, line_index, candidates[str(candidate_opcode)])
    else:
        baseline = question
    baseline_lines = baseline.splitlines()
    interventions = {
        "initial_0": _perturb_initial(baseline, keys[0], initial[keys[0]]),
        "initial_1": _perturb_initial(baseline, keys[1], initial[keys[1]]),
    }
    if value:
        body_start = baseline_lines[line_index].index(":") + 1
        baseline_lines[line_index] = _replace_one_integer(
            baseline_lines[line_index], value, _same_width_perturbation(value), after=body_start,
        )
        interventions["event_value"] = "\n".join(baseline_lines)
    else:
        swapped = baseline.splitlines()
        swapped[line_index] = _swap_literals(swapped[line_index], keys[0], keys[1])
        interventions["event_roles"] = "\n".join(swapped)
    query_index = _query_line_index(baseline.splitlines())
    swapped = baseline.splitlines()
    try:
        swapped[query_index] = _swap_literals(swapped[query_index], keys[0], keys[1])
        interventions["query_roles"] = "\n".join(swapped)
    except ValueError:
        pass
    return {
        "baseline": baseline,
        "interventions": interventions,
        "candidate_lines": candidates,
        "value": value,
    }


def operation_intervention_text(row, operation_index, candidate_opcode=None, channels=()):
    """Apply multiple independent matched interventions to one event prompt."""
    question = str(row["question"])
    keys = tuple(map(str, row["keys"]))
    initial = row["initial"]
    lines = question.splitlines()
    line_index = _event_line_indices(lines)[int(operation_index)]
    value, candidates = event_candidates(lines[line_index], keys)
    if candidate_opcode is not None:
        question = _replace_line(question, line_index, candidates[str(candidate_opcode)])
    requested = tuple(channels)
    if len(requested) != len(set(requested)):
        raise ValueError("intervention channels must be unique")
    lawful = {"initial_0", "initial_1", "query_roles"}
    lawful.add("event_value" if value else "event_roles")
    if not set(requested).issubset(lawful):
        raise ValueError("invalid intervention channel")

    result = question
    # Apply independent replacements in a fixed order so joint variants are reproducible.
    for channel in ("initial_0", "initial_1", "event_value", "event_roles", "query_roles"):
        if channel not in requested:
            continue
        if channel == "initial_0":
            result = _perturb_initial(result, keys[0], initial[keys[0]])
        elif channel == "initial_1":
            result = _perturb_initial(result, keys[1], initial[keys[1]])
        elif channel == "event_value":
            current = result.splitlines()
            body_start = current[line_index].index(":") + 1
            current[line_index] = _replace_one_integer(
                current[line_index], value, _same_width_perturbation(value), after=body_start,
            )
            result = "\n".join(current)
        elif channel == "event_roles":
            current = result.splitlines()
            current[line_index] = _swap_literals(current[line_index], keys[0], keys[1])
            result = "\n".join(current)
        else:
            current = result.splitlines()
            query_index = _query_line_index(current)
            current[query_index] = _swap_literals(current[query_index], keys[0], keys[1])
            result = "\n".join(current)
    return result


def query_intervention_bundle(row, candidate_code=None):
    question = str(row["question"])
    keys = tuple(map(str, row["keys"]))
    initial = row["initial"]
    lines = question.splitlines()
    query_index = _query_line_index(lines)
    candidates = query_candidates(lines[query_index], keys)
    baseline = (
        _replace_line(question, query_index, candidates[str(candidate_code)])
        if candidate_code is not None else question
    )
    return {
        "baseline": baseline,
        "interventions": {
            "initial_0": _perturb_initial(baseline, keys[0], initial[keys[0]]),
            "initial_1": _perturb_initial(baseline, keys[1], initial[keys[1]]),
        },
        "candidate_lines": candidates,
    }


def normalized_signature(baseline_states, intervention_states, eps=1e-8):
    """Normalize each intervention's future-state finite difference per layer."""
    if baseline_states.ndim != 2 or intervention_states.ndim != 3:
        raise ValueError("states must have [layers,width] and [interventions,layers,width]")
    if tuple(intervention_states.shape[1:]) != tuple(baseline_states.shape):
        raise ValueError("baseline/intervention state shapes differ")
    delta = intervention_states.float() - baseline_states.float().unsqueeze(0)
    return F.normalize(delta, dim=-1, eps=float(eps))


def normalized_curvature(baseline_states, first_states, second_states, joint_states, eps=1e-8):
    """Return normalized mixed finite differences for intervention pairs."""
    if baseline_states.ndim != 2:
        raise ValueError("baseline states must be [layers,width]")
    expected = (first_states, second_states, joint_states)
    if any(states.ndim != 3 for states in expected):
        raise ValueError("paired states must be [pairs,layers,width]")
    if any(tuple(states.shape[1:]) != tuple(baseline_states.shape) for states in expected):
        raise ValueError("curvature state shapes differ")
    if not (first_states.shape == second_states.shape == joint_states.shape):
        raise ValueError("curvature pair counts differ")
    curvature = (
        joint_states.float() - first_states.float() - second_states.float()
        + baseline_states.float().unsqueeze(0)
    )
    return F.normalize(curvature, dim=-1, eps=float(eps)), curvature.norm(dim=-1)


def informative_channel_order(candidate_signatures):
    """Order interventions by mean pairwise candidate separation."""
    if candidate_signatures.ndim != 4:
        raise ValueError("candidate signatures must be [candidates,channels,layers,width]")
    candidates, channels = candidate_signatures.shape[:2]
    if candidates < 2:
        raise ValueError("at least two candidates are required")
    scores = []
    for channel in range(channels):
        vectors = candidate_signatures[:, channel].reshape(candidates, -1)
        vectors = F.normalize(vectors.float(), dim=-1)
        similarity = vectors @ vectors.T
        mask = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
        separation = float((1.0 - similarity[mask]).mean().item())
        scores.append((separation, -channel, channel))
    return tuple(item[2] for item in sorted(scores, reverse=True))


def random_channel_order(channels, seed):
    order = list(range(int(channels)))
    random.Random(int(seed)).shuffle(order)
    return tuple(order)


def rank_signature_candidates(original_signature, candidate_signatures, channels):
    if original_signature.ndim != 3 or candidate_signatures.ndim != 4:
        raise ValueError("invalid signature ranks")
    selected = torch.tensor(tuple(map(int, channels)), dtype=torch.long, device=original_signature.device)
    if selected.numel() == 0:
        raise ValueError("at least one intervention channel is required")
    original = original_signature.index_select(0, selected)
    candidates = candidate_signatures.index_select(1, selected)
    similarity = (candidates * original.unsqueeze(0)).sum(-1).mean(dim=(1, 2))
    return torch.argsort(similarity, descending=True), similarity


def rank_direct_candidates(original_states, candidate_states):
    if original_states.ndim != 2 or candidate_states.ndim != 3:
        raise ValueError("invalid direct-state ranks")
    original = F.normalize(original_states.float(), dim=-1)
    candidates = F.normalize(candidate_states.float(), dim=-1)
    similarity = (candidates * original.unsqueeze(0)).sum(-1).mean(-1)
    return torch.argsort(similarity, descending=True), similarity
