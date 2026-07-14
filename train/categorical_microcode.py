"""Causal Microcode Bottleneck for compact internal execution.

The frozen language model compiles each event line and the final query into a
small categorical program. Numeric literals are extracted by a deterministic
lexer, while operation and register binding remain neural. A learned decimal
transition table executes the program without generating intermediate text.

This is a narrow neuro-symbolic architecture experiment. It is not a claim
that lexical number extraction or an exhaustive local transition basis is
general language reasoning.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


OPCODES = (
    "add_0", "add_1", "sub_0", "sub_1", "move_0_1",
    "move_1_0", "merge_0_1", "merge_1_0", "swap",
)
QUERIES = ("read_0", "read_1", "sum", "difference_0_1", "difference_1_0")
OPCODE_TO_ID = {name: index for index, name in enumerate(OPCODES)}
QUERY_TO_ID = {name: index for index, name in enumerate(QUERIES)}
STANDALONE_INTEGER = re.compile(r"(?<![\w-])\d+(?![\w-])")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opcode_for(operation, keys):
    key_index = {key: index for index, key in enumerate(keys)}
    kind = operation["kind"]
    if kind in {"add", "sub"}:
        return "{}_{}".format(kind, key_index[operation["target"]])
    if kind in {"move", "merge"}:
        return "{}_{}_{}".format(
            kind, key_index[operation["source"]], key_index[operation["target"]],
        )
    if kind == "swap":
        return "swap"
    raise ValueError("unknown operation {}".format(kind))


def query_for(query, keys):
    key_index = {key: index for index, key in enumerate(keys)}
    kind = query["kind"]
    if kind == "read":
        return "read_{}".format(key_index[query["key"]])
    if kind == "sum":
        return "sum"
    if kind == "difference":
        return "difference_{}_{}".format(
            key_index[query["high"]], key_index[query["low"]],
        )
    raise ValueError("unknown query {}".format(kind))


def _line_spans(text):
    spans, cursor = [], 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        spans.append((cursor, cursor + len(content), content))
        cursor += len(line)
    if not spans and text:
        spans.append((0, len(text), text))
    return spans


def _last_token_in_span(offsets, start, end):
    candidates = [
        index for index, (left, right) in enumerate(offsets)
        if right > left and left >= start and right <= end
    ]
    if not candidates:
        raise ValueError("line span has no tokenizer token")
    return candidates[-1]


def structural_positions(question, encoding):
    """Locate shared event/query read positions without solving the task."""
    spans = _line_spans(question)
    event_spans = [span for span in spans if span[2].startswith(("Step ", "Event "))]
    answer_spans = [span for span in spans if span[2].strip() in {"Answer:", "Result:"}]
    if not event_spans or not answer_spans:
        raise ValueError("question lacks event or answer lines")
    answer_start = answer_spans[-1][0]
    candidates = [span for span in spans if span[1] <= answer_start and span[2].strip()]
    query_span = candidates[-1]
    if query_span in event_spans:
        raise ValueError("query line collapsed into an event line")
    return (
        [_last_token_in_span(encoding.offsets, start, end) for start, end, _ in event_spans],
        _last_token_in_span(encoding.offsets, query_span[0], query_span[1]),
        [span[2] for span in event_spans],
    )


def lexical_initial_values(question):
    first_line = question.splitlines()[0]
    values = [int(value) for value in STANDALONE_INTEGER.findall(first_line)]
    if len(values) != 2:
        raise ValueError("intro must contain exactly two standalone quantities: {}".format(values))
    return values


def lexical_operation_value(line):
    tail = line.split(":", 1)[-1]
    values = [int(value) for value in STANDALONE_INTEGER.findall(tail)]
    if len(values) > 1:
        raise ValueError("event line has multiple standalone values: {}".format(values))
    return values[0] if values else 0


def operation_value(operation):
    return int(operation.get("value", 0))


@dataclass(frozen=True)
class CompiledExample:
    ids: tuple
    operation_positions: tuple
    operation_targets: tuple
    operation_values: tuple
    query_position: int
    query_target: int
    initial_values: tuple
    answer: int
    regime: str
    reference: str


def compile_example(row, tokenizer):
    question = row["question"]
    encoding = tokenizer.encode(question)
    event_positions, query_position, event_lines = structural_positions(question, encoding)
    operations = row["operations"]
    keys = row["keys"]
    if len(event_positions) != len(operations) or len(event_lines) != len(operations):
        raise ValueError("event count does not match structured program")
    initial = tuple(lexical_initial_values(question))
    expected_initial = tuple(int(row["initial"][key]) for key in keys)
    if initial != expected_initial:
        raise ValueError("lexical initial values do not match structured values")
    lexical_values = tuple(lexical_operation_value(line) for line in event_lines)
    expected_values = tuple(operation_value(operation) for operation in operations)
    if lexical_values != expected_values:
        raise ValueError("lexical event values do not match structured values")
    return CompiledExample(
        ids=tuple(encoding.ids),
        operation_positions=tuple(event_positions),
        operation_targets=tuple(OPCODE_TO_ID[opcode_for(operation, keys)] for operation in operations),
        operation_values=lexical_values,
        query_position=int(query_position),
        query_target=QUERY_TO_ID[query_for(row["query"], keys)],
        initial_values=initial,
        answer=int(row["answer"]),
        regime=str(row.get("eval_regime", "train")),
        reference=str(row.get("reference", "")),
    )


class CategoricalMicrocodeCompiler(nn.Module):
    """Shared line compiler over a frozen causal transformer."""

    def __init__(self, model, layer=19, hidden=256):
        super().__init__()
        if model.cfg.n_loop != 1:
            raise ValueError("categorical compiler requires n_loop=1")
        if not 0 <= int(layer) < len(model.blocks):
            raise ValueError("invalid compiler layer")
        self.model = model
        self.layer = int(layer)
        self.model.requires_grad_(False)
        width = model.cfg.d_model
        self.norm = nn.LayerNorm(width)
        self.trunk = nn.Sequential(
            nn.Linear(width, hidden, bias=False), nn.SiLU(),
            nn.Linear(hidden, hidden, bias=False), nn.SiLU(),
        )
        self.operation_head = nn.Linear(hidden, len(OPCODES))
        self.query_head = nn.Linear(hidden, len(QUERIES))
        # Each local decimal context owns a categorical next digit/carry
        # distribution. The table is learned from a complete 400-context basis.
        self.transition_logits = nn.Parameter(torch.zeros(2, 2, 10, 10, 20))

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("model."):
                yield parameter

    def adapter_num_params(self):
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def encode(self, ids):
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("ids must be rank-2 torch.long")
        x = self.model.tok(ids)
        cos = self.model.cos[:ids.shape[1]].to(x.device)
        sin = self.model.sin[:ids.shape[1]].to(x.device)
        for block in self.model.blocks[:self.layer + 1]:
            x, _ = block(x, cos, sin)
        return x.detach()

    def classify_hidden(self, hidden):
        features = self.trunk(self.norm(hidden))
        return self.operation_head(features), self.query_head(features)

    def classify_positions(self, hidden, batch_indices, token_positions, kind):
        selected = hidden[batch_indices, token_positions]
        operation_logits, query_logits = self.classify_hidden(selected)
        if kind == "operation":
            return operation_logits
        if kind == "query":
            return query_logits
        raise ValueError("unknown classification kind")

    def basis_loss(self):
        targets = transition_basis_targets(self.transition_logits.device)
        return F.cross_entropy(self.transition_logits.reshape(-1, 20), targets.reshape(-1))


def transition_basis_targets(device="cpu"):
    target = torch.empty((2, 2, 10, 10), dtype=torch.long, device=device)
    for operation in range(2):
        for carry in range(2):
            for left in range(10):
                for right in range(10):
                    if operation == 0:
                        total = left + right + carry
                        digit, next_carry = total % 10, total // 10
                    else:
                        total = left - right - carry
                        digit, next_carry = (total + 10) % 10, int(total < 0)
                    target[operation, carry, left, right] = digit * 2 + next_carry
    return target


def alu_basis_accuracy(transition_logits):
    predicted = transition_logits.argmax(dim=-1)
    target = transition_basis_targets(predicted.device)
    return int(predicted.eq(target).sum().item()), int(target.numel())


def _digits(value, width):
    if value < 0 or value >= 10 ** width:
        raise ValueError("register value {} does not fit {} digits".format(value, width))
    return [(int(value) // (10 ** index)) % 10 for index in range(width)]


def learned_arithmetic(transition_logits, operation, left, right, width=8):
    """Execute add/sub with only the learned local transition table."""
    operation_id = 0 if operation == "add" else 1 if operation == "sub" else None
    if operation_id is None:
        raise ValueError("operation must be add or sub")
    left_digits, right_digits = _digits(left, width), _digits(right, width)
    result, carry = 0, 0
    table = transition_logits.argmax(dim=-1).detach().cpu()
    for position, (left_digit, right_digit) in enumerate(zip(left_digits, right_digits)):
        code = int(table[operation_id, carry, left_digit, right_digit])
        digit, carry = divmod(code, 2)
        result += digit * (10 ** position)
    if operation == "add":
        result += carry * (10 ** width)
    elif carry:
        raise ValueError("learned subtraction ended with a borrow")
    return result


def execute_program(initial_values, opcodes, operation_values, query, transition_logits, width=8):
    if len(opcodes) != len(operation_values):
        raise ValueError("opcode/value lengths differ")
    registers = [int(initial_values[0]), int(initial_values[1])]
    for opcode, value in zip(opcodes, operation_values):
        name = OPCODES[int(opcode)] if isinstance(opcode, int) else str(opcode)
        value = int(value)
        if name.startswith("add_"):
            target = int(name[-1])
            registers[target] = learned_arithmetic(transition_logits, "add", registers[target], value, width)
        elif name.startswith("sub_"):
            target = int(name[-1])
            registers[target] = learned_arithmetic(transition_logits, "sub", registers[target], value, width)
        elif name.startswith("move_"):
            source, target = map(int, name.split("_")[1:])
            registers[source] = learned_arithmetic(transition_logits, "sub", registers[source], value, width)
            registers[target] = learned_arithmetic(transition_logits, "add", registers[target], value, width)
        elif name.startswith("merge_"):
            source, target = map(int, name.split("_")[1:])
            registers[target] = learned_arithmetic(
                transition_logits, "add", registers[target], registers[source], width,
            )
        elif name == "swap":
            registers[0], registers[1] = registers[1], registers[0]
        else:
            raise ValueError("unknown opcode {}".format(name))
    query_name = QUERIES[int(query)] if isinstance(query, int) else str(query)
    if query_name == "read_0":
        return registers[0]
    if query_name == "read_1":
        return registers[1]
    if query_name == "sum":
        return learned_arithmetic(transition_logits, "add", registers[0], registers[1], width)
    if query_name == "difference_0_1":
        return learned_arithmetic(transition_logits, "sub", registers[0], registers[1], width)
    if query_name == "difference_1_0":
        return learned_arithmetic(transition_logits, "sub", registers[1], registers[0], width)
    raise ValueError("unknown query {}".format(query_name))
