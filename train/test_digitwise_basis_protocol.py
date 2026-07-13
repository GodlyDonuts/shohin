"""Contracts for the reachable DRS local-transition basis."""
from digitwise_basis_protocol import context_label, reachable_contexts
from digitwise_protocol import initial_state


contexts = reachable_contexts((4,))
assert (4, "add", 0, 0, 0, 0) in contexts
assert (4, "add", 2, 1, 9, 9) in contexts
assert (4, "sub", 3, 0, 5, 5) in contexts
assert (4, "sub", 3, 1, 5, 5) not in contexts
assert (4, "sub", 3, 0, 4, 5) not in contexts
assert len(contexts) == 1300
assert context_label((4, "add", 2, 1, 9, 9)) == "w4-add-p2-c1-99"
print("digitwise basis protocol checks: passed")
