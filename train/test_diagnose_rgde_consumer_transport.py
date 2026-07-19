import torch

from diagnose_rgde_consumer_transport import rebind_packet


def test_rebind_packet_uses_initial_entity_and_preserves_other_fields():
    initial = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    packet = {
        "initial_entities": initial,
        "operations": (
            {"entity": torch.zeros(6), "literal": torch.ones(6)},
            {"entity": torch.zeros(6), "literal": 2 * torch.ones(6)},
        ),
        "query": torch.tensor([1.0, 2.0]),
    }
    rebound = rebind_packet(packet, [2, 0])
    assert torch.equal(rebound["operations"][0]["entity"], initial[2])
    assert torch.equal(rebound["operations"][1]["entity"], initial[0])
    assert torch.equal(rebound["operations"][0]["literal"], packet["operations"][0]["literal"])
    assert torch.equal(rebound["query"], packet["query"])
    assert torch.equal(packet["operations"][0]["entity"], torch.zeros(6))


def test_rebind_packet_rejects_invalid_identity():
    packet = {
        "initial_entities": torch.zeros(3, 2),
        "operations": ({"entity": torch.zeros(2)},),
        "query": torch.zeros(2),
    }
    try:
        rebind_packet(packet, [3])
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("invalid identity was accepted")
