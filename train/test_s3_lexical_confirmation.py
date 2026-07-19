from eval_s3_lexical_confirmation import apply_intervention


def row(index, query):
    names = ["a{}".format(index), "b{}".format(index), "c{}".format(index)]
    return {
        "depth": 3,
        "initial_order": names,
        "program": [
            {"kind": "left" if index % 2 else "right", "entity": names[0], "amount": 1},
            {"kind": "right", "entity": names[1], "amount": 2},
            {"kind": "left", "entity": names[2], "amount": 1},
        ],
        "query": {"position": query},
    }


def test_interventions_change_only_the_selected_packet_field():
    rows = [row(index, index % 3) for index in range(6)]
    packets = [{"operations": ("op{}".format(index),), "query": "q{}".format(index)}
               for index in range(6)]
    operation, changed = apply_intervention(rows, packets, "operations")
    assert changed == 6
    assert [value["query"] for value in operation] == [value["query"] for value in packets]
    assert all(value["operations"] != packets[index]["operations"]
               for index, value in enumerate(operation))
    query, changed = apply_intervention(rows, packets, "query")
    assert changed == 6
    assert [value["operations"] for value in query] == [
        value["operations"] for value in packets
    ]
    assert all(value["query"] != packets[index]["query"]
               for index, value in enumerate(query))
