from researcher_adaptive_interaction import (
    MAX_QUOTED_RESPONSE_CHARS,
    bounded_quote,
    probe_definitions,
    score_response,
)


def test_score_response_separates_semantics_from_strict_format():
    expected = {"text": "quill=85", "field": "quill", "state": {"quill": 85}}
    score = score_response("assignment", expected, "The answer is quill = 85.\nExtra")
    assert score["semantic_correct"]
    assert not score["first_line_exact"]
    assert not score["strict_exact"]


def test_score_response_handles_packet_and_memo():
    packet = score_response(
        "digit_packet",
        {"text": "digit=3;carry=1", "state": {"digit": 3, "carry": 1}},
        "digit=3;carry=1",
    )
    memo = score_response(
        "memo",
        {
            "text": "memo{r=225;seal=KITE}",
            "state": {"r": 225, "seal": "KITE"},
        },
        "memo { r = 225 ; seal = KITE }",
    )
    assert packet["semantic_correct"] and packet["strict_exact"]
    assert memo["semantic_correct"] and memo["strict_exact"]


def test_probe_dependencies_are_preceded_by_their_sources():
    definitions = probe_definitions()
    ids = [row["id"] for row in definitions]
    assert ids.index("scalar_plain") < ids.index("scalar_review")
    assert ids.index("scalar_plain") < ids.index("scalar_serialize_model")
    assert ids.index("digit_sum_plain") < ids.index("digit_packet_model")
    assert ids.index("memo_step_one") < ids.index("memo_step_two_model")


def test_bounded_quote_limits_context_and_strips_nul():
    quoted = bounded_quote("x\x00" + "y" * (MAX_QUOTED_RESPONSE_CHARS + 20))
    assert "\x00" not in quoted
    assert len(quoted) <= MAX_QUOTED_RESPONSE_CHARS
