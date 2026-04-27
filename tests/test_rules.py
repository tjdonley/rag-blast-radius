from rag_blast.rules import get_rule


def test_get_rule_is_case_insensitive() -> None:
    rule = get_rule("reembed_required")

    assert rule is not None
    assert rule.id == "REEMBED_REQUIRED"


def test_get_rule_returns_none_for_unknown_rule() -> None:
    assert get_rule("UNKNOWN_RULE") is None
