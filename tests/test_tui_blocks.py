from tui.blocks import extract_structured, fallback_content


def test_extract_structured_prefers_structured_response() -> None:
    result = {"structured_response": {"blocks": [{"type": "text", "markdown": "Hello"}]}}

    assert extract_structured(result) == result["structured_response"]


def test_fallback_content_uses_last_agent_message() -> None:
    result = {"messages": [{"content": "Earlier"}, {"content": "Final answer"}]}

    assert fallback_content(result) == "Final answer"
