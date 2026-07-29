from agent.context_memory import trim_messages_for_memory


def test_trim_messages_for_memory_removes_tool_traffic() -> None:
    messages = [
        {"role": "user", "content": "List experiments"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "list"}]},
        {"role": "tool", "content": "[{\"name\": \"demo\"}]"},
        {"role": "assistant", "content": "I found one experiment."},
    ]

    trimmed = trim_messages_for_memory(messages)

    assert trimmed == [messages[0], messages[3]]


def test_trim_messages_for_memory_keeps_a_non_tool_assistant_message() -> None:
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "I can help with MLflow."},
    ]

    assert trim_messages_for_memory(messages) == messages
