import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.agent_middleware import BlockResponse, TextBlock
from agent.response_formatter import (
    build_format_input,
    format_to_block_response,
)


def test_build_format_input_includes_query_tools_and_draft():
    messages = [
        HumanMessage(content="top 3 runs by accuracy"),
        AIMessage(content="", tool_calls=[{"id": "1", "name": "find_best_runs_by_metric_tool", "args": {}}]),
        ToolMessage(
            content='[{"run_id": "a1", "accuracy": 0.94}]',
            tool_call_id="1",
            name="find_best_runs_by_metric_tool",
        ),
        AIMessage(content="rf_v3 is the best run with accuracy 0.94."),
    ]

    result = build_format_input(messages, "top 3 runs by accuracy")

    assert "## User query" in result
    assert "top 3 runs by accuracy" in result
    assert "## Tool results" in result
    assert "find_best_runs_by_metric_tool" in result
    assert "0.94" in result
    assert "## Agent draft" in result
    assert "rf_v3 is the best run" in result


def test_build_format_input_no_tools():
    messages = [
        HumanMessage(content="what can you do?"),
        AIMessage(content="I can list experiments and runs."),
    ]

    result = build_format_input(messages, "what can you do?")

    assert "(no tool calls)" in result
    assert "I can list experiments and runs." in result


def test_format_to_block_response_success():
    mock_llm = MagicMock()
    structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = structured_llm
    structured_llm.invoke.return_value = BlockResponse(
        blocks=[TextBlock(type="text", markdown="Summary here.")]
    )

    messages = [
        HumanMessage(content="list experiments"),
        AIMessage(content="Found 2 experiments."),
    ]

    result = format_to_block_response(mock_llm, messages, "list experiments")

    assert result == {"blocks": [{"type": "text", "markdown": "Summary here."}]}
    mock_llm.with_structured_output.assert_called_once()
    structured_llm.invoke.assert_called_once()


def test_format_to_block_response_fallback_on_failure():
    mock_llm = MagicMock()
    structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = structured_llm
    structured_llm.invoke.side_effect = RuntimeError("API error")

    messages = [
        HumanMessage(content="list experiments"),
        AIMessage(content="Draft answer text."),
    ]

    result = format_to_block_response(mock_llm, messages, "list experiments")

    assert result == {"blocks": [{"type": "text", "markdown": "Draft answer text."}]}
    assert structured_llm.invoke.call_count == 2
