import logging
from typing import Any, List, Optional

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from agent.agent_middleware import BlockResponse

_logger = logging.getLogger(__name__)

FORMAT_SYSTEM_PROMPT = """You format MLflow assistant answers into a BlockResponse structure.

Rules:
- Use type="text" for summaries, analysis, and next steps.
- Use type="table" for comparisons and tabular data (markdown pipe tables).
- Use only values present in tool results and the agent draft; do not invent data.
- On tool errors, explain the issue in a text block.
- Always return at least one block."""


def _message_content(msg: Any) -> str:
    if isinstance(msg, dict):
        content = msg.get("content")
    else:
        content = getattr(msg, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def _message_type(msg: Any) -> str:
    if isinstance(msg, dict):
        return msg.get("type") or msg.get("role") or ""
    return type(msg).__name__


def _message_name(msg: Any) -> str:
    if isinstance(msg, dict):
        return msg.get("name") or "tool"
    return getattr(msg, "name", None) or "tool"


def _extract_tool_results(messages: List[Any]) -> List[str]:
    parts = []
    for msg in messages:
        msg_type = _message_type(msg)
        if msg_type in ("tool", "ToolMessage"):
            name = _message_name(msg)
            content = _message_content(msg)
            if content:
                parts.append(f"[{name}]:\n{content}")
    return parts


def _extract_draft_answer(messages: List[Any]) -> str:
    draft = ""
    for msg in messages:
        msg_type = _message_type(msg)
        if msg_type in ("ai", "AIMessage", "assistant"):
            content = _message_content(msg)
            if content:
                draft = content
    return draft


def build_format_input(messages: List[Any], user_query: str) -> str:
    tool_parts = _extract_tool_results(messages)
    draft = _extract_draft_answer(messages)

    sections = [f"## User query\n\n{user_query}"]
    if tool_parts:
        sections.append("## Tool results\n\n" + "\n\n".join(tool_parts))
    else:
        sections.append("## Tool results\n\n(no tool calls)")
    sections.append(f"## Agent draft\n\n{draft or '(no draft)'}")
    return "\n\n".join(sections)


def _to_block_response_dict(result: Any) -> dict:
    if isinstance(result, BlockResponse):
        return result.model_dump()
    if isinstance(result, dict) and isinstance(result.get("blocks"), list):
        return BlockResponse.model_validate(result).model_dump()
    raise ValidationError.from_exception_data(
        "BlockResponse",
        [{"type": "model_type", "loc": (), "msg": "Unexpected formatter output", "input": result}],
    )


def _fallback_response(draft: str) -> dict:
    return {"blocks": [{"type": "text", "markdown": draft or "No response generated."}]}


def format_to_block_response(
    formatter_llm: Any,
    messages: List[Any],
    user_query: str,
) -> dict:
    """Format agent output into BlockResponse via a dedicated LLM call."""
    draft = _extract_draft_answer(messages)
    format_input = build_format_input(messages, user_query)
    structured_llm = formatter_llm.with_structured_output(BlockResponse)

    last_error: Optional[Exception] = None
    for attempt in range(2):
        try:
            user_content = format_input
            if attempt == 1 and last_error is not None:
                user_content += f"\n\n## Previous error\n\n{last_error}"

            result = structured_llm.invoke([
                SystemMessage(content=FORMAT_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            return _to_block_response_dict(result)
        except (ValidationError, Exception) as e:
            last_error = e
            _logger.warning("Formatter attempt %d failed: %s", attempt + 1, e)

    _logger.error("Formatter failed after retries; using fallback text block.")
    return _fallback_response(draft)
