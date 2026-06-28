"""Session memory helpers: trim tool traffic from checkpointed message history."""

import logging
from typing import Any, List

from langchain.messages import AIMessage, HumanMessage, ToolMessage

_logger = logging.getLogger(__name__)


def _message_content(msg: Any) -> str:
    if isinstance(msg, dict):
        content = msg.get("content")
    else:
        content = getattr(msg, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _message_type(msg: Any) -> str:
    if isinstance(msg, dict):
        return msg.get("type") or msg.get("role") or ""
    return type(msg).__name__


def _has_tool_calls(msg: Any) -> bool:
    if isinstance(msg, dict):
        tool_calls = msg.get("tool_calls")
    else:
        tool_calls = getattr(msg, "tool_calls", None)
    return bool(tool_calls)


def _is_tool_message(msg: Any) -> bool:
    msg_type = _message_type(msg)
    return msg_type in ("tool", "ToolMessage") or isinstance(msg, ToolMessage)


def _is_human_message(msg: Any) -> bool:
    msg_type = _message_type(msg)
    return msg_type in ("human", "HumanMessage", "user") or isinstance(msg, HumanMessage)


def _is_ai_message(msg: Any) -> bool:
    msg_type = _message_type(msg)
    return msg_type in ("ai", "AIMessage", "assistant") or isinstance(msg, AIMessage)


def trim_messages_for_memory(messages: List[Any]) -> List[Any]:
    """Keep Human + final AI answers; drop ToolMessage and tool-only AI steps."""
    trimmed: List[Any] = []
    for msg in messages:
        if _is_tool_message(msg):
            continue
        if _is_ai_message(msg) and _has_tool_calls(msg) and not _message_content(msg):
            continue
        trimmed.append(msg)

    has_ai_answer = any(_is_ai_message(m) and _message_content(m) for m in trimmed)
    if not has_ai_answer and trimmed:
        last_human_idx = max(
            (i for i, m in enumerate(trimmed) if _is_human_message(m)),
            default=None,
        )
        if last_human_idx is not None:
            _logger.warning(
                "No AI text answer in turn; keeping history through last human message."
            )
            trimmed = trimmed[: last_human_idx + 1]

    return trimmed
