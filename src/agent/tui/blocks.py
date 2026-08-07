from __future__ import annotations

from typing import Any

from rich.markdown import Markdown as RichMarkdown
from textual.containers import HorizontalScroll
from textual.widget import Widget
from textual.widgets import Markdown, Static

from tui.widgets.chat import AssistantResponse, FlokiLabel


class StaticBlock(Static):
    """Plain text fallback for non-markdown block content."""


_NO_RESULT_MARKDOWN = "_No result._"
_MALFORMED_BLOCK_MESSAGE = "Unable to display this response block."
_TABLE_MINIMUM_WIDTH = 96


def extract_structured(result: Any) -> dict | None:
    """Return formatter output from a query result or a direct response."""
    if not isinstance(result, dict):
        return None
    structured = result.get("structured_response")
    if isinstance(structured, dict) and isinstance(structured.get("blocks"), list):
        return structured
    if isinstance(result.get("blocks"), list):
        return result
    return None


def fallback_content(result: Any) -> str:
    """Return the final agent text when formatter output is unavailable."""
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, (list, tuple)) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                content = last.get("content")
            else:
                content = getattr(last, "content", None)
            if isinstance(content, str):
                return content
    if isinstance(result, str):
        return result
    return ""


def render_blocks(container: Widget, block_resp: dict) -> tuple[int, list]:
    """Mount formatter blocks and return the number of mounted block rows."""
    blocks = block_resp.get("blocks")
    if not isinstance(blocks, list):
        return 0, []

    rendered = 0
    mounts = []
    for block in blocks:
        if not isinstance(block, dict):
            mounts.append(container.mount(StaticBlock(_MALFORMED_BLOCK_MESSAGE)))
            rendered += 1
            continue
        block_type = block.get("type")
        markdown = block.get("markdown")
        if block_type == "text" and isinstance(markdown, str) and markdown.strip():
            mounts.append(container.mount(Markdown(markdown)))
        elif block_type == "table" and isinstance(markdown, str) and markdown.strip():
            table = Static(RichMarkdown(markdown))
            table.styles.min_width = _TABLE_MINIMUM_WIDTH
            mounts.append(container.mount(HorizontalScroll(table, classes="table-scroll")))
        else:
            mounts.append(container.mount(StaticBlock(_MALFORMED_BLOCK_MESSAGE)))
        rendered += 1
    return rendered, mounts


def render_result(container: AssistantResponse, result: Any) -> list:
    """Render run_query result into an AssistantResponse container."""
    mounts = [container.mount(FlokiLabel())]
    structured = extract_structured(result)
    if structured is not None:
        rendered, block_mounts = render_blocks(container, structured)
        mounts.extend(block_mounts)
    else:
        rendered = 0
    if rendered:
        return mounts

    content = fallback_content(result)
    mounts.append(container.mount(Markdown(content or _NO_RESULT_MARKDOWN)))
    return mounts
