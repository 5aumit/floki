from __future__ import annotations

from typing import Any

from textual.widget import Widget
from textual.widgets import Markdown, Static

from tui.widgets.chat import AssistantResponse, FlokiLabel


class StaticBlock(Static):
    """Plain text fallback for non-markdown block content."""


def extract_structured(result: Any) -> dict | None:
    if not isinstance(result, dict):
        return None
    structured = result.get("structured_response")
    if isinstance(structured, dict) and isinstance(structured.get("blocks"), list):
        return structured
    if isinstance(result.get("blocks"), list):
        return result
    return None


def fallback_content(result: Any) -> str:
    if isinstance(result, dict):
        messages = result.get("messages")
        if messages:
            last = messages[-1]
            if isinstance(last, dict):
                content = last.get("content")
            else:
                content = getattr(last, "content", None)
            if isinstance(content, str):
                return content
    if isinstance(result, str):
        return result
    return str(result)


def render_blocks(container: Widget, block_resp: dict) -> None:
    """Mount formatter BlockResponse blocks as Markdown widgets."""
    for blk in block_resp.get("blocks") or []:
        if not isinstance(blk, dict):
            container.mount(StaticBlock(str(blk)))
            continue
        md = blk.get("markdown", "")
        if isinstance(md, str) and md.strip():
            container.mount(Markdown(md))
        else:
            container.mount(StaticBlock(str(blk)))


def render_result(container: AssistantResponse, result: Any) -> None:
    """Render run_query result into an AssistantResponse container."""
    if not result:
        container.mount(FlokiLabel())
        container.mount(Markdown("_No result._"))
        return

    structured = extract_structured(result)
    container.mount(FlokiLabel())
    if structured:
        render_blocks(container, structured)
        return

    container.mount(Markdown(fallback_content(result)))
