from __future__ import annotations

from typing import Any

from rich.align import Align
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

console = Console()

_BANNER_LINES = [
    "   ▄████████  ▄█        ▄██████▄     ▄█   ▄█▄  ▄█  ",
    "  ███    ███ ███       ███    ███   ███ ▄███▀ ███  ",
    "  ███    █▀  ███       ███    ███   ███▐██▀   ███▌ ",
    " ▄███▄▄▄     ███       ███    ███  ▄█████▀    ███▌ ",
    "▀▀███▀▀▀     ███       ███    ███ ▀▀█████▄    ███▌ ",
    "  ███        ███       ███    ███   ███▐██▄   ███  ",
    "  ███        ███▌    ▄ ███    ███   ███ ▀███▄ ███  ",
    "  ███        █████▄▄██  ▀██████▀    ███   ▀█▀ █▀   ",
    "             ▀                      ▀              ",
]

_BANNER_COLORS = [
    "#dd22ff",
    "#cc22ff",
    "#bb22ff",
    "#9922ff",
    "#7722ff",
    "#5533ff",
    "#3355ff",
    "#2266ff",
]


def _print_markdown(md: str) -> None:
    try:
        console.print(Markdown(md))
    except Exception:
        console.print(md)


def print_welcome(app_name: str = "Floki", version: str = "v0.1") -> None:
    t = Text()
    for line, color in zip(_BANNER_LINES, _BANNER_COLORS):
        t.append(line + "\n", style=f"bold {color}")
    t.append(f"\n  {app_name} {version} 🧭\n", style="bold white on dark_green")
    t.append("\n")
    t.append("Welcome! Type your question and press Enter. Type 'exit' to quit.\n", style="cyan")
    t.append("Tool outputs (tables/JSON) will appear below the assistant message when available.\n", style="dim")
    console.print(Panel(Align.center(t), padding=(1, 2), border_style="bright_blue"))


def _render_block_response(block_resp: dict) -> None:
    """Render ordered BlockResponse blocks (text or table markdown)."""
    for blk in block_resp.get("blocks") or []:
        if not isinstance(blk, dict):
            console.print(str(blk))
            continue
        md = blk.get("markdown", "")
        if isinstance(md, str) and md.strip():
            _print_markdown(md)
        else:
            console.print(str(blk))


def _extract_structured(result: Any) -> dict | None:
    if not isinstance(result, dict):
        return None
    structured = result.get("structured_response")
    if isinstance(structured, dict) and isinstance(structured.get("blocks"), list):
        return structured
    if isinstance(result.get("blocks"), list):
        return result
    return None


def _fallback_content(result: Any) -> str:
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


def print_result(result: Any) -> None:
    """Render agent output from structured_response BlockResponse, with thin fallback."""
    if not result:
        console.print("<no result>")
        return

    structured = _extract_structured(result)
    if structured:
        console.print("[bold white on dark_green] Floki [/bold white on dark_green]")
        _render_block_response(structured)
        return

    _print_markdown(_fallback_content(result))
