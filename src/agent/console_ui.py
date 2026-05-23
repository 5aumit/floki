from rich.console import Console
from rich.table import Table
from rich import box
from rich.markdown import Markdown
from rich.text import Text
import json
import re
from typing import Any, List, Optional

console = Console()


def print_header(title: str):
    console.rule(f"{title}")


def print_text(text: str):
    console.print(text)


def print_user(text: str):
    """Print a user message with a clear 'You' tag."""
    try:
        console.print(f"[bold white on dark_blue] You [/bold white on dark_blue] {text}")
    except Exception:
        console.print(f"You: {text}")


# Purple → Blue gradient with glow effect
def print_welcome(app_name: str = "Floki", version: str = "v0.1"):
    from rich.panel import Panel
    from rich.align import Align
    from rich.text import Text

    banner_lines = [
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

    # Purple → Blue gradient
    colors = [
        "#dd22ff",  # bright magenta
        "#cc22ff",
        "#bb22ff",
        "#9922ff",  # purple
        "#7722ff",
        "#5533ff",  # purple-blue
        "#3355ff",  # blue
        "#2266ff",  # bright blue
    ]

    t = Text()
    
    # Add glow layer (dim version as shadow)
    banner_glow = "\n".join(banner_lines)
    t.append(banner_glow + "\n\n\n", style="dim #5533ff")
    
    # Overlay gradient banner on top
    t_banner = Text()
    for line, color in zip(banner_lines, colors):
        t_banner.append(line + "\n", style=f"bold {color}")
    
    t = Text()
    for line, color in zip(banner_lines, colors):
        t.append(line + "\n", style=f"bold {color}")
    t.append("\n")
    t.append(f"  {app_name} {version} 🧭\n", style="bold white on dark_green")
    t.append("\n")
    t.append("Welcome! Type your question and press Enter. Type 'exit' to quit.\n", style="cyan")
    t.append("Tool outputs (tables/JSON) will appear below the assistant message when available.\n", style="dim")

    panel = Panel(Align.center(t), padding=(1, 2), border_style="bright_blue")
    console.print(panel)


def _extract_json(s: str) -> Optional[Any]:
    """Try to extract JSON from a string. Handles fenced blocks and inline JSON."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # Fenced code block with ```json
    m = re.search(r"```(?:json)?\n(.*?)\n```", s, re.DOTALL | re.IGNORECASE)
    if m:
        block = m.group(1).strip()
        try:
            return json.loads(block)
        except Exception:
            pass
    # Try to find first json object/array in text
    m2 = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
    if m2:
        candidate = m2.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            # attempt to fix single quotes to double quotes
            cand2 = candidate.replace("'", '"')
            try:
                return json.loads(cand2)
            except Exception:
                pass
    return None


def _parse_markdown_table(s: str) -> Optional[List[dict]]:
    """Parse a simple markdown table into list-of-dicts. Return None if not detected."""
    if not isinstance(s, str):
        return None
    lines = [ln.rstrip() for ln in s.splitlines()]
    # find a header line with pipes and a separator line with dashes
    for i in range(len(lines) - 1):
        if '|' in lines[i] and re.search(r"^\s*\|?\s*[-:]+", lines[i+1]):
            header = [h.strip() for h in lines[i].strip().strip('|').split('|')]
            rows = []
            j = i + 2
            while j < len(lines) and '|' in lines[j]:
                cols = [c.strip() for c in lines[j].strip().strip('|').split('|')]
                # pad
                while len(cols) < len(header):
                    cols.append('')
                rows.append(dict(zip(header, cols)))
                j += 1
            if rows:
                return rows
    return None


def _print_list_of_dicts(lst: List[dict]):
    # determine column order from union of keys
    keys = []
    for d in lst:
        for k in d.keys():
            if k not in keys:
                keys.append(k)
    table = Table(box=box.SIMPLE, show_lines=False)
    for k in keys:
        table.add_column(str(k))
    for d in lst:
        row = [str(d.get(k, '')) for k in keys]
        table.add_row(*row)
    console.print(table)


def _render_block_response(block_resp: dict):
    """Render a BlockResponse dict which contains ordered blocks.

    Each block is expected to be a dict with keys:
    - type: "text" or "table"
    - markdown: string containing markdown
    """
    blocks = block_resp.get("blocks") or []
    for blk in blocks:
        if not isinstance(blk, dict):
            console.print(str(blk))
            continue
        btype = blk.get("type")
        md = blk.get("markdown", "")
        if btype == "text":
            try:
                console.print(Markdown(md))
            except Exception:
                console.print(md)
        elif btype == "table":
            # Prefer to render as a parsed markdown table for nicer column alignment
            md_table = _parse_markdown_table(md)
            if md_table:
                _print_list_of_dicts(md_table)
            else:
                try:
                    console.print(Markdown(md))
                except Exception:
                    console.print(md)
        else:
            # Unknown block type; print raw representation
            if isinstance(md, str) and md.strip():
                try:
                    console.print(Markdown(md))
                except Exception:
                    console.print(md)
            else:
                console.print(str(blk))


def print_result(result: Any):
    """Render a langchain/agent result structure assuming BlockResponse schema.

    Behavior:
    - Prefer the assistant's final message and expect it to be a BlockResponse
      (JSON with a top-level 'blocks' array). Each block is rendered in order.
    - If the final message does not contain a valid BlockResponse, fall back to
      printing the content as Markdown or a parsed markdown table when possible.
    - Prints a clear 'Floki' tag before assistant outputs.
    """
    if not result:
        console.print("<no result>")
        return

    # Prefer structured_response from ToolStrategy when available
    if isinstance(result, dict) and isinstance(result.get("structured_response"), dict):
        _render_block_response(result["structured_response"])
        return

    messages = None
    if isinstance(result, dict):
        messages = result.get('messages')
    elif hasattr(result, 'messages'):
        messages = result.messages

    # If there are no messages, maybe the result itself is already a BlockResponse
    if not messages:
        if isinstance(result, dict) and isinstance(result.get("blocks"), list):
            _render_block_response(result)
            return
        # Fallback: print raw/pretty
        try:
            if isinstance(result, str):
                console.print(Markdown(result))
            else:
                console.print(result)
        except Exception:
            console.print(str(result))
        return

    # Focus on the assistant's final message only (ignore intermediate tool outputs)
    last = messages[-1]
    try:
        if isinstance(last, dict):
            content = last.get('content')
        else:
            content = getattr(last, 'content', None)
    except Exception:
        content = None

    # If content is already a dict-like BlockResponse
    if isinstance(content, dict) and isinstance(content.get('blocks'), list):
        _render_block_response(content)
        return

    # If content is a string, try to extract JSON (handles embedded/quoted JSON)
    if isinstance(content, str):
        extracted = _extract_json(content)
        if isinstance(extracted, dict) and isinstance(extracted.get('blocks'), list):
            _render_block_response(extracted)
            return
        # Try parsing a markdown table as a fallback, but render remaining text too
        md_table = _parse_markdown_table(content)
        if md_table:
            _print_list_of_dicts(md_table)
            # print remaining text after table if present
            import re as _re
            m = _re.search(r"(\|[^\n]+\n\|[ \-:|]+\n(?:\|.*\n)*)", content, _re.DOTALL)
            if m:
                rest = content.replace(m.group(1), "", 1).strip()
                if rest:
                    console.print(Markdown(rest))
            return
        # Otherwise render as Markdown
        console.print(Markdown(content))
        return

    # Final fallback: print stringified content
    try:
        console.print(str(content))
    except Exception:
        console.print(content)
