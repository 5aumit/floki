from rich.console import Console
from rich.table import Table
from rich import box
import json
import re
from typing import Any, List, Optional

console = Console()


def print_header(title: str):
    console.rule(f"{title}")


def print_text(text: str):
    console.print(text)


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


def print_result(result: Any):
    """Render a langchain/agent result structure nicely.

    Strategy:
    - Prefer structured tool outputs (JSON) found in tool messages.
    - If not found, try to parse a markdown table from the agent text.
    - Fallback to pretty JSON or plain text.
    """
    if not result:
        console.print("<no result>")
        return

    messages = None
    if isinstance(result, dict):
        messages = result.get('messages')
    elif hasattr(result, 'messages'):
        messages = result.messages

    if not messages:
        console.print(result)
        return

    # Show assistant final text first (if present)
    printed_assistant = False
    try:
        last = messages[-1]
        assistant_text = getattr(last, 'content') if hasattr(last, 'content') or isinstance(last, dict) else None
    except Exception:
        assistant_text = None
    if isinstance(assistant_text, str) and assistant_text.strip():
        console.print(assistant_text)
        printed_assistant = True

    # First pass: search messages from newest to oldest for structured outputs
    rendered_structured = False
    for msg in reversed(messages):
        # tool calls metadata might be in additional_kwargs or tool_calls
        tw = None
        try:
            tw = getattr(msg, 'additional_kwargs', None) or getattr(msg, 'tool_calls', None) or (msg.get('tool_calls') if isinstance(msg, dict) else None)
        except Exception:
            tw = None

        content = None
        try:
            content = getattr(msg, 'content')
        except Exception:
            try:
                content = msg.get('content') if isinstance(msg, dict) else None
            except Exception:
                content = None
        if not content:
            continue

        extracted = _extract_json(content)
        if extracted is not None:
            # Print any tool-call metadata associated with this message only
            if tw and not rendered_structured:
                console.print(f"\n[bold cyan]Parsed tool output (from message):[/bold cyan] {tw}")
            elif not rendered_structured:
                console.print(f"\n[bold cyan]Parsed tool output:[/bold cyan]")

            # Render the structured JSON from this message as supplemental output
            if isinstance(extracted, list) and extracted and isinstance(extracted[0], dict):
                _print_list_of_dicts(extracted)
                rendered_structured = True
                continue
            elif isinstance(extracted, dict):
                try:
                    console.print_json(json.dumps(extracted))
                except Exception:
                    console.print(str(extracted))
                rendered_structured = True
                continue

    # If we rendered any structured supplemental output, stop here (assistant text already shown)
    if rendered_structured:
        return

    # Second pass: try to find markdown table in message texts
    for msg in messages:
        try:
            content = getattr(msg, 'content')
        except Exception:
            try:
                content = msg.get('content') if isinstance(msg, dict) else None
            except Exception:
                content = None
        if not content:
            continue
        md_table = _parse_markdown_table(content)
        if md_table:
            _print_list_of_dicts(md_table)
            return

    # Fallback: print the last message content prettily
    last = messages[-1]
    try:
        content = getattr(last, 'content')
    except Exception:
        try:
            content = last.get('content') if isinstance(last, dict) else str(last)
        except Exception:
            content = str(last)
    # try JSON
    extracted = _extract_json(content) if isinstance(content, str) else None
    if extracted is not None:
        try:
            console.print_json(json.dumps(extracted))
        except Exception:
            console.print(extracted)
    else:
        console.print(content)
