"""Map MLflow tool names to loading-row verbs (fallback: thinking)."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

StatusCallback = Callable[[str], None]

# ponytail: process-global callback; exclusive query worker means one active query.
# threading.local alone fails if LangGraph runs tools on a pool thread.
_lock = threading.Lock()
_status_cb: Optional[StatusCallback] = None

# Keep in sync with docs/artifacts/TUI-mockups/tui-onedark-v2.html
TOOL_VERBS: dict[str, str] = {
    "list_experiments_tool": "listing",
    "list_runs_tool": "scanning",
    "count_runs_per_experiment_tool": "counting",
    "get_run_metrics_tool": "fetching",
    "get_run_params_tool": "inspecting",
    "find_best_runs_by_metric_tool": "ranking",
    "check_experiment_generalization_tool": "checking",
}

DEFAULT_VERB = "thinking"
FORMAT_VERB = "formatting"


def verb_for_tool(tool_name: str | None) -> str:
    if not tool_name:
        return DEFAULT_VERB
    return TOOL_VERBS.get(tool_name, DEFAULT_VERB)


def emit_status(verb: str) -> None:
    with _lock:
        cb = _status_cb
    if cb is not None:
        cb(verb)


def emit_tool_status(tool_name: str | None) -> None:
    emit_status(verb_for_tool(tool_name))


def tool_name_from_request(request: object) -> str | None:
    tc = getattr(request, "tool_call", None)
    if tc is None:
        return None
    if isinstance(tc, dict):
        name = tc.get("name")
    else:
        name = getattr(tc, "name", None)
    return name if isinstance(name, str) else None


@contextmanager
def status_callback(cb: StatusCallback | None) -> Iterator[None]:
    global _status_cb
    with _lock:
        prev, _status_cb = _status_cb, cb
    try:
        yield
    finally:
        with _lock:
            _status_cb = prev
