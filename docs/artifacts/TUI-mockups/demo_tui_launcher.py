#!/usr/bin/env python3
"""Fast TUI launcher for README demo capture (mocked agent, real colors)."""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "agent")]

from tui.app import FlokiApp  # noqa: E402

DEMO_RESULT = {
    "structured_response": {
        "blocks": [
            {
                "type": "text",
                "markdown": (
                    "## Experiments\n\n"
                    "Found **3** experiments on the tracking server:\n\n"
                    "- `customer-churn` — tree vs linear classifiers\n"
                    "- `fraud-detection` — ranking runs (F1 primary)\n"
                    "- `Default` — scratch / ad-hoc runs"
                ),
            },
            {
                "type": "table",
                "markdown": (
                    "| Experiment | Runs | Best metric |\n"
                    "| --- | ---: | --- |\n"
                    "| customer-churn | 24 | accuracy **0.91** |\n"
                    "| fraud-detection | 11 | f1 **0.87** |\n"
                    "| Default | 3 | — |"
                ),
            },
        ]
    }
}


def _fake_runtime() -> MagicMock:
    rt = MagicMock()
    rt.config = {
        "llm": {
            "gemini_model": "gemini-2.5-flash",
            "formatter_model": "gemini-2.5-flash-lite",
        }
    }
    rt.fuse_client = None
    rt.conversation_id = "demo"
    return rt


def _fake_run_query(user_query: str, on_status=None) -> dict:
    for verb, delay in [
        ("thinking", 0.6),
        ("listing", 0.8),
        ("scanning", 0.8),
        ("counting", 0.7),
        ("formatting", 0.5),
    ]:
        if on_status is not None:
            on_status(verb)
        time.sleep(delay)
    return DEMO_RESULT


def main() -> None:
    with (
        patch("tui.app.get_runtime", side_effect=_fake_runtime),
        patch("tui.app.run_query_traced", side_effect=_fake_run_query),
        patch("tui.app.finish_langfuse"),
        patch("tui.app.flush_langfuse_on_exit"),
    ):
        FlokiApp().run()


if __name__ == "__main__":
    main()
