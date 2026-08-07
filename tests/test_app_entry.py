from __future__ import annotations

from unittest.mock import patch

import pytest

import app as app_entry


@pytest.mark.parametrize("argv", [[], ["--tui"]])
def test_default_and_tui_flag_start_textual_interface(argv: list[str]) -> None:
    with (
        patch("app.main_tui") as run_tui,
        patch("app.run_legacy_repl") as run_repl,
        patch("app.flush_langfuse_on_exit") as flush,
    ):
        app_entry.run(argv)

    run_tui.assert_called_once_with()
    run_repl.assert_not_called()
    # FlokiApp owns normal TUI shutdown, so the entry point must not flush too.
    flush.assert_not_called()


def test_repl_flag_starts_only_legacy_interface() -> None:
    with (
        patch("app.main_tui") as run_tui,
        patch("app.run_legacy_repl") as run_repl,
        patch("app.flush_langfuse_on_exit") as flush,
    ):
        app_entry.run(["--repl"])

    run_tui.assert_not_called()
    run_repl.assert_called_once_with()
    flush.assert_called_once_with()


def test_interface_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        app_entry._parse_args(["--tui", "--repl"])


def test_startup_interrupt_finishes_langfuse_without_normal_flush() -> None:
    with (
        patch("app.main_tui", side_effect=KeyboardInterrupt) as run_tui,
        patch("app.finish_langfuse") as finish,
        patch("app.flush_langfuse_on_exit") as flush,
    ):
        app_entry.run([])

    run_tui.assert_called_once_with()
    finish.assert_called_once_with()
    flush.assert_not_called()
