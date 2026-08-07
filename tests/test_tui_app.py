import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from textual.color import Color
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Input, Markdown, Static

from tui.app import FlokiApp
from tui.widgets.chat import AssistantResponse, FlokiLabel, QueryLoading, SystemNotice, UserPrompt
from tui.widgets.welcome import WelcomePanel


_SAFE_ERROR_GUIDANCE = "Check your configuration or connection, then try again."


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        config={
            "llm": {
                "gemini_model": "agent-model",
                "formatter_model": "formatter-model",
                "gemini_api_key": "must-not-appear-in-status",
            }
        },
        fuse_client=None,
        conversation_id=None,
    )


def _result(markdown: str = "A grounded answer.") -> dict:
    return {
        "structured_response": {
            "blocks": [
                {"type": "text", "markdown": markdown},
                {"type": "table", "markdown": "| run | score |\n| - | - |\n| best | 0.9 |"},
            ]
        },
        "messages": [],
    }


async def _wait_until(pilot, predicate, message: str) -> None:
    """Wait only for a known event-loop state, not an arbitrary long sleep."""
    for _ in range(50):
        if predicate():
            return
        await pilot.pause(0.01)
    pytest.fail(message)


def test_tui_enables_input_after_runtime_initializes() -> None:
    async def run() -> None:
        with patch("tui.app.get_runtime", return_value=_runtime()):
            app = FlokiApp()
            async with app.run_test() as pilot:
                await pilot.pause(0.1)
                assert app.query_one("#query-input", Input).disabled is False
                status = str(app.query_one("#status-bar").render())
                assert "Ready" in status
                assert "agent-model" in status
                assert "formatter-model" in status
                assert "Langfuse off" in status
                assert "must-not-appear-in-status" not in status

    asyncio.run(run())


def test_tui_init_failure_keeps_composer_disabled_with_exit_help() -> None:
    async def run() -> None:
        with patch(
            "tui.app.get_runtime",
            side_effect=RuntimeError(
                "missing config: AWS_SECRET_ACCESS_KEY=demonstration-secret"
            ),
        ):
            app = FlokiApp()
            async with app.run_test() as pilot:
                await _wait_until(
                    pilot,
                    lambda: "Init failed:" in str(app.query_one("#status-bar").render()),
                    "runtime failure did not reach the status line",
                )
                assert app.query_one("#query-input", Input).disabled is True
                status = str(app.query_one("#status-bar").render())
                assert status == f"Init failed: {_SAFE_ERROR_GUIDANCE} Ctrl+Q exit"
                assert "AWS_SECRET_ACCESS_KEY" not in status
                assert "demonstration-secret" not in status

    asyncio.run(run())


@pytest.mark.parametrize(
    "error",
    [
        "AWS_SECRET_ACCESS_KEY=demonstration-secret",
        "private_key=demonstration-secret",
        "Authorization: Basic ZGVtb25zdHJhdGlvbi1zZWNyZXQ=",
        "https://user:demonstration-secret@example.com/private/path",
        "connection refused",
    ],
)
def test_tui_short_error_never_includes_raw_exception_text(error: str) -> None:
    """Error guidance has no details from an untrusted exception."""
    short_error = FlokiApp._short_error(RuntimeError(error))

    assert short_error == _SAFE_ERROR_GUIDANCE
    assert error not in short_error


def test_tui_ignores_empty_input_and_exits_for_exit_words() -> None:
    async def run() -> None:
        with (
            patch("tui.app.get_runtime", return_value=_runtime()),
            patch("tui.app.run_query_traced") as run_query,
        ):
            app = FlokiApp()
            async with app.run_test() as pilot:
                await _wait_until(
                    pilot,
                    lambda: not app.query_one("#query-input", Input).disabled,
                    "runtime did not enable the composer",
                )
                composer = app.query_one("#query-input", Input)
                composer.value = "   "
                await pilot.press("enter")
                await pilot.pause()
                assert not list(app.query_one("#chat-view", VerticalScroll).children)
                run_query.assert_not_called()

                with patch.object(app, "exit") as exit_app:
                    composer.value = "/quit"
                    await pilot.press("enter")
                    await pilot.pause()

                exit_app.assert_called_once_with()
                run_query.assert_not_called()

    asyncio.run(run())


def test_tui_submits_one_query_shows_work_status_and_renders_response() -> None:
    started_scanning = threading.Event()
    started_formatting = threading.Event()
    release_scanning = threading.Event()
    release_formatting = threading.Event()

    def query(query_text: str, on_status) -> dict:
        assert query_text == "Compare the best runs"
        on_status("scanning")
        started_scanning.set()
        assert release_scanning.wait(2)
        on_status("formatting")
        started_formatting.set()
        assert release_formatting.wait(2)
        return _result()

    async def run() -> None:
        with (
            patch("tui.app.get_runtime", return_value=_runtime()),
            patch("tui.app.run_query_traced", side_effect=query) as run_query,
        ):
            app = FlokiApp()
            async with app.run_test() as pilot:
                await _wait_until(
                    pilot,
                    lambda: not app.query_one("#query-input", Input).disabled,
                    "runtime did not enable the composer",
                )
                composer = app.query_one("#query-input", Input)
                composer.value = "Compare the best runs"
                await pilot.press("enter")
                assert await asyncio.to_thread(started_scanning.wait, 1)
                await _wait_until(
                    pilot,
                    lambda: "Scanning…" in str(app.query_one("#status-bar").render()),
                    "tool status did not update on the event loop",
                )

                chat = app.query_one("#chat-view", VerticalScroll)
                assert composer.disabled is True
                assert [type(row) for row in chat.children] == [
                    UserPrompt,
                    QueryLoading,
                    AssistantResponse,
                ]
                assert "Scanning…" in app.query_one(QueryLoading).render().plain
                run_query.assert_called_once()

                release_scanning.set()
                assert await asyncio.to_thread(started_formatting.wait, 1)
                await _wait_until(
                    pilot,
                    lambda: "Formatting…" in str(app.query_one("#status-bar").render()),
                    "formatting status did not update on the event loop",
                )

                release_formatting.set()
                await _wait_until(
                    pilot,
                    lambda: app._active_request_id is None,
                    "query did not return the shell to idle",
                )

                response = app.query_one(AssistantResponse)
                assert response.query_one(FlokiLabel)
                assert response.query(Markdown)
                assert composer.disabled is False
                assert "Ready" in str(app.query_one("#status-bar").render())

    asyncio.run(run())


def test_tui_query_error_hides_raw_text_and_restores_composer() -> None:
    async def run() -> None:
        with (
            patch("tui.app.get_runtime", return_value=_runtime()),
            patch(
                "tui.app.run_query_traced",
                side_effect=RuntimeError("connection refused"),
            ),
        ):
            app = FlokiApp()
            async with app.run_test() as pilot:
                await _wait_until(
                    pilot,
                    lambda: not app.query_one("#query-input", Input).disabled,
                    "runtime did not enable the composer",
                )
                composer = app.query_one("#query-input", Input)
                composer.value = "Explain the failure"
                await pilot.press("enter")
                await _wait_until(
                    pilot,
                    lambda: app._active_request_id is None,
                    "error did not restore the shell to idle",
                )

                response = app.query_one(AssistantResponse)
                notice = response.query_one(SystemNotice)
                assert notice.render().plain == (
                    f"Notice: Query failed: {_SAFE_ERROR_GUIDANCE}"
                )
                assert "connection refused" not in notice.render().plain
                assert composer.disabled is False
                assert "Ready" in str(app.query_one("#status-bar").render())

    asyncio.run(run())


def test_tui_system_notice_escapes_markup() -> None:
    """System notices display markup-like text as plain text."""

    async def run() -> None:
        with patch("tui.app.get_runtime", return_value=_runtime()):
            app = FlokiApp()
            async with app.run_test() as pilot:
                notice = SystemNotice("[bold]unsafe markup[/bold]")
                await app.query_one("#chat-view", VerticalScroll).mount(notice)
                await pilot.pause()

                assert notice.render().plain == "Notice: [bold]unsafe markup[/bold]"

    asyncio.run(run())


def test_tui_cancel_waits_for_query_end_and_discards_late_result() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_query(_query: str, on_status) -> dict:
        on_status("scanning")
        started.set()
        assert release.wait(2)
        return _result("This late answer must not render.")

    async def run() -> None:
        with (
            patch("tui.app.get_runtime", return_value=_runtime()),
            patch("tui.app.run_query_traced", side_effect=blocking_query),
        ):
            app = FlokiApp()
            async with app.run_test() as pilot:
                await _wait_until(
                    pilot,
                    lambda: not app.query_one("#query-input", Input).disabled,
                    "runtime did not enable the composer",
                )
                composer = app.query_one("#query-input", Input)
                composer.value = "Slow question"
                await pilot.press("enter")
                assert await asyncio.to_thread(started.wait, 1)
                await pilot.pause()

                app.action_cancel_or_quit()
                assert app._active_request_id is not None
                assert composer.disabled is True
                assert "Stopping current request…" in str(
                    app.query_one("#status-bar").render()
                )

                release.set()
                await _wait_until(
                    pilot,
                    lambda: app._active_request_id is None,
                    "cancelled query did not finish",
                )
                chat = app.query_one("#chat-view", VerticalScroll)
                notice = chat.query_one(SystemNotice)
                assert notice.render().plain == "Notice: Request cancelled."
                assert not chat.query(AssistantResponse)
                assert composer.disabled is False

    asyncio.run(run())


def test_tui_defers_ctrl_q_exit_until_active_query_finishes() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_query(_query: str, on_status) -> dict:
        del on_status
        started.set()
        assert release.wait(2)
        return _result()

    async def run() -> None:
        with (
            patch("tui.app.get_runtime", return_value=_runtime()),
            patch("tui.app.run_query_traced", side_effect=blocking_query),
        ):
            app = FlokiApp()
            async with app.run_test() as pilot:
                await _wait_until(
                    pilot,
                    lambda: not app.query_one("#query-input", Input).disabled,
                    "runtime did not enable the composer",
                )
                composer = app.query_one("#query-input", Input)
                composer.value = "Slow question"
                await pilot.press("enter")
                assert await asyncio.to_thread(started.wait, 1)
                await pilot.pause()

                app.action_quit()
                assert app.is_running is True
                assert app._exit_pending is True
                assert app._active_request_id is not None

                release.set()
                await _wait_until(
                    pilot,
                    lambda: not app.is_running,
                    "Ctrl+Q exited before the cancelled worker completed",
                )

    asyncio.run(run())


def test_tui_ctrl_c_exits_when_idle() -> None:
    """Ctrl+C follows its documented idle exit path."""

    async def run() -> None:
        with patch("tui.app.get_runtime", return_value=_runtime()):
            app = FlokiApp()
            async with app.run_test() as pilot:
                await _wait_until(
                    pilot,
                    lambda: not app.query_one("#query-input", Input).disabled,
                    "runtime did not enable the composer",
                )

                await pilot.press("ctrl+c")
                await _wait_until(
                    pilot,
                    lambda: not app.is_running,
                    "Ctrl+C did not exit while idle",
                )

    asyncio.run(run())


def test_tui_reports_unexpected_answer_render_failure() -> None:
    """A renderer failure leaves a readable terminal state in the transcript."""

    async def run() -> None:
        with (
            patch("tui.app.get_runtime", return_value=_runtime()),
            patch("tui.app.run_query_traced", return_value=_result()),
            patch("tui.app.render_result", side_effect=RuntimeError("renderer failed")),
        ):
            app = FlokiApp()
            async with app.run_test() as pilot:
                await _wait_until(
                    pilot,
                    lambda: not app.query_one("#query-input", Input).disabled,
                    "runtime did not enable the composer",
                )
                composer = app.query_one("#query-input", Input)
                composer.value = "Show the answer"
                await pilot.press("enter")
                await _wait_until(
                    pilot,
                    lambda: app._active_request_id is None,
                    "render failure did not restore the shell to idle",
                )

                response = app.query_one(AssistantResponse)
                notice = response.query_one(SystemNotice)
                assert notice.render().plain == (
                    "Notice: Could not display this response. Please retry."
                )
                assert composer.disabled is False

    asyncio.run(run())


def test_tui_user_prompt_uses_signal_blue_right_edge() -> None:
    """User rows retain the specified right-side Signal-blue accent."""

    async def run() -> None:
        with patch("tui.app.get_runtime", return_value=_runtime()):
            app = FlokiApp()
            async with app.run_test() as pilot:
                chat = app.query_one("#chat-view", VerticalScroll)
                prompt = UserPrompt("Which run is best?")
                await chat.mount(prompt)
                await pilot.pause(0.1)

                assert prompt.styles.border_right == (
                    "solid",
                    Color.parse("#61afef"),
                )
                assert prompt.styles.border_left[0] == ""

    asyncio.run(run())


@pytest.mark.parametrize(
    ("size", "expected_class", "expected_key_help"),
    [
        ((100, 32), None, "Enter send  |  Ctrl+C cancel/exit  |  Ctrl+Q exit"),
        ((60, 24), "narrow", "Enter send  |  Ctrl+C cancel/exit  |  Ctrl+Q exit"),
        ((59, 24), "compact", "Enter  |  Ctrl+C stop  |  Ctrl+Q exit"),
    ],
)
def test_tui_uses_responsive_layout_classes_and_visible_key_help(
    size: tuple[int, int],
    expected_class: str | None,
    expected_key_help: str,
) -> None:
    """Width classes replace unsupported TCSS media rules at narrow widths."""

    async def run() -> None:
        with patch("tui.app.get_runtime", return_value=_runtime()):
            app = FlokiApp()
            async with app.run_test(size=size) as pilot:
                await pilot.pause(0.1)

                assert app.screen.has_class("narrow") is (expected_class == "narrow")
                assert app.screen.has_class("compact") is (expected_class == "compact")
                key_help = app.query_one("#key-help", Static)
                assert str(key_help.render()) == expected_key_help
                assert key_help.region.width <= size[0]

                welcome = app.query_one(WelcomePanel)
                chat = app.query_one("#chat-view", VerticalScroll)
                command_deck = app.query_one("#command-deck", Vertical)
                prompt = UserPrompt("Question " + "details " * 8)
                response = AssistantResponse()
                await chat.mount(prompt, response)
                await pilot.pause(0.1)

                assert welcome.region.bottom <= chat.region.y
                assert chat.region.height > 0
                assert command_deck.region.bottom <= size[1]
                assert prompt.region.right <= chat.region.right
                assert response.region.right <= chat.region.right
                if expected_class == "narrow":
                    near_full_width = int(chat.virtual_size.width * 0.90)
                    assert prompt.region.width >= near_full_width
                    assert response.region.width >= near_full_width

    asyncio.run(run())


def test_tui_updates_responsive_layout_when_terminal_width_changes() -> None:
    """A live resize updates classes and compact controls without a restart."""

    async def run() -> None:
        with patch("tui.app.get_runtime", return_value=_runtime()):
            app = FlokiApp()
            async with app.run_test(size=(100, 32)) as pilot:
                await pilot.pause(0.1)
                assert not app.screen.has_class("narrow")
                assert not app.screen.has_class("compact")

                await pilot.resize_terminal(60, 24)
                await pilot.pause(0.1)
                assert app.screen.has_class("narrow")
                assert not app.screen.has_class("compact")

                await pilot.resize_terminal(59, 24)
                await pilot.pause(0.1)
                assert not app.screen.has_class("narrow")
                assert app.screen.has_class("compact")
                assert str(app.query_one("#key-help", Static).render()) == (
                    "Enter  |  Ctrl+C stop  |  Ctrl+Q exit"
                )

    asyncio.run(run())


@pytest.mark.parametrize("size", [(60, 24), (35, 24)])
def test_tui_welcome_panel_allocates_all_wrapped_instruction_content(
    size: tuple[int, int],
) -> None:
    """The persistent welcome text never loses a wrapped instruction line."""

    async def run() -> None:
        with patch("tui.app.get_runtime", return_value=_runtime()):
            app = FlokiApp()
            async with app.run_test(size=size) as pilot:
                await pilot.pause(0.1)

                welcome = app.query_one(WelcomePanel)
                rendered = welcome.render()
                required_lines = welcome.visual.get_height(
                    welcome.styles.get_rules(), welcome.content_size.width
                )
                border_height = int(welcome.styles.border_top[0] != "") + int(
                    welcome.styles.border_bottom[0] != ""
                )
                required_height = (
                    required_lines
                    + welcome.styles.padding.top
                    + welcome.styles.padding.bottom
                    + border_height
                )
                visible_content = " ".join(
                    welcome.render_line(y).text.strip()
                    for y in range(welcome.content_size.height)
                )

                assert "Ask about your MLflow experiments." in rendered.plain
                assert "Enter sends. Ctrl+C stops a request or exits." in rendered.plain
                assert welcome.region.height >= required_height
                assert "Ask about your MLflow experiments." in visible_content
                assert "Enter sends. Ctrl+C stops a request or exits." in visible_content

    asyncio.run(run())


def test_tui_scrolls_to_end_after_rendering_text_and_table_result() -> None:
    """A completed response remains at the latest table row after refresh."""
    text_block = "\n".join(
        f"Result {index}: evidence for the selected run." for index in range(20)
    )

    async def run() -> None:
        with (
            patch("tui.app.get_runtime", return_value=_runtime()),
            patch("tui.app.run_query_traced", return_value=_result(text_block)),
        ):
            app = FlokiApp()
            async with app.run_test(size=(60, 24)) as pilot:
                await _wait_until(
                    pilot,
                    lambda: not app.query_one("#query-input", Input).disabled,
                    "runtime did not enable the composer",
                )
                composer = app.query_one("#query-input", Input)
                composer.value = "Show the detailed result"
                await pilot.press("enter")
                await _wait_until(
                    pilot,
                    lambda: app._active_request_id is None,
                    "query did not return the shell to idle",
                )
                await pilot.pause(0.1)

                chat = app.query_one("#chat-view", VerticalScroll)
                assert chat.max_scroll_y > 0
                assert chat.scroll_y == chat.max_scroll_y

    asyncio.run(run())


@pytest.mark.parametrize("size", [(100, 32), (60, 24)])
def test_tui_tall_answer_stays_inside_scrollable_transcript(
    size: tuple[int, int],
) -> None:
    """Tall nested answer content cannot displace fixed UI regions."""

    async def run() -> None:
        with patch("tui.app.get_runtime", return_value=_runtime()):
            app = FlokiApp()
            async with app.run_test(size=size) as pilot:
                await pilot.pause(0.1)
                welcome = app.query_one(WelcomePanel)
                chat = app.query_one("#chat-view", VerticalScroll)
                command_deck = app.query_one("#command-deck", Vertical)
                response = AssistantResponse()
                await chat.mount(response)
                response.mount(
                    Markdown(
                        "\n".join(
                            f"- Detailed result {index}: " + "evidence " * 12
                            for index in range(50)
                        )
                    )
                )
                await pilot.pause(0.2)
                chat.scroll_end(animate=False)
                await pilot.pause(0.1)

                assert chat.max_scroll_y > 0
                assert chat.scroll_y > 0
                assert welcome.region.bottom <= chat.region.y
                assert chat.region.bottom <= command_deck.region.y
                assert command_deck.region.bottom <= size[1]
                assert app.query_one("#query-input", Input).region.bottom <= size[1]

    asyncio.run(run())


def test_tui_shutdown_calls_langfuse_cleanup_once() -> None:
    async def run() -> None:
        with (
            patch("tui.app.get_runtime", return_value=_runtime()),
            patch("tui.app.finish_langfuse") as finish,
            patch("tui.app.flush_langfuse_on_exit") as flush,
        ):
            app = FlokiApp()
            async with app.run_test() as pilot:
                await _wait_until(
                    pilot,
                    lambda: not app.query_one("#query-input", Input).disabled,
                    "runtime did not enable the composer",
                )

            finish.assert_called_once_with()
            flush.assert_called_once_with()

    asyncio.run(run())


@pytest.mark.parametrize("size", [(100, 32), (60, 24)])
def test_tui_keeps_welcome_and_command_deck_fixed_while_transcript_scrolls(
    size: tuple[int, int],
) -> None:
    """The transcript may scroll, but the command surface must not move with it."""

    async def run() -> None:
        with patch("tui.app.get_runtime", return_value=_runtime()):
            app = FlokiApp()
            async with app.run_test(size=size) as pilot:
                await pilot.pause(0.1)

                welcome = app.query_one(WelcomePanel)
                chat = app.query_one("#chat-view", VerticalScroll)

                # These direct-screen regions are the layout boundary that keeps
                # identity and controls visible while only chat history scrolls.
                assert welcome.parent is app.screen
                assert chat.parent is app.screen

                command_deck = app.query_one("#command-deck", Vertical)
                composer = command_deck.query_one("#query-input", Input)
                status = command_deck.query_one("#status-bar")
                key_help = command_deck.query_one("#key-help")
                assert composer.parent is command_deck
                assert status.parent is command_deck
                assert key_help.parent is command_deck

                welcome_y = welcome.region.y
                await chat.mount(
                    *[
                        UserPrompt(f"Question {index}: " + "details " * 12)
                        for index in range(20)
                    ]
                )
                await pilot.pause(0.1)
                chat.scroll_end(animate=False)
                await pilot.pause(0.1)

                assert chat.scroll_y > 0
                assert welcome.region.y == welcome_y
                assert welcome.region.y >= 0
                assert command_deck.region.bottom <= size[1]

    asyncio.run(run())
