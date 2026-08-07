"""The responsive Textual shell for the Floki agent.

This module owns only display state. The agent, formatter, memory, and
Langfuse work remain in :mod:`langchain_agent` and run through the one traced
query entry point.
"""

from __future__ import annotations

from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.events import Resize
from textual.markup import escape
from textual.theme import Theme
from textual.widgets import Input, Static
from textual.widget import MountError
from textual.worker import Worker

from agent.tool_status import FORMAT_VERB, TOOL_VERBS
from langchain_agent import (
    finish_langfuse,
    flush_langfuse_on_exit,
    get_runtime,
    run_query_traced,
)
from tui.blocks import render_result
from tui.widgets.chat import AssistantResponse, QueryLoading, SystemNotice, UserPrompt
from tui.widgets.welcome import WelcomePanel


# One Dark - matches docs/artifacts/TUI-mockups/tui-onedark-v2.html.
FLOKI_ONE_DARK = Theme(
    name="floki-one-dark",
    primary="#61afef",
    secondary="#56b6c2",
    accent="#c678dd",
    foreground="#abb2bf",
    background="#282c34",
    success="#98c379",
    warning="#e5c07b",
    error="#e06c75",
    surface="#21252b",
    panel="#2c313a",
    dark=True,
)


class FlokiApp(App):
    """Fixed command deck around one background agent request at a time."""

    CSS_PATH = "styles/floki.tcss"
    AUTO_FOCUS = "#query-input"
    BINDINGS = [
        ("ctrl+c", "cancel_or_quit", "Cancel or quit"),
        ("ctrl+q", "quit", "Quit"),
    ]
    _FULL_KEY_HELP = "Enter send  |  Ctrl+C cancel/exit  |  Ctrl+Q exit"
    _COMPACT_KEY_HELP = "Enter  |  Ctrl+C stop  |  Ctrl+Q exit"
    _FULL_PLACEHOLDER = "Ask about experiments, runs, or metrics. 'exit' to quit."
    _COMPACT_PLACEHOLDER = "Ask Floki. 'exit' to quit."

    def __init__(self) -> None:
        super().__init__()
        self._agent_ready = False
        self._query_worker: Worker | None = None
        self._request_counter = 0
        self._active_request_id: int | None = None
        self._active_loading: QueryLoading | None = None
        self._active_response: AssistantResponse | None = None
        self._active_worker_started = False
        self._cancellation_requested = False
        self._exit_pending = False
        self._shutdown_complete = False
        self._ready_status = "Ready"
        self._status_text = "Initializing agent…"

    def compose(self) -> ComposeResult:
        """Create fixed regions around the only scrolling transcript."""
        yield WelcomePanel(id="floki-welcome")
        yield VerticalScroll(id="chat-view")
        with Vertical(id="command-deck"):
            yield Input(
                placeholder=self._FULL_PLACEHOLDER,
                id="query-input",
                disabled=True,
            )
            yield Static("Initializing agent…", id="status-bar")
            yield Static(self._FULL_KEY_HELP, id="key-help")

    def on_mount(self) -> None:
        self.register_theme(FLOKI_ONE_DARK)
        self.theme = FLOKI_ONE_DARK.name
        self.call_after_refresh(self._sync_responsive_layout)
        self._init_runtime()

    def on_resize(self, event: Resize) -> None:
        """Apply CSS classes because Textual 8 does not support TCSS media queries."""
        self._sync_responsive_layout(event.size.width)

    def _sync_responsive_layout(self, width: int | None = None) -> None:
        """Keep compact terminal controls readable without invalid TCSS media rules."""
        width = self.size.width if width is None else width
        is_narrow = 60 <= width < 80
        is_compact = width < 60
        self.screen.set_class(is_narrow, "narrow")
        self.screen.set_class(is_compact, "compact")

        try:
            self.query_one("#key-help", Static).update(
                self._COMPACT_KEY_HELP if is_compact else self._FULL_KEY_HELP
            )
            self.query_one("#query-input", Input).placeholder = (
                self._COMPACT_PLACEHOLDER if is_compact else self._FULL_PLACEHOLDER
            )
        except MountError:
            # A resize can arrive during application shutdown.
            return
        self._set_status(self._status_text)

    @work(thread=True, exclusive=True, group="runtime", exit_on_error=False)
    def _init_runtime(self) -> None:
        """Load the expensive runtime outside the Textual event loop."""
        try:
            runtime = get_runtime()
        except Exception as exc:
            self.call_from_thread(self._on_init_failed, str(exc))
            return
        self.call_from_thread(self._on_init_ready, runtime)

    def _on_init_failed(self, error: str) -> None:
        """Leave an actionable, safe error state when setup fails."""
        self._agent_ready = False
        self._set_status(f"Init failed: {escape(self._short_error(error))} Ctrl+Q exit")

    def _on_init_ready(self, runtime: Any) -> None:
        """Store non-secret runtime facts and make the composer usable."""
        llm_config = getattr(runtime, "config", {}).get("llm", {}) or {}
        agent_model = self._safe_status_value(llm_config.get("gemini_model"), "agent")
        formatter_model = self._safe_status_value(
            llm_config.get("formatter_model"), "formatter"
        )
        tracing = "Langfuse on" if getattr(runtime, "fuse_client", None) else "Langfuse off"
        self._ready_status = (
            f"Ready | {agent_model} | fmt {formatter_model} | {tracing}"
        )
        self._agent_ready = True
        self._set_status(self._ready_status)
        try:
            composer = self.query_one("#query-input", Input)
            composer.disabled = False
            composer.focus()
        except Exception:
            # The app may be closing while its init worker finishes.
            pass

    @staticmethod
    def _safe_status_value(value: Any, fallback: str) -> str:
        """Return a compact status value without exposing structured config."""
        if not isinstance(value, str):
            return fallback
        compact = " ".join(value.split())
        return compact[:48] if compact else fallback

    @staticmethod
    def _short_error(_error: object) -> str:
        """Return useful guidance without displaying untrusted exception text."""
        return "Check your configuration or connection, then try again."

    def _set_status(self, text: str) -> None:
        self._status_text = text
        displayed_text = "Ready" if text == self._ready_status and (
            self.screen.has_class("narrow") or self.screen.has_class("compact")
        ) else text
        try:
            self.query_one("#status-bar", Static).update(displayed_text)
        except MountError:
            # A late thread callback must not crash a closing TUI.
            pass

    def _scroll_transcript_to_end(self, *, immediate: bool = False) -> None:
        try:
            self.query_one("#chat-view", VerticalScroll).scroll_end(
                animate=False, immediate=immediate
            )
        except Exception:
            pass

    async def _scroll_after_response_mounts(self, mounts: list) -> None:
        """Scroll after each response child has mounted and layout is complete."""
        for mount in mounts:
            await mount
        try:
            chat = self.query_one("#chat-view", VerticalScroll)
            await chat.wait_for_refresh()
            chat.scroll_end(animate=False, immediate=True)
            await chat.wait_for_refresh()
            chat.scroll_end(animate=False, immediate=True)
        except Exception:
            pass

    def _has_active_request(self) -> bool:
        return self._active_request_id is not None

    def _is_active_request(self, request_id: int) -> bool:
        return self._active_request_id == request_id

    @staticmethod
    def _display_work_status(verb: str) -> str:
        """Map runtime tool verbs to the short, sentence-case status contract."""
        normalized = str(verb or "").strip().rstrip(".…").lower()
        if normalized == FORMAT_VERB:
            return "Formatting…"
        if normalized in TOOL_VERBS.values():
            return f"{normalized.capitalize()}…"
        return "Thinking…"

    def _on_query_status(self, request_id: int, verb: str) -> None:
        """Apply a worker status update only to the matching visible request."""
        if not self._is_active_request(request_id) or self._cancellation_requested:
            return
        status = self._display_work_status(verb)
        loading = self._active_loading
        try:
            if loading is not None and loading.is_mounted:
                loading.set_verb(status)
        except Exception:
            pass
        self._set_status(status)

    @on(Input.Submitted, "#query-input")
    def on_query_submitted(self, event: Input.Submitted) -> None:
        """Add transcript rows and start the sole query worker."""
        if not self._agent_ready or self._has_active_request():
            return

        query = (event.value or "").strip()
        if not query:
            return
        event.input.value = ""
        if query.lower() in {"exit", "quit", "/exit", "/quit"}:
            self.action_quit()
            return

        self._request_counter += 1
        request_id = self._request_counter
        self._active_request_id = request_id
        self._active_worker_started = False
        self._cancellation_requested = False

        chat = self.query_one("#chat-view", VerticalScroll)
        loading = QueryLoading()
        response = AssistantResponse()
        self._active_loading = loading
        self._active_response = response
        chat.mount(UserPrompt(query), loading, response)
        event.input.disabled = True
        self._set_status("Agent running…")
        self._scroll_transcript_to_end()
        self._query_worker = self._run_query(request_id, query)

    @work(thread=True, exclusive=True, group="query", exit_on_error=False)
    def _run_query(self, request_id: int, query: str) -> None:
        """Run agent and formatter work, then hand terminal state to the event loop."""
        try:
            self.call_from_thread(self._on_query_worker_started, request_id)
        except RuntimeError:
            return

        # A cancellation may arrive before the executor starts this function.
        # In that case no agent work has begun, so finish as cancelled instead
        # of relying on Worker.cancel() to interrupt a still-pending task.
        if self._cancellation_requested:
            try:
                self.call_from_thread(self._on_query_finished, request_id, None, None)
            except RuntimeError:
                pass
            return

        def on_status(verb: str) -> None:
            try:
                self.call_from_thread(self._on_query_status, request_id, verb)
            except RuntimeError:
                # Textual has already shut down. The query may still end safely.
                pass

        try:
            result = run_query_traced(query, on_status=on_status)
        except Exception as exc:
            result: Any = None
            error: str | None = str(exc)
        else:
            error = None

        try:
            self.call_from_thread(self._on_query_finished, request_id, result, error)
        except RuntimeError:
            # There is no UI to update after application shutdown.
            pass

    def _on_query_worker_started(self, request_id: int) -> None:
        """Record that the thread is live before asking Textual to cancel it."""
        if not self._is_active_request(request_id):
            return
        self._active_worker_started = True
        if self._cancellation_requested:
            worker = self._query_worker
            if worker is not None:
                worker.cancel()

    def _remove_active_loading(self) -> None:
        loading = self._active_loading
        self._active_loading = None
        try:
            if loading is not None and loading.is_mounted:
                loading.remove()
        except Exception:
            pass

    def _restore_idle_composer(self) -> None:
        """Return to idle only after the current query has reached a terminal UI state."""
        self._active_request_id = None
        self._active_response = None
        self._active_worker_started = False
        self._query_worker = None
        self._cancellation_requested = False

        if self._exit_pending:
            self._exit_pending = False
            self.exit()
            return

        self._set_status(self._ready_status)
        try:
            composer = self.query_one("#query-input", Input)
            composer.disabled = False
            composer.focus()
        except Exception:
            pass

    def _finish_cancelled_request(self, request_id: int) -> None:
        """Discard a late result and record cancellation after worker work ends."""
        if not self._is_active_request(request_id):
            return
        response = self._active_response
        self._remove_active_loading()
        try:
            if response is not None and response.is_mounted:
                response.remove()
        except Exception:
            pass
        try:
            self.query_one("#chat-view", VerticalScroll).mount(
                SystemNotice("Request cancelled.")
            )
        except Exception:
            pass
        self._scroll_transcript_to_end()
        self._restore_idle_composer()

    def _on_query_finished(
        self,
        request_id: int,
        result: Any,
        error: str | None,
    ) -> None:
        """Render one final state, ignoring work from stale request identifiers."""
        if not self._is_active_request(request_id):
            return
        if self._cancellation_requested:
            self._finish_cancelled_request(request_id)
            return

        response = self._active_response
        self._remove_active_loading()
        if response is not None and response.is_mounted:
            if error is not None:
                try:
                    response.mount(SystemNotice(f"Query failed: {self._short_error(error)}"))
                except MountError:
                    # The app is closing before the terminal notice can mount.
                    pass
            else:
                try:
                    mounts = render_result(response, result)
                    self.run_worker(
                        self._scroll_after_response_mounts(mounts),
                        group="response-scroll",
                        exclusive=True,
                        exit_on_error=False,
                    )
                except MountError:
                    # The response was unmounted during shutdown.
                    pass
                except Exception:
                    try:
                        response.mount(
                            SystemNotice("Could not display this response. Please retry.")
                        )
                    except MountError:
                        # The app is closing before the fallback notice can mount.
                        pass
        self._scroll_transcript_to_end()
        self._restore_idle_composer()

    def _request_cancellation(self, *, exit_after_request: bool) -> None:
        """Request cooperative cancellation without allowing a second query."""
        if not self._has_active_request():
            if exit_after_request:
                self.exit()
            return

        if exit_after_request:
            self._exit_pending = True
        if self._cancellation_requested:
            return

        self._cancellation_requested = True
        self._set_status("Stopping current request…")
        worker = self._query_worker
        if worker is not None and self._active_worker_started:
            worker.cancel()

    def action_cancel_or_quit(self) -> None:
        """Ctrl+C exits at idle and otherwise requests best-effort cancellation."""
        if not self._has_active_request():
            self.exit()
            return
        self._request_cancellation(exit_after_request=False)

    def action_quit(self) -> None:
        """Ctrl+Q exits at idle, or exits after an active request has ended."""
        self._request_cancellation(exit_after_request=True)

    def _cleanup_runtime(self) -> None:
        """Run the TUI-owned Langfuse shutdown sequence exactly once."""
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        finish_langfuse()
        flush_langfuse_on_exit()

    def on_unmount(self) -> None:
        self._cleanup_runtime()


def run_tui() -> None:
    """Run the Textual interface."""
    FlokiApp().run()
