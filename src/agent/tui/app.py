from __future__ import annotations

from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.markup import escape
from textual.widgets import Input, Static
from textual.worker import Worker, WorkerState

from langchain_agent import (
    finish_langfuse,
    flush_langfuse_on_exit,
    get_runtime,
    run_query_traced,
)
from tui.blocks import render_result
from tui.widgets.chat import AssistantResponse, QueryLoading, UserPrompt
from tui.widgets.welcome import WelcomePanel


class FlokiApp(App):
    """Textual chat shell around run_query_traced / BlockResponse rendering."""

    CSS_PATH = "styles/floki.tcss"
    AUTO_FOCUS = "#query-input"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._agent_ready = False
        self._busy = False
        self._query_worker: Worker | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat-view"):
            yield WelcomePanel()
        yield Static(id="tool-trace")
        yield Input(
            placeholder="Type your question. 'exit' to quit.",
            id="query-input",
            disabled=True,
        )
        yield Static("Initializing agent…", id="status-bar")

    def on_mount(self) -> None:
        self._init_runtime()

    @work(thread=True)
    def _init_runtime(self) -> None:
        try:
            rt = get_runtime()
        except Exception as exc:
            self.call_from_thread(self._on_init_failed, str(exc))
            return
        self.call_from_thread(self._on_init_ready, rt)

    def _on_init_failed(self, err: str) -> None:
        self.query_one("#status-bar", Static).update(
            f"Init failed: {err}  (ctrl+q to quit)"
        )
        # Keep input disabled; user can still quit via bindings.

    def _on_init_ready(self, rt: Any) -> None:
        llm = rt.config.get("llm", {}) or {}
        models = f"{llm.get('gemini_model', '?')} / fmt {llm.get('formatter_model', '?')}"
        if rt.fuse_client is not None:
            status = f"Langfuse on | {rt.conversation_id or 'n/a'} | {models}"
        else:
            status = f"Ready | {models}"
        self.query_one("#status-bar", Static).update(status)

        inp = self.query_one("#query-input", Input)
        inp.disabled = False
        self._agent_ready = True
        inp.focus()

    @on(Input.Submitted, "#query-input")
    def on_query_submitted(self, event: Input.Submitted) -> None:
        if not self._agent_ready or self._busy:
            return
        text = (event.value or "").strip()
        event.input.value = ""
        if not text:
            return
        if text.lower() in {"exit", "quit", "/exit", "/quit"}:
            self.action_quit()
            return

        chat = self.query_one("#chat-view", VerticalScroll)
        chat.mount(UserPrompt(text))
        loading = QueryLoading()
        response = AssistantResponse()
        chat.mount(loading)
        chat.mount(response)
        chat.scroll_end(animate=False)

        self._busy = True
        event.input.disabled = True
        self.query_one("#status-bar", Static).update("Agent running…")
        self._query_worker = self._run_query(text, response, loading)

    @work(thread=True, exclusive=True, group="query")
    def _run_query(self, query: str, response: AssistantResponse, loading: QueryLoading) -> None:
        try:
            result = run_query_traced(query)
            self.call_from_thread(self._on_query_done, response, loading, result, None)
        except Exception as exc:
            self.call_from_thread(self._on_query_done, response, loading, None, str(exc))

    def _on_query_done(
        self,
        response: AssistantResponse,
        loading: QueryLoading,
        result: Any,
        err: str | None,
    ) -> None:
        if not self.is_running:
            return
        try:
            loading.remove()
        except Exception:
            pass

        try:
            if err is not None:
                response.mount(Static(f"[red]Error:[/red] {escape(err)}"))
            else:
                render_result(response, result)
            self.query_one("#chat-view", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

        self._busy = False
        self._query_worker = None
        try:
            inp = self.query_one("#query-input", Input)
            inp.disabled = False
            inp.focus()
            self.query_one("#status-bar", Static).update("Ready")
        except Exception:
            pass

    def action_quit(self) -> None:
        worker = self._query_worker
        if worker is not None and worker.state == WorkerState.RUNNING:
            worker.cancel()
        self.exit()

    def on_unmount(self) -> None:
        finish_langfuse()
        flush_langfuse_on_exit()


def run_tui() -> None:
    FlokiApp().run()
