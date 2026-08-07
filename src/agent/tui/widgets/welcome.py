from __future__ import annotations

from textual.markup import escape
from textual.widgets import Static


class WelcomePanel(Static):
    """Compact, persistent identity panel for the Floki command surface."""

    def __init__(self, app_name: str = "Floki", version: str = "v0.1", **kwargs) -> None:
        kwargs.setdefault("id", "floki-welcome")
        super().__init__(**kwargs)
        self._app_name = app_name
        self._version = version

    def on_mount(self) -> None:
        wordmark = escape(self._app_name.upper())
        version = escape(self._version)
        self.update(
            f"[bold #61afef]{wordmark}[/] [dim]{version}[/]  "
            "[dim]DATA -> RUNS -> EVIDENCE[/]\n"
            "Ask about your MLflow experiments.  "
            "[dim]Enter sends. Ctrl+C stops a request or exits.[/]"
        )
