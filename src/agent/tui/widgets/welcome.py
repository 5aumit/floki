from __future__ import annotations

from textual.widgets import Static

from console_ui import _BANNER_COLORS, _BANNER_LINES


class WelcomePanel(Static):
    """Floki welcome banner and instructions."""

    def __init__(self, app_name: str = "Floki", version: str = "v0.1", **kwargs) -> None:
        super().__init__(**kwargs)
        self._app_name = app_name
        self._version = version

    def on_mount(self) -> None:
        lines: list[str] = []
        for line, color in zip(_BANNER_LINES, _BANNER_COLORS):
            lines.append(f"[bold {color}]{line}[/]")
        lines.append("")
        lines.append(f"[bold white on dark_green]  {self._app_name} {self._version} 🧭  [/]")
        lines.append("")
        lines.append("[cyan]Welcome! Type your question and press Enter. Type 'exit' to quit.[/]")
        lines.append(
            "[dim]Tool outputs (tables/JSON) will appear below the assistant message when available.[/]"
        )
        self.update("\n".join(lines))
