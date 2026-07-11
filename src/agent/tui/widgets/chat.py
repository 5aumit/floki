from __future__ import annotations

from textual.containers import Vertical
from textual.markup import escape
from textual.widgets import Static


class UserPrompt(Static):
    """User message bubble."""

    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = text

    def on_mount(self) -> None:
        self.update(f"[bold blue]You[/bold blue]\n{escape(self._text)}")


class FlokiLabel(Static):
    """Floki assistant badge."""

    def on_mount(self) -> None:
        self.update(" Floki ")


class QueryLoading(Static):
    """In-chat loading row while the query worker runs."""

    def on_mount(self) -> None:
        self.update("… thinking")


class AssistantResponse(Vertical):
    """Container for formatter-produced response blocks."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
