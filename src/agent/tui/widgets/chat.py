from __future__ import annotations

from textual.containers import Vertical
from textual.markup import escape
from textual.timer import Timer
from textual.widgets import Static

from agent.tool_status import DEFAULT_VERB

_PULSE = ("·  ", "·· ", "···", " ··")


class UserPrompt(Static):
    """User message bubble."""

    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = text

    def on_mount(self) -> None:
        self.update(f"[bold #61afef]You[/]\n{escape(self._text)}")


class FlokiLabel(Static):
    """Floki assistant badge."""

    def on_mount(self) -> None:
        self.update(" Floki ")


class QueryLoading(Static):
    """In-chat loading row: pulse + tool verb (fallback thinking)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._verb = DEFAULT_VERB
        self._frame = 0
        self._timer: Timer | None = None

    def on_mount(self) -> None:
        self._paint()
        self._timer = self.set_interval(0.35, self._tick)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def set_verb(self, verb: str) -> None:
        self._verb = (verb or DEFAULT_VERB).strip() or DEFAULT_VERB
        self._paint()

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_PULSE)
        self._paint()

    def _paint(self) -> None:
        pulse = _PULSE[self._frame]
        self.update(f"[bold #61afef]{pulse}[/] [#61afef]{escape(self._verb)}[/]")


class AssistantResponse(Vertical):
    """Container for formatter-produced response blocks."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
