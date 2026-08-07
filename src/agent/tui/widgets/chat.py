from __future__ import annotations

from textual.containers import Vertical
from textual.markup import escape
from textual.timer import Timer
from textual.widgets import Static

from agent.tool_status import DEFAULT_VERB, FORMAT_VERB, TOOL_VERBS

_PULSE = (".  ", ".. ", "...")
_KNOWN_VERBS = frozenset((*TOOL_VERBS.values(), DEFAULT_VERB, FORMAT_VERB))


class UserPrompt(Static):
    """A submitted user prompt in the transcript."""

    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = text

    def on_mount(self) -> None:
        self.update(f"[bold #61afef]You[/]\n{escape(self._text)}")


class FlokiLabel(Static):
    """Small assistant identity label mounted before an answer."""

    def on_mount(self) -> None:
        self.update("FLOKI")


class QueryLoading(Static):
    """Small in-transcript work indicator with a terminal-safe pulse."""

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

    def set_verb(self, verb: str | None) -> None:
        """Set a known work verb, or show the safe ``Thinking`` fallback."""
        normalized = (verb or "").strip().lower().rstrip(".").rstrip("…")
        self._verb = normalized if normalized in _KNOWN_VERBS else DEFAULT_VERB
        self._paint()

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_PULSE)
        self._paint()

    def _paint(self) -> None:
        label = f"{self._verb.capitalize()}…"
        pulse = _PULSE[self._frame]
        self.update(
            f"[bold #e5c07b]Work:[/] [#e5c07b]{escape(label)}[/] [dim]{pulse}[/]"
        )


class SystemNotice(Static):
    """A short, escaped transcript notice for errors and cancellation."""

    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = text

    def on_mount(self) -> None:
        self.update(f"[bold #e06c75]Notice:[/] {escape(self._text)}")


class AssistantResponse(Vertical):
    """Container for formatter-produced response blocks."""
