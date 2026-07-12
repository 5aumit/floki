"""Floki Textual TUI."""

__all__ = ["FlokiApp"]


def __getattr__(name: str):
    if name == "FlokiApp":
        from tui.app import FlokiApp

        return FlokiApp
    raise AttributeError(name)
