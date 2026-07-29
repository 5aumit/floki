import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from textual.widgets import Input

from tui.app import FlokiApp


def test_tui_enables_input_after_runtime_initializes() -> None:
    runtime = SimpleNamespace(
        config={"llm": {"gemini_model": "agent-model", "formatter_model": "formatter-model"}},
        fuse_client=None,
        conversation_id=None,
    )

    async def run() -> None:
        with patch("tui.app.get_runtime", return_value=runtime):
            app = FlokiApp()
            async with app.run_test() as pilot:
                await pilot.pause(0.1)
                assert app.query_one("#query-input", Input).disabled is False
                assert "Ready" in str(app.query_one("#status-bar").render())

    asyncio.run(run())
