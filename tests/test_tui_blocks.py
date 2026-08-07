import asyncio
from pathlib import Path

from rich.markdown import Markdown as RichMarkdown
from textual.app import App, ComposeResult
from textual.containers import HorizontalScroll
from textual.widgets import Markdown, Static

from tui.blocks import (
    StaticBlock,
    extract_structured,
    fallback_content,
    render_result,
)
from tui.widgets.chat import AssistantResponse, FlokiLabel


class BlockRenderApp(App):
    """Small Textual host for observing the renderer's mounted widgets."""

    CSS_PATH = Path(__file__).parent.parent / "src/agent/tui/styles/floki.tcss"

    def __init__(self) -> None:
        super().__init__()
        self.response = AssistantResponse()

    def compose(self) -> ComposeResult:
        yield self.response


def _render(result: object) -> list[object]:
    async def run() -> list[object]:
        app = BlockRenderApp()
        async with app.run_test() as pilot:
            render_result(app.response, result)
            await pilot.pause()
            await pilot.pause(0.1)
            return list(app.response.children)

    return asyncio.run(run())


def test_extract_structured_prefers_structured_response() -> None:
    direct = {"blocks": [{"type": "text", "markdown": "Direct"}]}
    result = {
        "structured_response": {"blocks": [{"type": "text", "markdown": "Preferred"}]},
        **direct,
    }

    assert extract_structured(result) == result["structured_response"]


def test_extract_structured_accepts_direct_block_response() -> None:
    result = {"blocks": [{"type": "text", "markdown": "Hello"}]}

    assert extract_structured(result) is result


def test_fallback_content_uses_final_agent_message() -> None:
    result = {"messages": [{"content": "Earlier"}, {"content": "Final answer"}]}

    assert fallback_content(result) == "Final answer"


def test_fallback_content_hides_unusable_dictionary_data() -> None:
    assert fallback_content({"unrelated": "internal data"}) == ""


def test_empty_result_mounts_a_clear_no_result_message() -> None:
    children = _render(None)

    assert isinstance(children[0], FlokiLabel)
    assert isinstance(children[1], Markdown)


def test_malformed_blocks_mount_safe_plain_text_fallbacks() -> None:
    children = _render(
        {
            "blocks": [
                None,
                {"type": "text", "markdown": ""},
                {"type": "unknown", "markdown": "Not supported"},
            ]
        }
    )

    assert isinstance(children[0], FlokiLabel)
    assert all(isinstance(child, StaticBlock) for child in children[1:])


def test_render_result_mounts_text_and_scrollable_table_markdown() -> None:
    table_markdown = "| Run | F1 |\n| --- | --- |\n| best | 0.93 |"

    async def run() -> None:
        app = BlockRenderApp()
        async with app.run_test() as pilot:
            render_result(
                app.response,
                {
                    "structured_response": {
                        "blocks": [
                            {"type": "text", "markdown": "The best run is below."},
                            {"type": "table", "markdown": table_markdown},
                        ]
                    }
                },
            )
            await pilot.pause()
            await pilot.pause(0.1)

            children = list(app.response.children)
            assert isinstance(children[0], FlokiLabel)
            assert isinstance(children[1], Markdown)
            assert isinstance(children[2], HorizontalScroll)
            assert children[2].has_class("table-scroll")
            assert len(children[2].children) == 1
            table = children[2].children[0]
            assert isinstance(table, Static)
            assert isinstance(table.content, RichMarkdown)

    asyncio.run(run())


def test_wide_table_scrolls_horizontally_without_tall_cell_wrapping() -> None:
    """A narrow terminal keeps table columns intact inside the scroll region."""
    table_markdown = "\n".join(
        [
            "| Experiment name | Validation score | Training duration | Artifact location |",
            "| --- | --- | --- | --- |",
            "| production-candidate | 0.938 | 04:32:18 | models/champion/2026-08-07 |",
        ]
    )

    async def run() -> None:
        app = BlockRenderApp()
        async with app.run_test(size=(60, 24)) as pilot:
            render_result(
                app.response,
                {"blocks": [{"type": "table", "markdown": table_markdown}]},
            )
            await pilot.pause(0.1)
            await pilot.pause(0.1)

            table_scroll = app.response.query_one(HorizontalScroll)
            table = next(iter(table_scroll.children))
            assert isinstance(table, Static)
            assert isinstance(table.content, RichMarkdown)
            assert table_scroll.max_scroll_x > 0
            assert table.region.width > table_scroll.region.width
            assert table.region.height == 6

    asyncio.run(run())
