#!/usr/bin/env python3
"""Capture README banner PNG + demo GIF from the One Dark TUI (mocked agent)."""

from __future__ import annotations

import asyncio
import io
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import cairosvg
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
AGENT = SRC / "agent"
sys.path[:0] = [str(SRC), str(AGENT)]

from tui.app import FlokiApp  # noqa: E402

ASSETS = ROOT / "assets"
BANNER_OUT = ASSETS / "floki-banner-2.png"
GIF_OUT = ASSETS / "floki-demo.gif"
TERM_SIZE = (120, 36)
GIF_WIDTH = 1200

DEMO_QUERY = "What experiments are in the tracking server?"

DEMO_RESULT = {
    "structured_response": {
        "blocks": [
            {
                "type": "text",
                "markdown": (
                    "## Experiments\n\n"
                    "Found **3** experiments on the tracking server:\n\n"
                    "- `customer-churn` — classification runs comparing tree and linear models\n"
                    "- `fraud-detection` — ranking runs with F1 as the primary metric\n"
                    "- `Default` — scratch / ad-hoc runs"
                ),
            },
            {
                "type": "table",
                "markdown": (
                    "| Experiment | Runs | Best metric |\n"
                    "| --- | ---: | --- |\n"
                    "| customer-churn | 24 | accuracy **0.91** |\n"
                    "| fraud-detection | 11 | f1 **0.87** |\n"
                    "| Default | 3 | — |"
                ),
            },
        ]
    }
}


def _fake_runtime() -> MagicMock:
    rt = MagicMock()
    rt.config = {
        "llm": {
            "gemini_model": "gemini-2.5-flash",
            "formatter_model": "gemini-2.5-flash-lite",
        }
    }
    rt.fuse_client = None
    rt.conversation_id = "demo"
    return rt


def _fake_run_query(user_query: str, on_status=None) -> dict:
    sequence = [
        ("thinking", 0.7),
        ("listing", 0.9),
        ("scanning", 0.8),
        ("counting", 0.7),
        ("formatting", 0.6),
    ]
    for verb, delay in sequence:
        if on_status is not None:
            on_status(verb)
        time.sleep(delay)
    return DEMO_RESULT


def _svg_to_image(svg: str, width: int = GIF_WIDTH) -> Image.Image:
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=width)
    return Image.open(io.BytesIO(png)).convert("RGBA")


def _crop_banner(img: Image.Image) -> Image.Image:
    """Crop to welcome band; matches prior ~1247x277 README banner proportions."""
    w, h = img.size
    # Keep top ~42% (banner + welcome copy), full width
    top = int(h * 0.04)
    bottom = int(h * 0.46)
    cropped = img.crop((0, top, w, bottom))
    # Trim excess empty sides a bit
    return cropped


async def main() -> None:
    frames: list[tuple[Image.Image, int]] = []

    patches = [
        patch("tui.app.get_runtime", side_effect=_fake_runtime),
        patch("tui.app.run_query_traced", side_effect=_fake_run_query),
        patch("tui.app.finish_langfuse"),
        patch("tui.app.flush_langfuse_on_exit"),
    ]
    for p in patches:
        p.start()
    try:
        app = FlokiApp()
        async with app.run_test(size=TERM_SIZE) as pilot:
            # Wait for mocked init
            for _ in range(40):
                await pilot.pause(0.05)
                if app._agent_ready:
                    break
            await pilot.pause(0.2)

            # Banner: welcome screen
            welcome = _svg_to_image(app.export_screenshot(title="Floki"))
            banner = _crop_banner(welcome)
            banner.convert("RGB").save(BANNER_OUT, optimize=True)
            print(f"wrote {BANNER_OUT} ({banner.size[0]}x{banner.size[1]})")

            frames.append((_svg_to_image(app.export_screenshot(title="Floki")), 1800))

            # Type query with a couple mid-type frames
            await pilot.click("#query-input")
            half = DEMO_QUERY[: len(DEMO_QUERY) // 2]
            await pilot.press(*half)
            await pilot.pause(0.15)
            frames.append((_svg_to_image(app.export_screenshot(title="Floki")), 700))
            await pilot.press(*DEMO_QUERY[len(half) :])
            await pilot.pause(0.15)
            frames.append((_svg_to_image(app.export_screenshot(title="Floki")), 900))

            await pilot.press("enter")

            # Capture while agent "runs" (thinking → tool verbs → format)
            for _ in range(30):
                await pilot.pause(0.25)
                frames.append((_svg_to_image(app.export_screenshot(title="Floki")), 280))
                if app._active_request_id is None:
                    break

            await pilot.pause(0.4)
            frames.append((_svg_to_image(app.export_screenshot(title="Floki")), 2800))
    finally:
        for p in patches:
            p.stop()

    # Assemble GIF (palette-quantized)
    rgb_frames = [im.convert("RGB").quantize(colors=256, method=Image.MEDIANCUT) for im, _ in frames]
    durations = [d for _, d in frames]
    rgb_frames[0].save(
        GIF_OUT,
        save_all=True,
        append_images=rgb_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    print(f"wrote {GIF_OUT} ({len(frames)} frames, {GIF_OUT.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    asyncio.run(main())
