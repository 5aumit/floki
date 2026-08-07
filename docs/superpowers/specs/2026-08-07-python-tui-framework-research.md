# Python TUI framework research

Date: 2026-08-07
Scope: Select a Python framework for the Floki coding-agent TUI.

## Decision

Keep **Textual**. It is the best fit for this project.

Floki already uses Textual in `src/agent/tui/`, includes it in `environment.yml`,
and has a headless Textual test. Textual gives the required application structure,
input, scrolling, Markdown output, styling, and background work without a framework
change. It also uses Rich renderables, so existing Rich output knowledge stays useful.

This choice supports a coding-agent style interface: a fixed identity area, a scrolling
conversation, a command input, clear agent activity, and rich Markdown answers.

## Required layout rule

The banner and welcome text must stay visible after the conversation scrolls. Do not
put them inside the conversation `VerticalScroll`.

Put a small `WelcomePanel` above the conversation and give it `dock: top`, or use a
top layout row that does not scroll. Textual documents that a docked widget stays fixed
at an edge and names sticky headers and footers as a use case. The conversation alone
should scroll. Keep the input fixed at the bottom.

This rule is more important than the framework choice. The current `FlokiApp` mounts
`WelcomePanel` inside `#chat-view`, so the current layout does not meet the persistent
welcome requirement.

## Comparison

| Framework | Fit for Floki | Main benefit | Main cost | Result |
| --- | --- | --- | --- | --- |
| [Textual](https://textual.textualize.io/) | Strong | Full application framework with widgets, layouts, CSS-like styling, worker support, Markdown, input, logs, and tests. | Team must learn Textual's widget and message model. | Use it. |
| [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/en/stable/) | Good | Very capable input editing, key bindings, full-screen layout, and native `asyncio`. | More low-level assembly for a polished chat UI, scrolling transcript, and persistent regions. | Keep only for the legacy REPL. Do not move the new TUI to it. |
| [Rich](https://rich.readthedocs.io/) | Limited | Excellent terminal rendering, Markdown, tables, and live updates. It is already a project dependency. | It provides renderables and live displays, not a complete interactive widget application model. Input, focus, event routing, and application state need more custom code. | Use through Textual or retain for the REPL. Do not use it alone for the rework. |

## Evidence

### Textual

- The official [layout guide](https://textual.textualize.io/guide/layout/#docking)
  documents docking as fixed edge placement and gives sticky UI areas as an example.
  This directly supports a persistent banner and a fixed input area.
- The official [worker guide](https://textual.textualize.io/guide/workers/) states that
  slow network, subprocess, and compute work must not block UI updates. Its worker API
  provides async and thread worker modes. Thread workers use
  [`call_from_thread`](https://textual.textualize.io/guide/workers/#posting-messages)
  for safe UI changes. This matches the current blocking agent and formatter calls.
- The official [Markdown widget reference](https://textual.textualize.io/widgets/markdown/)
  describes a widget that displays Markdown and can append fragments. Its streaming
  section explains how it combines many quick updates when needed. This supports
  formatted answers now and token streaming later.
- The official [testing guide](https://textual.textualize.io/guide/testing/) documents
  `App.run_test()` and `Pilot` to test user input without a real terminal. The project
  already uses this pattern in `tests/test_tui_app.py`.

### prompt_toolkit

- Its official [full-screen application guide](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/full_screen_apps.html)
  says it can build complex full-screen terminal applications from layouts and key
  bindings.
- Its official [asyncio guide](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/advanced_topics/asyncio.html)
  says version 3 uses `asyncio` natively and provides `run_async` for an existing event
  loop.
- These features make it a sound REPL and input framework. They do not outweigh the
  extra layout and widget work compared with the existing Textual implementation.

### Rich

- The official [Live display guide](https://rich.readthedocs.io/en/stable/live.html)
  documents a live renderable with an adjustable refresh rate and an alternate-screen
  mode. It can make a simple full-screen display with `Layout`.
- The official [Layout guide](https://rich.readthedocs.io/en/stable/layout.html)
  documents split regions. Rich does not document an application-level event, focus, or
  widget system comparable to Textual. A coding-agent UI would need substantial custom
  control code around it.

## Implementation direction

1. Keep `FlokiApp` and the current Textual dependency.
2. Move `WelcomePanel` out of `#chat-view`; make it a compact, fixed top area.
3. Keep only answers, user prompts, and tool activity in the scrolling area.
4. Keep `Input` fixed at the bottom and use Textual workers for agent work. Do not
   update widgets directly from a thread worker.
5. Add headless tests for the fixed welcome area, input state, and response rendering.
6. Use a terminal recording or screenshot for a final visual check at normal and small
   terminal sizes.

## Source policy

All external links above are official framework documentation. Project facts come from
the current repository files named in this note.
