# Floki Textual TUI — Implementation Plan

Date: 2026-06-28  
Branch: `tui`  
Status: Draft  
Owner: Saumit Paul

## Overview

Replace the current Rich + prompt_toolkit REPL with a **Textual** full-screen TUI while preserving every existing behavior. The agent core (`run_query`, LangGraph, **formatter LLM**, MLflow tools, Langfuse tracing, session memory) stays unchanged; only the presentation and input loop move behind a Textual `App`.

## Architectural decision: keep the formatter

The **formatter LLM stays in the pipeline**. Textual is a rendering layer only; it does not replace `format_to_block_response()`.

| Layer | Responsibility | Changes in TUI branch |
|-------|----------------|----------------------|
| Agent LLM | Tool calling + draft prose | None |
| Formatter LLM | `BlockResponse` with `text` / `table` markdown blocks | None |
| `run_query()` | Agent invoke → formatter → return `structured_response` | None (optional `AgentRuntime` extract for lazy init) |
| UI | Render `structured_response.blocks` | Rich REPL → Textual widgets |

**Why keep the formatter**

- Reliable split between prose (`text` blocks) and comparisons (`table` blocks) without depending on agent markdown habits.
- Grounding rules in `FORMAT_SYSTEM_PROMPT` constrain output to tool results + agent draft.
- Cheaper/faster `formatter_model` handles layout; agent model stays focused on MLflow reasoning.
- REPL and TUI share one output contract (`BlockResponse`); no duplicate render logic per UI.
- Formatter retry + fallback (`_fallback_response`) already handled in `response_formatter.py`.

**Explicitly out of scope**

- Removing the formatter and rendering agent draft + raw tool JSON directly.
- Moving `BlockResponse` structured output onto the main agent.
- Code-driven `DataTable` from tool JSON as a replacement for formatter table blocks (may be added later as an enhancement, not a substitute).

## Goals

- Map **all** current CLI functionality to equivalent TUI behavior.
- Keep `run_query()` as the single orchestration entry point (agent + formatter, no duplicate logic).
- Render formatter-produced `BlockResponse` blocks in Textual (`Markdown` for both `text` and `table` types in phase 1).
- Run long agent+formatter invocations in background workers so the UI stays responsive.
- Preserve Langfuse session lifecycle and graceful shutdown from `app.py`.

## Non-goals (this branch)

- Removing or bypassing the formatter LLM.
- Streaming token output from the agent or formatter (future enhancement).
- Experiment browser sidebar (phase 2 stretch; listed but not required for parity).
- `textual serve` browser deployment.
- Automated pytest harness for the TUI (manual checklist only for now).

---

## Current functionality inventory

| # | Feature | Current location | Behavior |
|---|---------|------------------|----------|
| F1 | Entry point | `src/app.py` → `langchain_agent.main()` | Runs agent; catches `KeyboardInterrupt`; finishes Langfuse run + flush |
| F2 | Config load | `langchain_agent.py` (module init) | Reads `config.json`; merges `GEMINI_API_KEY` from env |
| F3 | LLM setup | `llm/inference_engine.py` | Agent model + formatter model from config |
| F4 | Langfuse tracing | `llm/tracing.py` | Handler, client, `conversation_id`, `lf_run`, `FLUSH_PER_QUERY`, `propagate_attributes` |
| F5 | Agent + tools | `langchain_agent.py` | LangGraph agent, 7 MLflow tools, `InMemorySaver`, error middleware |
| F6 | Session memory | `context_memory.py` | Trim tool traffic after each turn |
| F7 | Two-phase response | `run_query()` | Agent invoke → `format_to_block_response()` → structured dict |
| F8 | Block rendering | `console_ui.py` | `text` / `table` markdown blocks; fallback to last AI message |
| F9 | Welcome banner | `console_ui.print_welcome()` | ASCII art + instructions panel |
| F10 | Startup UX | `main()` | Clear screen, "Initializing…", 3s spinner |
| F11 | Langfuse status line | `main()` | Prints dashboard URL when tracing enabled |
| F12 | Interactive loop | `_interactive_loop()` | Read → query → print → repeat |
| F13 | User input | `_get_user_input()` | prompt_toolkit HTML prompt; Rich/input fallbacks |
| F14 | Exit commands | `_interactive_loop()` | `exit` / `quit` → flush Langfuse → goodbye |
| F15 | EOF / Ctrl+C | `_interactive_loop()`, `app.py` | EOF breaks loop; Ctrl+C in loop continues; Ctrl+C at top level cleans Langfuse |
| F16 | Per-query Langfuse span | `_interactive_loop()` | `start_as_current_observation` with input/output; optional `FLUSH_PER_QUERY` |
| F17 | MLflow tools | `mlflow_tools/data_access.py` | 7 tools: list experiments, list runs, count runs, metrics, params, best runs, generalization |
| F18 | Tool error handling | `agent_middleware.handle_tool_errors` | Wraps tool failures as `ToolMessage` |
| F19 | Formatter retries | `response_formatter.py` | 2 attempts; fallback text block from agent draft |

---

## Feature → TUI mapping

| Feature | TUI component / pattern |
|---------|-------------------------|
| F1 Entry + shutdown | `FlokiApp.run()`; `on_unmount` / signal handler mirrors `app.py` Langfuse cleanup |
| F2–F6 Agent core | **No change** — import from `langchain_agent` after lazy init |
| F7 Two-phase `run_query` | `@work(thread=True)` worker runs full pipeline (agent → formatter); disable input while running |
| F8 Block rendering | `tui/blocks.py` renders formatter `BlockResponse`: `text` → `Markdown`, `table` → `Markdown` (phase 1) |
| F19 Formatter | **Unchanged** — `format_to_block_response()` with retries + draft fallback |
| F9 Welcome | `on_mount` → mount `WelcomePanel` (reuse banner styles from `console_ui`) |
| F10 Startup | `LoadingScreen` or footer `LoadingIndicator` during lazy agent init worker |
| F11 Langfuse status | `Footer` subtitle or status bar: tracing on/off + base URL |
| F12 Chat loop | `Input.Submitted` handler instead of `while True` |
| F13 Input | Textual `Input` widget (replaces prompt_toolkit) |
| F14 Exit | `/exit`, `/quit`, or bare `exit`/`quit` in input handler |
| F15 Interrupt | `Binding("ctrl+c", …)` — cancel worker or quit; replicate `app.py` cleanup |
| F16 Langfuse span | Same logic inside worker, before/after `run_query()` |
| F17 Tool visibility | **Enhancement within parity**: show "Running…" / "Formatting…" in status; optional `RichLog` tool trace (phase 1b) |
| F18 Tool errors | Unchanged — `handle_tool_errors` middleware; formatter surfaces errors in text blocks |

---

## Target architecture

```
src/
  app.py                    # argparse: --tui (default) | --repl
  agent/
    langchain_agent.py      # extract init + run_query; thin main() delegates to UI
    console_ui.py           # kept for --repl fallback
    tui/
      __init__.py
      app.py                # FlokiApp(Textual App)
      screens.py            # LoadingScreen (optional)
      widgets/
        chat.py             # ChatView, UserPrompt, AssistantResponse
        welcome.py          # WelcomePanel (banner)
        status.py           # StatusBar
      blocks.py             # Formatter BlockResponse → Textual widgets (shared with console_ui later)
      styles/
        floki.tcss          # Layout + Floki theme
```

### Layout (phase 1 — parity)

```
┌─ Header ─────────────────────────────────────────────────────┐
│ Floki v0.1                                                  │
├─ VerticalScroll (#chat-view) ──────────────────────────────┤
│  [WelcomePanel]                                             │
│  [UserPrompt]  You: compare runs …                          │
│  [AssistantResponse]  Floki + BlockResponse blocks          │
│  [LoadingIndicator]  (in-place while worker runs)           │
├─ Input (placeholder: Type your question. 'exit' to quit.) ─┤
├─ Footer ───────────────────────────────────────────────────┤
│  model | mlflow URI | Langfuse session id                   │
└────────────────────────────────────────────────────────────┘
```

### Data flow

```mermaid
sequenceDiagram
    participant User
    participant FlokiApp
    participant Worker
    participant Agent as LangGraph agent
    participant Formatter as format_to_block_response()
    participant LF as Langfuse

    User->>FlokiApp: Input.Submitted
    FlokiApp->>FlokiApp: mount UserPrompt + empty AssistantResponse
    FlokiApp->>Worker: @work(thread=True)
    Worker->>LF: start_as_current_observation (optional)
    Worker->>Agent: agent.invoke(messages)
    Agent-->>Worker: messages (tools + draft)
    Worker->>Formatter: format_to_block_response(formatter_llm, messages, query)
    Formatter-->>Worker: BlockResponse dict (text + table blocks)
    Worker->>LF: update output + flush (optional)
    Worker->>FlokiApp: call_from_thread(render_result, blocks)
    FlokiApp->>User: Markdown widgets per block; scroll to end
```

**Note:** `run_query()` wraps agent + formatter; the worker calls `run_query()` as a single unit so the formatter is never skipped.

---

## Implementation phases

### Phase 0 — Branch setup & dependencies

**Tasks**

1. Add `textual>=0.80.0` to `environment.yml` and `requirements.txt`.
2. Add `prompt_toolkit` explicitly to deps (already used, not declared).
3. Add `--tui` / `--repl` flag to `src/app.py` (default: `--tui` once stable; start with `--tui` opt-in).

**Acceptance:** `pip install textual` succeeds; existing REPL still runs via `--repl`.

---

### Phase 1 — Minimal Textual shell (parity MVP)

**Tasks**

1. **`agent/tui/app.py` — `FlokiApp`**
   - `compose()`: `Header`, `VerticalScroll#chat-view`, `Input`, `Footer`.
   - `CSS_PATH = "styles/floki.tcss"`.
   - `AUTO_FOCUS = "Input"`.

2. **`agent/tui/widgets/welcome.py`**
   - Port `_BANNER_LINES` / colors from `console_ui.py` into a `Static` or `Markdown` widget.
   - Same instructional text as `print_welcome()`.

3. **`agent/tui/widgets/chat.py`**
   - `UserPrompt(Static)` — styled user bubble.
   - `AssistantResponse(Vertical)` — container for rendered blocks.
   - `FlokiLabel(Static)` — "Floki" badge matching `[bold white on dark_green]`.

4. **`agent/tui/blocks.py`** — formatter output renderer
   - Input is always `result["structured_response"]` from `run_query()` (formatter-produced `BlockResponse`).
   - `render_blocks(parent, block_resp)`:
     - `type=="text"` → mount `Markdown(block["markdown"])`.
     - `type=="table"` → mount `Markdown(block["markdown"])` (formatter emits markdown pipe tables).
   - `render_result(parent, result)`:
     - Primary path: `_extract_structured(result)` → `render_blocks`.
     - Fallback (no structured_response): single `Markdown` from last AI message (same as `console_ui._fallback_content`).
   - Do **not** parse `ToolMessage` JSON for tables; formatter owns layout.

5. **Lazy agent initialization**
   - Move heavy module-level init in `langchain_agent.py` behind `get_agent_runtime() -> AgentRuntime` dataclass (config, llm, agent, fuse_client, session_thread_id, etc.).
   - `FlokiApp.on_mount` → `@work(thread=True) init_runtime()` → enable input when ready.
   - Fixes: `data_access.py` import-time `print()` must be suppressed or redirected when TUI active (use `logging` or env `FLOKI_QUIET_STARTUP=1`).

6. **Query worker**
   - `@on(Input.Submitted)`:
     - Ignore empty; handle `exit`/`quit`.
     - Clear input; mount `UserPrompt` + `AssistantResponse`; show `LoadingIndicator`.
     - `run_query_worker(query, response_widget)`.
   - `@work(thread=True) run_query_worker`:
     - Copy Langfuse span logic from `_interactive_loop`.
     - Call `run_query(query)` — includes agent invoke **and** formatter LLM call.
     - Optional status updates: "Calling tools…" then "Formatting response…" (footer only; formatter is not skippable).
     - `call_from_thread(render_result, response_widget, result)`.
     - Remove loading indicator; `chat_view.scroll_end()`.

7. **Shutdown**
   - `FlokiApp.on_unmount` + `action_quit`: flush/finish Langfuse (copy from `app.py`).
   - `exit`/`quit` input → `self.exit()`.

8. **Entry point**
   - `langchain_agent.main_tui()` → `FlokiApp().run()`.
   - `app.py`: `if args.tui: main_tui() else: main()`.

**Acceptance (manual)**

- [ ] Welcome banner visible on launch.
- [ ] Startup spinner/status while agent loads.
- [ ] Langfuse message visible in footer when enabled.
- [ ] Ask MLflow question → formatter produces `text` + `table` blocks; both render in TUI.
- [ ] `exit` quits cleanly with Langfuse flush.
- [ ] All 7 tools reachable via natural-language prompts (use manual checklist sample prompts).
- [ ] Formatter retry (2 attempts) works; `_fallback_response` renders agent draft when formatter fails.
- [ ] Session memory: multi-turn follow-up respects trimmed history.
- [ ] Footer shows both `gemini_model` and `formatter_model` from config.

---

### Phase 1b — Status & tool visibility (parity+)

**Tasks**

1. Footer shows: agent model, **formatter model**, `conversation_id`, MLflow tracking URI.
2. While worker runs: footer phases — "Agent running…" → "Formatting…" / `LoadingIndicator`.
3. Optional `RichLog` (collapsed by default) appending tool names from `result["messages"]` (display only; formatter still produces final blocks).

**Acceptance:** User sees activity during 10–60s queries; no frozen screen.

---

### Phase 2 — Enhanced rendering (optional, post-parity)

Formatter remains the source of truth for block content. Phase 2 only improves how formatter output is displayed.

| Enhancement | Widget | Notes |
|-------------|--------|-------|
| Table blocks | `DataTable` | Parse formatter `TableBlock.markdown` pipe tables for sortable UI (formatter still generates markdown) |
| Experiment sidebar | `ListView` | Browse experiments; click inserts context into input |
| Keyboard shortcuts | `BINDINGS` | `ctrl+l` clear chat, `ctrl+c` cancel worker |
| Streaming | `Markdown.update` | Stream formatter output if formatter LLM supports streaming |
| `textual serve` | CLI docs | Browser demo |

---

## Refactor details: `langchain_agent.py`

Extract without changing behavior:

```python
@dataclass
class AgentRuntime:
    config: dict
    llm: Any
    formatter_llm: Any
    agent: Any
    langfuse_handler: Any
    fuse_client: Any
    conversation_id: str | None
    lf_run: Any
    langfuse_user: str | None
    flush_per_query: bool
    session_thread_id: str

_runtime: AgentRuntime | None = None

def get_runtime() -> AgentRuntime:
    global _runtime
    if _runtime is None:
        _runtime = _build_runtime()
    return _runtime

def run_query(user_query: str) -> dict:
    rt = get_runtime()
    agent_result = rt.agent.invoke(...)
    structured = format_to_block_response(rt.formatter_llm, agent_result["messages"], user_query)
    # persist memory, return structured_response + messages
```

`main()` (REPL) and `main_tui()` both call `get_runtime()` before first query. **Do not add a code path that skips `format_to_block_response()`.**

---

## Styling (`floki.tcss`)

Reuse console_ui color language:

- User prompts: blue tint (`$primary 10%`), right margin.
- Assistant responses: green border (`$success`), left margin.
- Floki badge: white on dark green (match Rich style).
- Banner gradient: `#dd22ff` → `#2266ff` via markup on `Static`.

Reference: Textual `examples/mother.py` chat CSS.

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| MLflow `print()` corrupts TUI | Lazy import; redirect stdout during init; log to file |
| Long blocking `run_query` | Thread worker; disable input; show spinner |
| LangChain logs to stdout | Set log levels (already WARNING); avoid `print` in hot path |
| Ctrl+C during worker | `Worker.cancel()`; show "Cancelled" message |
| WSL terminal compatibility | Test Windows Terminal; document `$TERM` requirements |
| Duplicate UI code | `blocks.py` shared extract later; both UIs consume same `BlockResponse` shape from formatter |
| Formatter latency | Footer shows "Formatting…"; worker keeps UI responsive; no second render path to skip it |

---

## Testing plan

1. **REPL regression** — `python src/app.py --repl` unchanged behavior.
2. **TUI smoke** — launch, welcome, one query, exit.
3. **Manual checklist** — run subset of `docs/superpowers/specs/2026-05-21-mlflow-agent-manual-test-checklist-design.md` via TUI; same pass/fail criteria.
4. **Langfuse** — verify traces appear with same `conversation_id` pattern (`cli-{uuid}`).
5. **Edge cases** — empty input, EOF (Ctrl+D), formatter failure (retry + draft fallback), tool error, missing API key.
6. **Formatter contract** — responses with comparisons include at least one `table` block; summaries include `text` blocks (manual checklist grounding checks).

---

## File change summary

| File | Action |
|------|--------|
| `environment.yml`, `requirements.txt` | Add `textual`, `prompt_toolkit` |
| `src/app.py` | CLI flags, dispatch |
| `src/agent/langchain_agent.py` | Extract `AgentRuntime`, `main_tui()` |
| `src/agent/console_ui.py` | Keep for `--repl`; same `BlockResponse` render contract as TUI |
| `src/agent/response_formatter.py` | **No changes** — formatter stays in pipeline |
| `src/agent/tui/*` | **New** — App, widgets, styles |
| `src/mlflow_tools/data_access.py` | Replace `print` with `logging` (startup quiet) |
| `run_agent.sh` | Pass `--tui` when default switched |
| `README.md` | Document TUI usage, terminal requirements |

---

## Milestone checklist

- [ ] Phase 0: deps + branch
- [ ] Phase 1: TUI MVP with full feature parity (F1–F19)
- [ ] Phase 1b: loading status + tool trace
- [ ] Phase 2: DataTable, sidebar (stretch)
- [ ] README + manual test sign-off

---

## References

- Textual chat example: https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/
- `examples/mother.py` in Textual repo
- Workers guide: https://textual.textualize.io/guide/workers/
- Current UI: `src/agent/console_ui.py`, loop: `src/agent/langchain_agent.py`
