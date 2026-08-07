# Textual TUI rework design

Date: 2026-08-07
Status: Approved design
Scope: Rework the Floki terminal interface into a responsive coding-agent style
interface.

## Decision

Keep Textual as the TUI framework. It is already a project dependency and the
current TUI already uses its widgets, CSS, workers, Markdown support, and
headless test support. It can provide fixed areas and a scrolling transcript
without a new framework.

The new layout uses option 1, the fixed command deck.

- A compact welcome panel stays fixed at the top.
- Only the transcript scrolls.
- A composer, status line, and key help stay fixed at the bottom.
- The visual style is dark, calm, compact, and similar to a modern coding
  agent terminal such as Claude Code or Codex.

The TUI is the normal interactive interface. `--repl` stays available as the
simple Rich and prompt-toolkit fallback. `--tui` can remain accepted as a
compatible explicit alias.

## Goals

1. Give Floki a clear, fast interface for multi-turn MLflow questions.
2. Keep the Floki banner and welcome text visible while the user reads old
   messages.
3. Keep the screen responsive during agent, tool, and formatter work.
4. Preserve the current agent behavior, tracing, memory, formatter, and REPL
   fallback.
5. Work well in both normal and narrow terminals.

## Existing behavior that must stay unchanged

The UI is a presentation layer. It must not create a second agent path.

| Existing behavior | Required result |
| --- | --- |
| `run_query_traced()` | Each submitted question calls this function once. It remains the only TUI entry to the query pipeline. |
| Agent and MLflow tools | The current LangGraph agent and all current MLflow tools keep their behavior. |
| Formatter | `run_query()` still calls `format_to_block_response()`. The UI renders its `structured_response` blocks. |
| Session memory | The existing `InMemorySaver`, session thread ID, and `trim_messages_for_memory()` stay in use. A follow-up question can use prior conversation context. |
| Langfuse | Per-query observations, input and output fields, session and user attributes, optional per-query flush, finish, and final flush stay in the same lifecycle. |
| Errors from tools or formatter | The current agent middleware and formatter fallback keep their behavior. The UI presents the returned response or an error message. |
| Legacy terminal mode | `python src/app.py --repl` continues to run the Rich and prompt-toolkit REPL. |

## Architecture

The rework keeps one boundary between the interface and the agent runtime.

```text
Input.Submitted
  -> FlokiApp adds user row and activity row
  -> one Textual thread worker
  -> run_query_traced(query, on_status)
  -> Langfuse observation -> agent -> MLflow tools -> formatter -> memory trim
  -> worker posts a result to the Textual event loop
  -> FlokiApp renders formatter blocks in the transcript
```

`FlokiApp` owns screen state only: readiness, one active query, the current
worker, and the visible transcript. `langchain_agent.py` owns runtime setup,
agent execution, memory, and Langfuse work. Rendering helpers own only the
conversion from a returned result into Textual widgets.

The application must have one active query at a time. This protects the shared
agent session, its checkpointed memory, and the process-wide tool-status
callback. The composer is disabled while that query is active.

## Screen structure

`FlokiApp.compose()` mounts these direct screen regions in this order:

```text
Screen
├── WelcomePanel                 fixed top identity area
├── VerticalScroll#chat-view     the only scrolling region
└── CommandDeck                  fixed bottom area
    ├── Input#query-input        composer
    ├── Static#status-bar        short state and model/tracing state
    └── Static#key-help          key hints
```

`WelcomePanel` must not be a child of `#chat-view`. This is the main layout
rule. It makes the banner and welcome text persistent after transcript
scrolling. The command deck must not be a child of `#chat-view` either.

### Welcome panel

The panel is compact, not a large splash screen. It contains:

- a Floki banner wordmark and version;
- the welcome text: "Ask about your MLflow experiments.";
- one short instruction: "Enter sends. Ctrl+C stops a request or exits.";
- a small ready or loading marker when useful.

It uses a dark surface, a thin blue accent, and limited vertical space. It is
always visible. Its content does not change when messages are added.

### Transcript

The transcript contains chronological rows only:

- `UserPrompt` for a submitted question;
- `QueryLoading` while work is active;
- `AssistantResponse` for the final result;
- `SystemNotice` for a cancelled request or an inline error.

The app adds the user row before it starts work. It then adds one activity row
and an empty assistant response container. It removes the activity row when
the result or error is ready. The transcript scrolls to its end after a user
submission and after final rendering. The user may scroll upward without the
welcome panel or command deck moving.

`AssistantResponse` first shows a small Floki label. `blocks.render_result()`
then renders formatter text and table blocks as Textual `Markdown` widgets.
If no structured response exists, it renders the last agent message. If there
is no usable result, it shows a clear no-result message. The UI does not parse
raw tool JSON into a new table format. The formatter remains the response
format owner.

### Command deck

The command deck is a single visual unit at the bottom. It contains the input
above a one-line status and one-line key help area.

The composer starts disabled while `get_runtime()` runs. Its placeholder tells
the user that Enter sends a question and `exit` or `quit` exits. When ready,
it receives focus. Empty input does nothing. A submitted non-empty input is
trimmed, shown in the transcript, and then cleared.

The status line uses short state text:

- `Initializing agent…`
- `Ready`
- `Agent running…`
- a tool verb such as `Scanning…`
- `Formatting…`
- `Stopping current request…`
- a concise initialization or query error

When idle, the line also shows useful, short runtime facts: agent model,
formatter model, and whether Langfuse is on. It must not expose secrets.

The key-help line shows only supported controls: `Enter send`, `Ctrl+C
cancel/exit`, and `Ctrl+Q exit`. It is informational, not a second command
system.

## Query and status flow

1. `Input.Submitted` rejects input until runtime initialization finishes and
   rejects new input while a query is active.
2. For a valid question, the app adds `UserPrompt`, `QueryLoading`, and an
   empty `AssistantResponse` to `#chat-view`.
3. A Textual thread worker calls `run_query_traced(query, on_status=...)`.
   The event loop stays free to redraw, scroll, and handle keys.
4. `on_status` sends the current verb back with `call_from_thread()`. It
   updates the activity row and the status line only on the Textual event
   loop. Known tools use the current verb map. Unknown work says `Thinking…`.
5. On success, the event loop removes the activity row, renders the result in
   the prepared response container, sets the status to `Ready`, and enables
   and focuses the composer.
6. On failure, the event loop removes the activity row, adds a short escaped
   error message to that same response container, sets the status to `Ready`,
   and enables and focuses the composer.

The worker must never update Textual widgets directly. It sends all UI work to
the event loop with `call_from_thread()`.

## Cancel, exit, and error behavior

`exit`, `quit`, `/exit`, and `/quit` exit the application when the composer is
ready. `Ctrl+Q` exits when no query is active.

`Ctrl+C` has two clear cases:

1. If no query is active, it exits the application.
2. If a query is active, it requests cancellation of that worker and changes
   the status to `Stopping current request…`.

The current agent path uses synchronous network and tool calls. A thread worker
cannot safely force-stop a call that is already inside that code. Therefore,
cancel is best effort: the app must ignore a cancelled query result and keep
the composer disabled until the worker actually finishes. This prevents stale
content, concurrent queries, and memory corruption. It then removes the
activity row, records a `Request cancelled.` system row, restores `Ready`, and
enables the composer.

If the user presses `Ctrl+Q` while a query is active, the app requests the
same cancellation and sets an exit-pending flag. It exits only after that
worker ends and Langfuse work can finish cleanly. The app does not offer a
force-close key because it cannot stop the active synchronous call safely.

Unexpected initialization errors leave the composer disabled and show a short
error with the available exit keys. Unexpected query errors leave old
transcript content visible and return the composer to the ready state.

Application shutdown calls `finish_langfuse()` and `flush_langfuse_on_exit()`
once through the Textual shutdown path. `src/app.py` must not add a second
final flush after a normal TUI exit. Its outer interrupt guard continues to
cover startup and non-TUI paths. Cleanup functions must remain safe when no
runtime or Langfuse client exists.

## Responsive behavior

The interface must be useful from 60 columns upward. It has two defined modes.

| Area | Normal: 80 columns or more | Narrow: 60 to 79 columns |
| --- | --- | --- |
| Welcome panel | One compact row with wordmark, version, welcome text, and instruction. | Two short rows. The wordmark remains visible. The instruction can wrap once. |
| Transcript rows | User rows align toward the right and assistant rows toward the left, with modest side margins. | Both row types use nearly full width with one-column margins. No text is clipped. |
| Command deck | Composer, full status facts, and key help fit on separate fixed lines. | Composer stays fixed. Status prefers activity and tracing state. Model names may be shortened. Key help uses short labels. |
| Markdown tables | Render tables in a container with horizontal scrolling. | Preserve table content. Use horizontal scrolling rather than squeeze columns into unreadable text. |

At less than 60 columns, the app remains usable but may wrap status and welcome
text. It must keep the welcome panel, transcript, composer, and exit path.
The rework does not promise a polished phone-sized terminal layout.

CSS media rules, not Python width checks, should control these layout changes.
The theme uses a One Dark based palette already used by Floki: charcoal
backgrounds, blue for input and user actions, green for assistant success,
yellow for active work, and red for errors. Color is an aid only. Labels and
text must still identify each state.

## Visual direction

The screen should feel like an expert command surface, not a dashboard.

- Prefer dark surfaces, thin borders, restrained color, and readable contrast.
- Make the transcript the largest visual area.
- Use compact spacing. Do not use large cards, repeated decorations, or a
  permanent raw tool log.
- Distinguish user and assistant rows by alignment, border accent, and label.
- Keep motion small. The activity pulse may animate, but no other animation is
  required.
- Use clear sentence case text. Do not use noisy status details or token
  counters.

The interface may show high-level tool activity through a verb such as
`Scanning…`. It must not display raw tool inputs, raw JSON, credentials, or
internal trace data by default.

## Files and responsibilities

| File | Change responsibility |
| --- | --- |
| `src/app.py` | Make the Textual interface the default interactive mode. Keep `--repl` and the `--tui` compatibility flag. Keep outer interrupt cleanup. |
| `src/agent/tui/app.py` | Compose the fixed regions, manage initialization, query work, cancellation state, status updates, focus, and shutdown. |
| `src/agent/tui/styles/floki.tcss` | Define the fixed command-deck layout, normal and narrow media rules, colors, spacing, and visible focus state. |
| `src/agent/tui/widgets/welcome.py` | Render the compact persistent banner and required welcome text. |
| `src/agent/tui/widgets/chat.py` | Define message rows, the activity row, assistant label, and a `SystemNotice` row for cancelled and error messages. |
| `src/agent/tui/blocks.py` | Keep formatter block and fallback rendering isolated from app state. |
| `src/agent/langchain_agent.py` | Keep `run_query_traced()`, Langfuse work, memory trimming, and the REPL path intact. Make only small integration changes needed by the new UI contract. |
| `tests/test_tui_app.py` | Test composition, fixed areas, input state, query state, errors, cancellation, and shutdown calls. |
| `tests/test_tui_blocks.py` | Test structured and fallback response rendering helpers. |
| `tests/test_tool_status.py` | Keep tool-status verb coverage. |

## Testing and validation

Automated tests use Textual `App.run_test()` and mocks for agent runtime and
query execution. They must not require Gemini credentials, Langfuse access, or
an MLflow server.

Required automated coverage:

1. Runtime success enables and focuses the composer. Runtime failure keeps it
   disabled and reports an error.
2. `WelcomePanel` is mounted outside `#chat-view`, and it remains present after
   enough transcript rows are added to scroll.
3. The command deck is outside `#chat-view`, and the composer remains mounted
   while the transcript scrolls.
4. A submitted question adds a user row and activity row, disables the
   composer, calls `run_query_traced()` once, then renders structured blocks
   and restores the composer.
5. Status callbacks update the activity row through the Textual event loop.
6. A query exception creates an escaped error row and restores the composer.
7. Cancel marks the current request as cancelled, ignores its late result, and
   restores the composer only after the worker ends.
8. Exit commands and app unmount call Langfuse finish and flush safely once.
   A busy `Ctrl+Q` request waits for the cancelled worker before it exits.
9. Existing block fallback, tool-status mapping, response formatter, memory,
   and data-access tests still pass.

Manual validation uses the real TUI in a normal terminal and a narrow terminal.
Capture a terminal recording or screenshots for both sizes. Check that the
fixed top panel and fixed command deck remain visible while the transcript
scrolls. Run at least one real MLflow question and one multi-turn follow-up.
Confirm that the formatter response, a table response, tool activity, a query
error, cancellation, `exit`, and `--repl` all behave as specified.

## In scope

- A complete Textual layout and style rework to the fixed command deck design.
- Default TUI startup with the legacy `--repl` fallback.
- Persistent compact banner and welcome text.
- Responsive normal and narrow layouts.
- Clear loading, tool-status, error, cancel, exit, focus, and input behavior.
- Headless tests and manual visual evidence for the new layout.
- Small code changes that are required to keep the current query, memory, and
  Langfuse contracts correct.

## Out of scope

- Replacing Textual, LangGraph, the agent model, the formatter model, or
  MLflow tools.
- Changing the meaning of agent prompts, memory trimming, or Langfuse data.
- Token streaming, partial formatter streaming, or a forced network-call stop.
- A new experiment sidebar, file picker, command palette, model switcher, or
  raw tool inspector.
- A browser or web deployment of the TUI.
- Rich table parsing into a new `DataTable` representation.

## Acceptance criteria

The rework is ready when all of these statements are true:

1. Starting `python src/app.py` opens the Textual interface, and
   `python src/app.py --repl` still opens the legacy REPL.
2. The welcome banner and welcome text are fixed outside the transcript and
   stay visible while transcript content scrolls.
3. The composer, status line, and key help are fixed outside the transcript.
4. A real query uses `run_query_traced()`, renders the formatter response, and
   keeps session memory and Langfuse behavior intact.
5. The interface stays responsive while the agent, tools, and formatter run.
6. The app shows high-level activity, a useful error, and clear cancellation
   behavior without showing raw tool data or secrets.
7. Normal and narrow terminal checks show no clipped critical controls or
   hidden persistent regions.
8. The specified automated tests pass, existing relevant tests pass, and
   manual normal and narrow terminal checks provide visual evidence.
