# Textual TUI rework implementation plan

Date: 2026-08-07
Status: Ready for implementation
Scope: Replace the current basic Textual screen with the approved fixed command
deck design. Keep the agent runtime and the Rich REPL behavior.

## Purpose

This plan implements
[`2026-08-07-textual-tui-rework-design.md`](2026-08-07-textual-tui-rework-design.md).
It keeps Textual. It does not add a second agent path, a new formatter path, or
a new TUI framework.

The current code already has the required foundation:

- `run_query_traced()` is the TUI query entry point.
- A thread worker keeps the event loop free during agent and formatter work.
- `blocks.py` renders formatter Markdown blocks.
- The agent runtime has session memory, tool status callbacks, and Langfuse
  cleanup functions.

The main defects are in the screen structure and state handling. The welcome
panel is inside `#chat-view`, the input is docked as separate widgets instead
of one command deck, the default entry point is the REPL, and cancellation
exits the screen before the synchronous query has ended.

## Requirements and proof

| Requirement | Implementation proof | Verification proof |
| --- | --- | --- |
| Textual is the selected framework | No framework or dependency change. Use the existing `FlokiApp`. | `environment.yml` still contains Textual. TUI tests run. |
| Welcome banner and required welcome text stay visible | `WelcomePanel` is a direct child of `Screen`, outside `#chat-view`. | Headless tree test and manual scroll test. |
| The TUI has a coding-agent style command surface | Fixed welcome panel, transcript, and command deck. Compact dark CSS. | Visual check at 100 and 60 columns. |
| The screen stays responsive | Runtime setup and one query run in Textual thread workers. Workers only schedule UI work on the event loop. | Headless tests and a manual slow-query check. |
| The agent path stays unchanged | The query worker calls `run_query_traced(query, on_status=...)` once. | Mock call-count test and runtime contract review. |
| Formatter, memory, tracing, and REPL stay available | Do not change `run_query()`, `format_to_block_response()`, memory code, or the REPL loop. | Full tests, manual REPL smoke test, and diff review. |
| Normal and narrow terminals work | `FlokiApp` sets responsive `Screen` classes on resize. Valid TCSS class selectors control layout and compact text. Tables scroll horizontally. | Headless size tests and manual size checks. |
| Shutdown is safe | A single idempotent TUI shutdown path calls Langfuse finish and flush. `src/app.py` does not flush again after a normal TUI exit. | Shutdown call-count test and manual exit check. |

## Visual implementation brief

### Design purpose

Floki is an MLflow research assistant for people who need evidence from
experiments, runs, and metrics. The screen has one job: let the user ask a
question, see the work state, and read a grounded answer without visual noise.
It must look like a focused command surface, not a dashboard or a generic chat
app.

### Compact color system

Use these six colors from the current One Dark theme. Do not add gradients,
rainbow accents, or a second theme. Use transparency of Night for small
surface differences instead of new colors.

| Name | Hex | Use |
| --- | --- | --- |
| Night | `#282c34` | Main screen and transcript background. |
| Fog | `#abb2bf` | Body text, quiet rules, and inactive utility text. |
| Signal blue | `#61afef` | User edge, input focus, and the active cursor area. |
| Evidence green | `#98c379` | Floki label and the assistant answer edge. |
| Work amber | `#e5c07b` | Loading pulse and active work text. |
| Fault red | `#e06c75` | Error and cancellation notice edge and text. |

Do not use color as the only signal. Every colored state must also have a
label or plain text that states its meaning. Use Fog at reduced opacity for
utility text and thin rules. Do not put bright colors behind long Markdown
text.

### Terminal type roles

The terminal controls the font. Do not depend on a specific installed font,
icon, emoji, box character, special character combination, or measured
character width.

| Role | Treatment | Use |
| --- | --- | --- |
| Identity | Bold uppercase plain text, such as `FLOKI`, with one version value. | Fixed welcome panel only. |
| Reading text | Normal terminal font, sentence case, Fog foreground. | Prompts, Markdown answers, errors, and notices. |
| Utility text | Normal terminal font with reduced contrast. Short labels use uppercase only where it improves scanning. | Status, key help, version, and provenance rail. |
| Code and tables | Text blocks use Textual `Markdown`. Table blocks use Rich `Markdown` in Textual `Static`, inside horizontal scrolling. | Code spans, answer tables, and values that need alignment. |

Do not fake a display font with ASCII art. Do not use all caps for sentences.
Keep the exact required welcome and instruction text in sentence case.

### Signature: MLflow evidence rail

Place one small, static rail in `WelcomePanel` beside or below the wordmark:

```text
DATA -> RUNS -> EVIDENCE
```

This rail is Floki's signature. It shows that Floki moves from MLflow data
through recorded runs to an evidence-based answer. It is not a progress
bar, a live trace, or a count of query stages. It must not change while a
query runs. Render it as plain text, with the arrows and labels readable in a
basic terminal. At 60 columns it can use `DATA>RUNS>EVIDENCE`; below 60
columns it may wrap after `RUNS`, but it must not hide the wordmark or the
required welcome text.

The rail costs one `Static` render and no timer, network call, measurement, or
animation. It therefore does not affect query speed or compete with the
existing small loading pulse.

### Layout, spacing, and borders

- Do not draw an outer frame around the screen. The terminal edge is the frame.
- Give `#floki-welcome` a thin Signal blue bottom rule. Use one cell of top
  and bottom padding and two cells of side padding at normal widths.
- Keep the welcome content to two visual lines at normal widths. The wordmark
  and evidence rail are the first line. The required welcome text and
  instruction are the second line, and may wrap once in a narrow terminal.
- Give `#chat-view` the available height and one cell of side inset. It is the
  only scrolling region. Keep one blank row between transcript items.
- Use short edge rules, not full cards. A user row has a Signal blue right
  edge and aligns right. An assistant response has an Evidence green left edge
  and aligns left. `SystemNotice` has a Fault red left edge. Do not put a
  border around each Markdown child.
- Use no more than one cell of internal padding for transcript rows. The
  command deck has one top rule, one cell of side inset, and the input above
  the status and key-help lines.
- The focused input has a full Signal blue border. The unfocused input has a
  quiet Fog rule. This focus change must remain visible without a mouse.
- Use Work amber only for `QueryLoading` and active status. Keep its current
  small pulse. Do not add other moving elements.

### Responsive wireframes

Normal view, 80 columns or more. The transcript keeps most of the height.

```text
 FLOKI v0.1   DATA -> RUNS -> EVIDENCE
 Ask about your MLflow experiments.  Enter sends. Ctrl+C stops a request or exits.
 ------------------------------------------------------------------------------
 Floki  Here is the comparison from the selected experiment.
        [Markdown text and tables scroll only inside the transcript]

                                      You  Compare the best two runs by F1.
 Work: Scanning...
 ------------------------------------------------------------------------------
 > Ask about experiments, runs, or metrics
 Ready | agent model | formatter model | Langfuse off
 Enter send  |  Ctrl+C cancel/exit  |  Ctrl+Q exit
```

Narrow view, 60 to 79 columns. The message rows use nearly the full width.
The status shows current work before model facts.

```text
 FLOKI v0.1  DATA>RUNS>EVIDENCE
 Ask about your MLflow experiments.
 Enter sends. Ctrl+C stops a request or exits.
 ----------------------------------------------------------
 Floki  Here is the comparison from the selected experiment.
        Wide tables scroll sideways.

 You  Compare the best two runs by F1.
 Work: Scanning...
 ----------------------------------------------------------
 > Ask about experiments, runs, or metrics
 Ready
 Enter send | Ctrl+C cancel/exit | Ctrl+Q exit
```

At widths below 60 columns, preserve the same order. Welcome and utility text
may wrap, but do not clip the composer, transcript, wordmark, welcome text,
or exit help.

### Self-critique and revision

The first concept used a dark One Dark screen, blue user panels, green
assistant panels, and several colored status parts. It was clear, but it
looked like the default visual language for many coding-agent tools. Full
colored panels also made long research answers look heavy.

The revised concept removes the colored cards and extra status decoration. It
uses color only on thin message edges, focus, and active work. The static
`DATA -> RUNS -> EVIDENCE` evidence rail replaces generic decoration with a
small statement of Floki's ML research purpose. It makes Floki different while
it keeps reading, speed, and narrow-terminal use as the priority.

## Shared interface contract

The implementation agents must use this contract. It avoids behavior drift
between their changes.

### Widget tree

`FlokiApp.compose()` must create these direct screen children in this order:

```text
Screen
├── WelcomePanel#floki-welcome
├── VerticalScroll#chat-view
└── Vertical#command-deck
    ├── Input#query-input
    ├── Static#status-bar
    └── Static#key-help
```

Only `#chat-view` scrolls. `WelcomePanel` and `#command-deck` must not be
children of it. Do not use `dock` for the command deck or its children. The
normal vertical screen layout must keep the first and last regions fixed while
`#chat-view` uses the available height.

### Responsive class contract

Installed Textual 8.2.7 rejects TCSS `@media` syntax. `FlokiApp` must handle
Textual `Resize` events and set responsive classes on `Screen`. Set `narrow`
for widths from 60 through 79 columns. Set `compact` for widths below 60
columns. Remove both classes at widths of 80 columns or more. The two classes
must not be active at the same time. TCSS must use valid `Screen.narrow` and
`Screen.compact` selectors to apply the responsive layout.

### App state and request identity

`FlokiApp` owns only UI state. Add clear private fields in
`src/agent/tui/app.py` for:

- runtime readiness;
- the active `Worker`, or `None`;
- the active request identifier;
- cancellation requested for that request;
- exit requested after the active request ends; and
- whether Langfuse shutdown has already run.

Start request identifiers at one and increase them for each submitted query.
Pass the identifier, not a widget update instruction, from the worker to the
event loop. Every status, success, failure, and cancellation handler must
ignore an identifier that is no longer active. This protects the screen from a
late callback after cancellation.

The input is disabled during runtime setup and during every active request. Do
not start another query until the active worker has ended, even after
`Worker.cancel()` was called. Textual cannot safely stop the synchronous agent
or network call at once.

### Query flow

1. Reject input while runtime setup is not ready or a request is active.
2. Trim non-empty input. Treat `exit`, `quit`, `/exit`, and `/quit` as an exit
   request only when no query is active.
3. Add `UserPrompt`, `QueryLoading`, and an empty `AssistantResponse` to
   `#chat-view`. Clear and disable the input. Scroll the transcript to its end.
4. Start one exclusive Textual thread worker. It calls
   `run_query_traced(query, on_status=on_status)` exactly once.
5. `on_status` calls `self.call_from_thread(...)`. The scheduled handler
   updates only the matching loading row and status text on the event loop.
   It must not update a widget directly from the worker thread.
6. On success, remove the loading row and call `render_result()` on the
   prepared response. Scroll to the transcript end. Set `Ready`, enable the
   input, and focus it.
7. On a normal query error, remove the loading row. Mount an escaped
   `SystemNotice` in the prepared assistant response. Keep earlier transcript
   rows. Then restore the ready input state.
8. On a cancelled request, ignore the result or error. Remove the loading row
   and the empty assistant response. Add `SystemNotice("Request cancelled.")`
   as the next transcript row. Restore the ready input state only after the
   worker ends. If exit is pending, exit at this point instead.

`SystemNotice` inside an error response satisfies both design rules: the error
uses the prepared response container and has its own visible notice row. A
cancelled request uses a top-level notice because it has no assistant answer.

### Status and keyboard contract

Use sentence case display text. The status line must use these values:

- `Initializing agent…`
- `Ready`
- `Agent running…`
- known tool verbs such as `Scanning…`
- `Thinking…` for unknown work
- `Formatting…`
- `Stopping current request…`
- a short `Init failed: ...` or `Query failed: ...` message

When idle, append short non-secret facts for the agent model, formatter model,
and Langfuse on or off state. The narrow layout may shorten model names and
must prefer current activity over extra facts.

Keep only these key help labels: `Enter send`, `Ctrl+C cancel/exit`, and
`Ctrl+Q exit`.

Bind `Ctrl+C` to an action that exits when idle or requests cancellation when
busy. Bind `Ctrl+Q` to an action that exits when idle or requests cancellation
and records an exit-pending flag when busy. A second key press must not start a
second worker or force-close the process.

### Runtime and cleanup contract

Do not change the implementation of `run_query_traced()`, `run_query()`,
`format_to_block_response()`, `trim_messages_for_memory()`, or the Langfuse
span lifecycle unless an integration defect is proved. The current status
callback and exclusive worker make the process-global tool callback safe for
one active TUI query.

Put one idempotent cleanup helper in `FlokiApp`. It calls
`finish_langfuse()` and `flush_langfuse_on_exit()` once from the Textual
shutdown event. `src/app.py` keeps its outer `KeyboardInterrupt` cleanup for
startup and non-TUI paths, but it must not run an unconditional normal-exit
flush after `main_tui()` returns.

## Ordered implementation tasks

### Task 1: Record the baseline and define the test harness

**Owner:** integration agent
**Files:** `tests/test_tui_app.py`, `tests/test_tui_blocks.py`,
`tests/test_tool_status.py`
**Do not edit:** application files in this task.

1. Run the current focused TUI tests and the full suite before code changes.
2. Add test helpers that build a small fake runtime. The fake runtime must have
   `config`, `fuse_client`, and `conversation_id` values. It must not require
   a Gemini key, Langfuse keys, or a live MLflow server.
3. Add a small fake successful result with a text block and a table block. Add
   a fake failure result and a blocking query helper that uses
   `threading.Event` for cancellation tests.
4. Keep test helpers local to their test file. Do not add production test-only
   switches.

**Expected result:** later tasks can test app state without network access.

**Validation commands:**

```bash
PYTHONPATH=src python -m pytest tests/test_tui_app.py tests/test_tui_blocks.py tests/test_tool_status.py -q
PYTHONPATH=src python -m pytest -q
```

### Task 2: Build persistent welcome and transcript row widgets

**Owner:** presentation agent
**Files:** `src/agent/tui/widgets/welcome.py`,
`src/agent/tui/widgets/chat.py`, `tests/test_tui_app.py`
**Do not edit:** `src/agent/tui/app.py`, `src/agent/tui/blocks.py`, or TCSS.

1. Replace the large copied ASCII banner with a compact `WelcomePanel`. It
   must show a Floki wordmark, a version, this exact welcome text, and this
   short instruction:

   ```text
   Ask about your MLflow experiments.
   Enter sends. Ctrl+C stops a request or exits.
   ```

2. Do not import the old multi-line banner from `console_ui.py`. The compact
   panel must fit in a 60-column terminal. Its content must not change when
   messages are added.
3. Keep `UserPrompt`, `QueryLoading`, `AssistantResponse`, and `FlokiLabel`.
   Add `SystemNotice` for cancelled and error text. `SystemNotice` must escape
   error text before it sends the text to a markup-aware widget.
4. Keep the loading pulse small. Make `set_verb()` accept empty or unknown
   values and display the fallback thinking verb. The app, not the widget,
   controls sentence case and the trailing ellipsis in its status line.
5. Update `tests/test_tui_app.py` first. Test the required welcome text,
   panel identity, safe escaped notice text, and a changed loading verb.

**Expected result:** the app has compact reusable rows that match the design
without changing agent behavior.

**Validation commands:**

```bash
PYTHONPATH=src python -m pytest tests/test_tui_app.py -q
python -m compileall -q src/agent/tui/widgets
```

### Task 3: Make response rendering complete and table-safe

**Owner:** block-rendering agent
**Files:** `src/agent/tui/blocks.py`, `tests/test_tui_blocks.py`
**Do not edit:** app state, widgets, TCSS, or runtime files.

1. Keep `extract_structured()` as the first rendering path. It accepts both
   `result["structured_response"]` and a direct block response.
2. Keep `fallback_content()` as the fallback path. It must use the last agent
   message when available and return a safe string for other result values.
3. Keep the Floki label as the first child of a non-cancelled assistant
   response. Render valid formatter text blocks with Textual `Markdown`.
   Render valid table blocks with Rich `Markdown` in a Textual `Static`. Do
   not parse raw tool JSON and do not create a second table data model.
4. Put each table `Static` in a Textual `HorizontalScroll` container. Give the
   container a stable class name, such as `table-scroll`, for the TCSS task.
   Text blocks do not need this wrapper. The table wrapper must preserve table
   data instead of shrinking it until text is unreadable. Use Rich Markdown
   for tables because its synchronous render makes the table height available
   before the transcript performs its final scroll. Textual Markdown mounts
   nested table widgets asynchronously, which can leave the transcript height
   stale at that point.
5. Keep a clear `No result.` Markdown message for empty results. Do not show
   Python `None` or raw dictionary output to the user when a usable fallback
   is available.
6. Write tests before the implementation change for structured preference,
   direct blocks, final-message fallback, empty results, malformed blocks, and
   text plus table widget mounting. Assert that a table scroll wrapper contains
   a Textual `Static` with a Rich `Markdown` renderable. Also test horizontal
   overflow without tall cell wrapping. Do not test Textual internals.

**Expected result:** formatter output is the single owner of answer structure,
and table data remains accessible in narrow terminals.

**Validation commands:**

```bash
PYTHONPATH=src python -m pytest tests/test_tui_blocks.py -q
python -m compileall -q src/agent/tui/blocks.py
```

### Task 4: Implement the fixed visual layout

**Owner:** presentation agent
**Files:** `src/agent/tui/styles/floki.tcss`
**Input contract:** Task 2 widget names and the Task 5 screen IDs.
**Do not edit:** Python files in this task.

1. Use the existing One Dark theme values. Use charcoal surfaces, blue user
   accents, green assistant accents, yellow activity, and red error text.
   Labels and text must identify every state without color.
2. Define the three-region vertical layout. `#chat-view` gets the flexible
   height and its own scrollbar. `#floki-welcome` and `#command-deck` keep
   their content height. Remove the current bottom docking rules.
3. Style the command deck as one low visual unit. The focused input must have
   a visible blue border. Keep the status and key-help lines one line in a
   normal terminal.
4. Style user rows on the right and assistant rows on the left with modest
   side margins. Use sensible relative widths. Do not use `width: 100%` with
   opposing large margins, because it can clip or overflow rows.
5. Style `SystemNotice`, `QueryLoading`, `FlokiLabel`, Markdown, and
   `.table-scroll`. The table-scroll rule must allow horizontal scrolling and
   must not hide content.
6. Handle Textual `Resize` events in `FlokiApp`. Set `Screen.narrow` for widths
   from 60 through 79 columns and set `Screen.compact` for widths below 60.
   Remove both classes at widths of 80 columns or more. This Python code must
   only select the responsive mode.
7. Use valid `Screen.narrow` and `Screen.compact` selectors in TCSS. In narrow
   mode, use near-full transcript width, two short welcome rows, short key
   help, and reduced model facts. In compact mode, keep the welcome panel,
   transcript, input, and exit path visible. Text can wrap below 60 columns.
   Do not use `@media` syntax because installed Textual 8.2.7 rejects it.

**Expected result:** the interface looks like a compact coding-agent command
surface. The transcript is the largest screen area.

**Validation commands:**

```bash
python -m compileall -q src
PYTHONPATH=src python -m pytest tests/test_tui_app.py tests/test_tui_blocks.py -q
```

### Task 5: Rebuild `FlokiApp` state and query flow

**Owner:** application-shell agent
**Files:** `src/agent/tui/app.py`, `tests/test_tui_app.py`
**Do not edit:** widget implementation, block rendering, TCSS, or runtime
behavior.

1. Change `compose()` to create the required direct screen tree from the
   shared interface contract. Use a `Vertical` command deck. Set the initial
   status to `Initializing agent…`, add the required key-help text, and keep
   the input disabled until runtime setup succeeds.
2. Keep runtime initialization in `_init_runtime()` as a thread worker. On
   success, save only the UI facts needed for the idle status, enable and focus
   the input, and set `Ready` with short model and Langfuse information. On
   failure, leave the input disabled and show a concise error plus exit keys.
3. Refactor query methods around one active request identifier. Keep the
   worker exclusive. It must call `run_query_traced()` once, pass its
   `on_status` callback, and schedule all state changes with
   `call_from_thread()`.
4. Make the status handler show `Formatting…` for `FORMAT_VERB`, a capitalized
   known tool verb such as `Scanning…`, and `Thinking…` for unknown work. It
   must update both the loading row and status line only for the active
   request.
5. On success, error, and cancellation, follow the shared query flow exactly.
   Use `SystemNotice` for an error or cancellation. Escape errors. Always
   remove the activity row before the terminal state. Always scroll after a
   new user message and final rendering.
6. Replace the current `action_quit()` with separate idle-or-cancel behavior
   for `Ctrl+C` and exit-pending behavior for `Ctrl+Q`. Do not call
   `self.exit()` while the active worker may still use the runtime.
7. Add one private idempotent cleanup helper. Call it from the Textual
   shutdown event. It calls `finish_langfuse()` then `flush_langfuse_on_exit()`
   once, including when runtime setup did not finish.
8. Preserve `run_tui()` as a thin `FlokiApp().run()` function.

**Tests to add before or with the implementation:**

- startup has direct fixed regions and the welcome panel is not a child of
  `#chat-view`;
- successful initialization enables and focuses input and does not expose a
  secret in status;
- initialization failure leaves input disabled and shows usable exit text;
- empty input changes nothing;
- exit words exit without calling `run_query_traced()`;
- one submission creates user, activity, and assistant rows, disables input,
  calls `run_query_traced()` once, renders the result, and returns to ready;
- a status callback updates the active activity text and uses `Formatting…`;
- a query exception escapes its text, keeps earlier rows, and restores input;
- a blocking cancelled request stays busy until its worker returns, discards a
  late result, shows `Request cancelled.`, then restores input;
- `Ctrl+Q` during the same blocking request waits for worker completion before
  exit;
- shutdown calls each Langfuse cleanup function once;
- `run_test(size=(100, 32))` and `run_test(size=(60, 24))` keep the welcome,
  chat view, command deck, and input mounted and visible.

Use `threading.Event` to control a mocked query worker in cancellation tests.
Do not use arbitrary long sleeps. Use short `pilot.pause()` calls only to let
the event loop process a known state change.

**Expected result:** the screen is responsive and safe during setup, normal
queries, errors, cancellation, and exit.

**Validation commands:**

```bash
PYTHONPATH=src python -m pytest tests/test_tui_app.py -q
PYTHONPATH=src python -m pytest tests/test_tui_app.py tests/test_tui_blocks.py tests/test_tool_status.py -q
python -m compileall -q src
```

### Task 6: Make the TUI the default without breaking the REPL

**Owner:** entry-point agent
**Files:** `src/app.py`, `tests/test_app_entry.py`
**Do not edit:** TUI or runtime files.

1. Make no flag start `main_tui()`. Keep `--tui` as a compatible explicit
   alias. Keep `--repl` as the only option that starts the Rich and
   prompt-toolkit loop.
2. Give the parser and dispatch code a small testable function. Import the
   legacy function with an unambiguous local name so a new entry function does
   not hide it.
3. Keep the outer `KeyboardInterrupt` cleanup. Remove the current normal-exit
   `else` flush, because `FlokiApp` owns normal TUI cleanup. Do not duplicate
   cleanup after a normal TUI exit.
4. Write parser and dispatch tests before the change. Verify no flag and
   `--tui` select TUI, `--repl` selects the REPL, mutually exclusive options
   fail, normal TUI return does not call the outer flush, and an outer startup
   interrupt still calls `finish_langfuse()`.

**Expected result:** `bash run_agent.sh`, `python src/app.py`, and
`python src/app.py --tui` start the new TUI. `python src/app.py --repl` keeps
the fallback interface.

**Validation commands:**

```bash
PYTHONPATH=src python -m pytest tests/test_app_entry.py -q
PYTHONPATH=src python -m pytest -q
python -m compileall -q src
python src/app.py --help
```

### Task 7: Check that the runtime contract has not changed

**Owner:** integration agent
**Files:** inspect `src/agent/langchain_agent.py`, `src/agent/tool_status.py`,
`src/agent/context_memory.py`, and `src/agent/response_formatter.py`. Edit
only if a failing focused test proves a small UI integration defect.

1. Confirm that `run_query_traced()` remains the only function called by the
   TUI worker.
2. Confirm that `run_query()` still calls `format_to_block_response()` after
   the agent invocation and before it returns a structured response.
3. Confirm that `_persist_trimmed_memory()` still uses the same session thread
   ID and `trim_messages_for_memory()`.
4. Confirm that `status_callback()` still covers agent and formatter work and
   that one active query is enforced by the app.
5. Confirm that Langfuse input, output, session, user, optional per-query
   flush, finish, and final flush behavior is unchanged except for the removed
   duplicate outer normal-exit flush.

**Expected result:** the rework is presentation-only outside its defined UI
integration points.

**Validation commands:**

```bash
PYTHONPATH=src python -m pytest tests/test_context_memory.py tests/test_response_formatter.py tests/test_tool_status.py -q
PYTHONPATH=src python -m pytest -q
```

### Task 8: Run manual visual and behavior checks

**Owner:** integration agent
**Files:** no product file changes. Store temporary notes or recordings only
outside the repository unless they have long-term value.

Use a configured local environment and the sample MLflow data. Run
`bash src/scripts/run_mlflow.sh` in one terminal when the local tracker is
needed. Run the agent in another terminal with `bash run_agent.sh`.

Check each item in a normal terminal at least 100 columns by 32 rows:

1. Startup shows the compact Floki wordmark and the exact welcome text. The
   input is disabled until the runtime is ready.
2. The normal idle status shows both model roles and Langfuse on or off with no
   API key, URL credential, raw trace field, or tool JSON.
3. Submit one simple question from the manual MLflow checklist. The user row,
   pulse, tool verb, formatting state, answer label, Markdown text, and a
   formatter table appear in order.
4. Submit a follow-up question. Confirm that the agent can use prior context.
5. Scroll the transcript upward after several rows. Confirm the welcome panel
   and command deck do not move.
6. Start a slow query, then press `Ctrl+C`. Confirm the status changes to
   `Stopping current request…`, the composer remains disabled, a late answer
   does not appear, and the cancellation notice appears only after work ends.
7. Start another slow query, press `Ctrl+Q`, and confirm the screen exits only
   after the request has ended.
8. Run `python src/app.py --repl`, submit one safe question or exit at the
   prompt, and confirm the legacy interface still starts.

Repeat layout checks at 60 columns by 24 rows and at a width below 60 columns:

- wordmark remains visible and welcome text wraps without clipping;
- composer stays fixed and usable;
- user and assistant rows have readable one-column side space;
- model facts reduce before activity text is lost;
- a wide Markdown table can scroll horizontally and does not lose columns;
- `Ctrl+C`, `Ctrl+Q`, and exit words still work.

Record the terminal size, tested question type, and pass or fail result. Do
not record credentials, private experiment names, or response data in a PR.

### Task 9: Run complete checks and review the diff

**Owner:** final review agent
**Files:** review all changed files. Do not make unrelated changes.

1. Run every supported automated check.
2. Inspect the final diff against the branch base. Check for a second query
   path, a direct worker-thread widget update, a missing cancellation guard,
   welcome content inside the transcript, a duplicate cleanup call, raw tool
   JSON, secrets in status, and accidental changes to the REPL or runtime.
3. Run whitespace checks. There is no configured linter in this repository.
   Do not add a linter as part of this UI task.
4. If a small safe test or style defect is found, fix it and repeat the focused
   test and the complete checks. Ask before a larger unrelated change.

**Required commands:**

```bash
PYTHONPATH=src python -m pytest -q
python -m compileall -q src
git diff --check
git status --short
git diff -- src/app.py src/agent/tui tests README.md environment.yml
```

## Safe parallel work split

Use up to three implementation agents after Task 1. Do not have two agents
edit the same file.

| Agent | Exclusive files | Dependency |
| --- | --- | --- |
| Presentation | `widgets/welcome.py`, `widgets/chat.py`, `styles/floki.tcss` | Follow the shared tree and class contract. Finish before final app visual test. |
| Block rendering | `blocks.py`, `tests/test_tui_blocks.py` | No dependency on app state. Finish before app success-path tests. |
| Application shell | `tui/app.py`, `tests/test_tui_app.py`, `src/app.py`, `tests/test_app_entry.py` | Use the widget names and TCSS IDs in this plan. Rebase its work only through normal shared working-tree coordination, not Git commands. |

Run Task 7 through Task 9 only after all three agents finish. If the
presentation agent must change `tests/test_tui_app.py`, let it add only widget
tests before the application-shell agent starts. Otherwise, the shell agent is
the sole owner of that file.

## Out of scope

- Token streaming.
- A permanent raw tool log or raw tool JSON view.
- A sidebar, experiment browser, or dashboard.
- A second table parser or `DataTable` conversion path.
- A new TUI framework or dependency change.
- A force-close key that can interrupt a synchronous network call.

## Completion checklist

- [ ] The persistent welcome panel is outside the scrolling transcript.
- [ ] The command deck is fixed and contains the input, status, and key help.
- [ ] The default command starts Textual. `--repl` starts the fallback REPL.
- [ ] One user action starts one traced query worker and no concurrent query.
- [ ] Status, success, error, cancellation, and deferred exit follow the
  shared contract.
- [ ] Formatter blocks and fallback answers render without raw tool JSON.
- [ ] Session memory, formatter, Langfuse, and REPL behavior pass contract
  review.
- [ ] Normal and narrow visual checks pass.
- [ ] Full tests, compile check, and diff check pass.
- [ ] Final diff has no unrelated changes or exposed secrets.
