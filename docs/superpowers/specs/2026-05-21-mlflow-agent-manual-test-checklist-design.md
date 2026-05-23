# Manual MLflow Agent Test Checklist — Design

Date: 2026-05-21  
Owner: Saumit Paul  
Status: Draft

## Overview
Design a **manual prompt checklist** (with expected outcomes) that validates a CLI agent’s real-world MLflow workflows. The checklist is dataset-agnostic and uses placeholders, so it can run against any shared MLflow tracking DB.

## Goals
- Provide a repeatable manual test suite that mirrors ML engineer workflows.
- Ensure answers are grounded in MLflow data with concrete experiment/run references.
- Validate summaries, comparisons, regressions, artifacts, failures, and reproducibility.
- Make pass/fail decisions unambiguous through explicit acceptance criteria.

## Non-goals
- Automating tests in pytest or building test harnesses.
- Writing synthetic data or seeding MLflow.
- Enforcing UI/UX formatting beyond the required fields.

## Assumptions & Constraints
- All engineers use a shared MLflow DB with populated experiments and runs.
- Prompts are run via the CLI agent, which can call MLflow tools.
- Placeholders (e.g., `<experiment_name>`) will be replaced by real values.
- The agent should not hallucinate experiments/runs.

## Checklist Format (per item)
**ID** · **Goal** · **Prompt** · **Expected outcome** · **Pass/Fail checks** · **Notes/Assumptions**

## Acceptance Criteria (applies to every item)
1. **Grounding:** Response cites concrete experiment/run IDs or names from MLflow; no invented entities.
2. **Completeness:** Required fields are present (e.g., run_id, metric values, date ranges).
3. **Correctness:** Rankings/aggregations match MLflow data (e.g., top 5 truly top by metric).
4. **Missing-data handling:** Absent fields are explicitly called out.
5. **Concision:** Avoid full metric/param dumps unless asked.
6. **Tooling:** Uses MLflow tools; does not answer from assumptions.

---

## Category A — Recency & Team Activity

**REC-1**  
**Goal:** Summarize the last week’s work.  
**Prompt:** “Summarize experiments and runs created or updated in `<date_range>` (last 7 days), grouped by experiment.”  
**Expected outcome:** List of experiments (name + ID), run counts, notable metrics/tags, brief summary per experiment.  
**Pass/Fail checks:** Grounded IDs, date range stated, counts present.  
**Notes:** Use `<date_range>` placeholder.

**REC-2**  
**Goal:** Identify most active experiments.  
**Prompt:** “Which experiments had the most new runs in `<date_range>`?”  
**Expected outcome:** Ranked list with experiment IDs, run counts, and date range.  
**Pass/Fail checks:** Correct ordering, counts shown.

**REC-3**  
**Goal:** Highlight significant recent runs.  
**Prompt:** “Show the top 3 best runs added in `<date_range>` by `<metric_name>` across all experiments.”  
**Expected outcome:** Table with run_id, experiment_name, metric value, key params.  
**Pass/Fail checks:** Proper ranking; run IDs present.

## Category B — Experiment Discovery & Metadata

**DISC-1**  
**Goal:** Find experiments by dataset/model tags.  
**Prompt:** “List experiments that mention `<dataset>` or `<model_type>` in their tags or names.”  
**Expected outcome:** Experiment IDs/names with tag evidence.  
**Pass/Fail checks:** No hallucinations; evidence cited.

**DISC-2**  
**Goal:** Provide a quick experiment overview.  
**Prompt:** “Give a one‑paragraph overview of `<experiment_name>` including run count and top metric.”  
**Expected outcome:** Experiment ID, run count, top metric summary.  
**Pass/Fail checks:** Includes ID and metric name/value.

**DISC-3**  
**Goal:** Inventory experiments by owner/team tag.  
**Prompt:** “Show experiments tagged with `<tag_key>=<tag_value>`.”  
**Expected outcome:** Experiment list with IDs and tag proof.  
**Pass/Fail checks:** Tag evidence present.

## Category C — Run Comparison & Leaderboards

**COMP-1**  
**Goal:** Compare top runs within an experiment.  
**Prompt:** “Compare the top 5 runs in `<experiment_name>` by `<metric_name>`.”  
**Expected outcome:** Table with run_id, metric, key params; sorted by metric.  
**Pass/Fail checks:** Ordering correct; includes run IDs and metric values.

**COMP-2**  
**Goal:** Compare two specific runs.  
**Prompt:** “Compare run `<run_id_a>` vs `<run_id_b>` on metrics and params.”  
**Expected outcome:** Side‑by‑side comparison of key metrics/params.  
**Pass/Fail checks:** Both run IDs referenced; no missing fields unacknowledged.

**COMP-3**  
**Goal:** Compare runs by a tag filter.  
**Prompt:** “Among runs tagged `<tag_key>=<tag_value>` in `<experiment_name>`, show the best 3 by `<metric_name>`.”  
**Expected outcome:** Filtered ranked list with run IDs and metric values.  
**Pass/Fail checks:** Filter applied; ranking correct.

## Category D — Regression Tracking

**REG-1**  
**Goal:** Detect metric regression over time.  
**Prompt:** “Has `<metric_name>` regressed in `<experiment_name>` over `<date_range>` compared to the prior period?”  
**Expected outcome:** Regression/Stable/Improved + cited runs/metrics.  
**Pass/Fail checks:** Includes referenced runs and date windows.

**REG-2**  
**Goal:** Identify last known good run.  
**Prompt:** “What was the last run in `<experiment_name>` before `<date_range>` that achieved `<metric_name>` ≥ `<threshold>`?”  
**Expected outcome:** Single run_id with metric value and timestamp.  
**Pass/Fail checks:** Run ID and timestamp present.

**REG-3**  
**Goal:** Explain regression drivers.  
**Prompt:** “If there’s a regression, which params changed most between the best run last period and best run this period?”  
**Expected outcome:** Param diffs tied to specific runs.  
**Pass/Fail checks:** References both runs and their params.

## Category E — Hyperparameter Sweeps

**HYP-1**  
**Goal:** Identify impactful hyperparameters.  
**Prompt:** “Which hyperparameters most influence `<metric_name>` in `<experiment_name>`?”  
**Expected outcome:** Ranked list with evidence from runs.  
**Pass/Fail checks:** Mentions run IDs or aggregated evidence.

**HYP-2**  
**Goal:** Find optimal param value.  
**Prompt:** “For `<param_name>`, which value correlates with the best `<metric_name>`?”  
**Expected outcome:** Best value + supporting runs.  
**Pass/Fail checks:** Value and run evidence present.

**HYP-3**  
**Goal:** Sweep coverage.  
**Prompt:** “How many unique values were tried for `<param_name>` in `<experiment_name>`?”  
**Expected outcome:** Count and list of values (if small).  
**Pass/Fail checks:** Count present; values or explicit truncation.

## Category F — Artifact Inspection

**ART-1**  
**Goal:** Locate best model artifact.  
**Prompt:** “Show the model artifact path for the best run in `<experiment_name>` by `<metric_name>`.”  
**Expected outcome:** Run_id, artifact URI/path, metric value.  
**Pass/Fail checks:** All fields present.

**ART-2**  
**Goal:** Retrieve evaluation artifacts.  
**Prompt:** “List available evaluation artifacts (e.g., confusion matrix) for run `<run_id>`.”  
**Expected outcome:** Artifact list with paths.  
**Pass/Fail checks:** Run_id referenced; artifacts enumerated or ‘none found’.

**ART-3**  
**Goal:** Compare artifacts across runs.  
**Prompt:** “Compare the evaluation artifacts for the top 2 runs in `<experiment_name>`.”  
**Expected outcome:** Run IDs and artifact differences.  
**Pass/Fail checks:** Evidence of both runs and artifacts.

## Category G — Failure & Debugging

**DBG-1**  
**Goal:** Find failed runs.  
**Prompt:** “List runs marked FAILED in `<date_range>` with their error tags or notes.”  
**Expected outcome:** Run IDs, failure status, error tags/notes.  
**Pass/Fail checks:** Status and error evidence included.

**DBG-2**  
**Goal:** Detect missing metrics.  
**Prompt:** “Are there runs in `<experiment_name>` missing `<metric_name>`?”  
**Expected outcome:** Run IDs missing the metric and count.  
**Pass/Fail checks:** Explicit missing list or clear ‘none’.

**DBG-3**  
**Goal:** Negative test for nonexistent experiment.  
**Prompt:** “Summarize `<nonexistent_experiment>`.”  
**Expected outcome:** Clear ‘not found’ response with a suggestion to list experiments.  
**Pass/Fail checks:** No hallucinated results.

## Category H — Reproducibility & Lineage

**REP-1**  
**Goal:** Produce a reproducibility recipe.  
**Prompt:** “Give a reproducibility recipe for run `<run_id>` (params, git SHA, data version, env).”  
**Expected outcome:** All fields listed; missing fields explicitly stated.  
**Pass/Fail checks:** Clear list + missing data callouts.

**REP-2**  
**Goal:** Identify lineage for a best run.  
**Prompt:** “For the best run in `<experiment_name>`, list dataset version, code version, and model artifact.”  
**Expected outcome:** Run_id, dataset tag, git SHA/tag, artifact path.  
**Pass/Fail checks:** All fields present or missing noted.

**REP-3**  
**Goal:** Detect inconsistent tagging.  
**Prompt:** “Which runs in `<experiment_name>` are missing a git SHA tag?”  
**Expected outcome:** Run IDs missing the tag; count.  
**Pass/Fail checks:** Count and run IDs or explicit ‘none’.

## Category I — Performance & Governance

**PERF-1**  
**Goal:** Identify slow runs.  
**Prompt:** “Which runs in `<experiment_name>` are slowest (by duration), and how do their params differ?”  
**Expected outcome:** Ranked durations + run IDs + parameter diffs.  
**Pass/Fail checks:** Durations and run IDs included.

**PERF-2**  
**Goal:** Find unusually long runs.  
**Prompt:** “List runs with duration > `<duration_threshold>` in `<date_range>`.”  
**Expected outcome:** Run IDs, durations, experiment names.  
**Pass/Fail checks:** Threshold applied; IDs present.

**GOV-1**  
**Goal:** Identify stale experiments.  
**Prompt:** “Which experiments have no runs in `<date_range>`?”  
**Expected outcome:** Experiment IDs/names with zero‑run confirmation.  
**Pass/Fail checks:** IDs present; date range stated.

**GOV-2**  
**Goal:** Identify orphaned runs.  
**Prompt:** “Are there runs without a meaningful experiment description/tag?”  
**Expected outcome:** Run IDs or experiment IDs lacking descriptions/tags.  
**Pass/Fail checks:** Evidence of missing metadata.

---

## Execution Notes
- Replace placeholders with real values from your MLflow DB.
- Start with REC-1 to validate basic connectivity and grounding.
- If any item fails grounding or correctness, capture the prompt and response verbatim for triage.
