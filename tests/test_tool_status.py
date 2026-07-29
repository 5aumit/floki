"""verb_for_tool mapping — one runnable check for tool status verbs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.tool_status import DEFAULT_VERB, FORMAT_VERB, verb_for_tool


def test_verb_for_known_tools():
    assert verb_for_tool("list_experiments_tool") == "listing"
    assert verb_for_tool("list_runs_tool") == "scanning"
    assert verb_for_tool("count_runs_per_experiment_tool") == "counting"
    assert verb_for_tool("get_run_metrics_tool") == "fetching"
    assert verb_for_tool("get_run_params_tool") == "inspecting"
    assert verb_for_tool("find_best_runs_by_metric_tool") == "ranking"
    assert verb_for_tool("check_experiment_generalization_tool") == "checking"


def test_verb_fallback_thinking():
    assert verb_for_tool(None) == DEFAULT_VERB
    assert verb_for_tool("") == DEFAULT_VERB
    assert verb_for_tool("unknown_tool_xyz") == "thinking"
    assert DEFAULT_VERB == "thinking"
    assert FORMAT_VERB == "formatting"


def test_tool_name_from_request():
    from agent.tool_status import tool_name_from_request

    class Req:
        tool_call = {"name": "list_runs_tool", "id": "1", "args": {}}

    assert tool_name_from_request(Req()) == "list_runs_tool"
    assert verb_for_tool(tool_name_from_request(Req())) == "scanning"
