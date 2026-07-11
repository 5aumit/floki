"""
LangGraph-based agent for MLflow Experiment Q&A
----------------------------------------------
This agent uses LangGraph to orchestrate LLM reasoning and MLflow tool calls.
A separate formatter LLM produces structured BlockResponse output for rendering.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

load_dotenv()

import logging as _logging

_logging.basicConfig(level=_logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

os.environ["MLFLOW_LOGGING_LEVEL"] = "WARNING"

from langchain.agents import create_agent
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from agent.agent_middleware import handle_tool_errors
from agent.context_memory import trim_messages_for_memory
from agent.response_formatter import format_to_block_response
from llm.inference_engine import get_formatter_llm_from_config, get_llm_from_config
from llm.tracing import propagate_attributes, setup_langfuse

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config.json"))

AGENT_SYSTEM_PROMPT = (
    "You are a precise MLflow experiment assistant.\n"
    "1) Always use tools to fetch MLflow data. Never invent data.\n"
    "2) After tool results, write a clear markdown summary for the user.\n"
    "3) On tool errors, explain what went wrong.\n"
    "4) Ask for confirmation before destructive actions."
)


@dataclass
class AgentRuntime:
    config: dict
    llm: Any
    formatter_llm: Any
    agent: Any
    langfuse_handler: Any
    fuse_client: Any
    conversation_id: Optional[str]
    lf_run: Any
    langfuse_user: Optional[str]
    flush_per_query: bool
    session_thread_id: str


_runtime: Optional[AgentRuntime] = None


def _load_config() -> dict:
    logging.info("Loading config from: %s", CONFIG_PATH)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    llm_config = config.setdefault("llm", {})
    if not llm_config.get("gemini_api_key"):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            llm_config["gemini_api_key"] = gemini_api_key
    return config


def _build_runtime() -> AgentRuntime:
    from mlflow_tools import data_access

    config = _load_config()
    llm_config = config.get("llm", {})
    logging.info("Loading agent model: %s", llm_config.get("gemini_model", "Not Set"))
    logging.info("Loading formatter model: %s", llm_config.get("formatter_model", "Not Set"))

    llm = get_llm_from_config(llm_config)
    formatter_llm = get_formatter_llm_from_config(llm_config)
    langfuse_handler, fuse_client, conversation_id, lf_run, langfuse_user, flush_per_query = setup_langfuse(config)
    session_thread_id = conversation_id or f"session-{uuid.uuid4().hex[:8]}"

    mlflow_tools = data_access.get_all_tools()
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=llm,
        tools=mlflow_tools,
        checkpointer=checkpointer,
        system_prompt=AGENT_SYSTEM_PROMPT,
        middleware=[handle_tool_errors],
    )

    return AgentRuntime(
        config=config,
        llm=llm,
        formatter_llm=formatter_llm,
        agent=agent,
        langfuse_handler=langfuse_handler,
        fuse_client=fuse_client,
        conversation_id=conversation_id,
        lf_run=lf_run,
        langfuse_user=langfuse_user,
        flush_per_query=flush_per_query,
        session_thread_id=session_thread_id,
    )


def get_runtime() -> AgentRuntime:
    global _runtime
    if _runtime is None:
        _runtime = _build_runtime()
    return _runtime


def _build_invoke_config(rt: AgentRuntime) -> dict:
    config_kwargs: dict = {
        "configurable": {"thread_id": rt.session_thread_id},
    }
    if rt.langfuse_handler is not None:
        logging.info("Attaching Langfuse handler to agent invocation.")
        config_kwargs["callbacks"] = [rt.langfuse_handler]
        meta = config_kwargs.setdefault("metadata", {})
        if rt.conversation_id is not None:
            # CallbackHandler promotes these to trace sessionId / userId
            meta["langfuse_session_id"] = rt.conversation_id
            meta["conversation_id"] = rt.conversation_id
        if rt.langfuse_user:
            meta["langfuse_user_id"] = rt.langfuse_user
    return config_kwargs


def _persist_trimmed_memory(rt: AgentRuntime, agent_result: dict, invoke_config: dict) -> None:
    trimmed = trim_messages_for_memory(agent_result["messages"])
    rt.agent.update_state(
        invoke_config,
        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed]},
    )


def run_query(user_query: str) -> dict:
    rt = get_runtime()
    messages = [{"role": "user", "content": user_query}]
    invoke_config = _build_invoke_config(rt)
    agent_result = rt.agent.invoke({"messages": messages}, config=invoke_config)

    structured = format_to_block_response(
        rt.formatter_llm,
        agent_result["messages"],
        user_query,
    )
    _persist_trimmed_memory(rt, agent_result, invoke_config)
    return {
        "structured_response": structured,
        "messages": agent_result["messages"],
    }


def _extract_output_snippet(result: dict) -> Optional[str]:
    try:
        structured = result.get("structured_response")
        if structured is not None:
            return json.dumps(structured, default=str)
    except Exception:
        pass
    try:
        return result.get("messages")[-1].content
    except Exception:
        try:
            return str(result)
        except Exception:
            return None


def run_query_traced(user_query: str) -> dict:
    """Run query with Langfuse span + session/user attributes (safe for worker threads)."""
    rt = get_runtime()
    fuse_client = rt.fuse_client

    def _with_span() -> dict:
        if fuse_client is None or not hasattr(fuse_client, "start_as_current_observation"):
            return run_query(user_query)
        try:
            obs_cm = fuse_client.start_as_current_observation(as_type="span", name="langchain-call")
        except Exception:
            logging.exception("Failed to open Langfuse span; running untraced")
            return run_query(user_query)

        with obs_cm as obs_ctx:
            try:
                obs_ctx.update(input={"query": user_query})
            except Exception:
                pass
            result = run_query(user_query)
            output_snippet = _extract_output_snippet(result)
            if output_snippet is not None:
                try:
                    obs_ctx.update(output={"result": output_snippet})
                except Exception:
                    pass
            if rt.flush_per_query:
                try:
                    fuse_client.flush()
                except Exception:
                    pass
            return result

    if rt.conversation_id and propagate_attributes is not None:
        with propagate_attributes(
            session_id=rt.conversation_id,
            user_id=rt.langfuse_user,
        ):
            return _with_span()
    return _with_span()


def finish_langfuse() -> None:
    rt = _runtime
    if rt is None:
        return
    try:
        if rt.lf_run is not None:
            if hasattr(rt.lf_run, "finish"):
                rt.lf_run.finish()
            elif rt.fuse_client is not None and hasattr(rt.fuse_client, "finish_run"):
                rt.fuse_client.finish_run(rt.conversation_id)
        if rt.fuse_client is not None:
            rt.fuse_client.flush()
    except Exception:
        pass


def flush_langfuse_on_exit() -> None:
    rt = _runtime
    if rt is None or rt.fuse_client is None or rt.flush_per_query:
        return
    try:
        rt.fuse_client.flush()
    except Exception:
        pass


@contextmanager
def langfuse_session_context() -> Iterator[None]:
    rt = get_runtime()
    if rt.conversation_id and rt.fuse_client and propagate_attributes is not None:
        with propagate_attributes(session_id=rt.conversation_id, user_id=rt.langfuse_user):
            yield
    else:
        yield


def _print_result(result: dict) -> None:
    try:
        import console_ui as ui

        ui.print_result(result)
        return
    except Exception:
        pass
    try:
        print(f"\n{result['messages'][-1].content}")
    except Exception:
        try:
            print(f"\n{result}")
        except Exception:
            pass


def loading_animation(message: str, duration: int = 3) -> None:
    spinner = ["|", "/", "-", "\\"]
    print(message, end="", flush=True)
    print("\n", end="", flush=True)
    for i in range(duration * 4):
        print(f" {spinner[i % 4]}", end="\r", flush=True)
        time.sleep(0.25)
    print(" " * (len(message) + 2), end="\r")


def _get_user_input() -> str:
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import HTML

        session = PromptSession()
        prompt_html = HTML("<ansiblue>You</ansiblue> ")
        return session.prompt(prompt_html)
    except Exception:
        try:
            import console_ui as ui

            return ui.console.input("\n[bold blue]You[/bold blue] ")
        except Exception:
            return input("\n> ")


def _interactive_loop(fuse_client_local: Any) -> None:
    while True:
        try:
            user_query = _get_user_input()
        except EOFError:
            print("\nGoodbye!")
            break
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            continue
        if user_query.strip().lower() in {"exit", "quit"}:
            flush_langfuse_on_exit()
            print("Goodbye!")
            break

        result = run_query_traced(user_query)
        _print_result(result)


def main() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    print("Initializing agent and loading tools...")
    loading_animation("Starting up, please wait...", duration=3)
    get_runtime()

    try:
        import console_ui as ui

        ui.print_welcome()
    except Exception:
        logging.info("console_ui not available for welcome banner")

    rt = get_runtime()
    if rt.fuse_client is not None:
        print(f"Langfuse tracing enabled. See at {os.getenv('LANGFUSE_BASE_URL', 'your Langfuse dashboard')}")

    with langfuse_session_context():
        _interactive_loop(rt.fuse_client)


def main_tui() -> None:
    from tui.app import run_tui

    run_tui()


# Backward-compatible module-level names for app.py imports after runtime init
def __getattr__(name: str) -> Any:
    if name in ("fuse_client", "conversation_id", "lf_run", "FLUSH_PER_QUERY"):
        rt = get_runtime()
        return {
            "fuse_client": rt.fuse_client,
            "conversation_id": rt.conversation_id,
            "lf_run": rt.lf_run,
            "FLUSH_PER_QUERY": rt.flush_per_query,
        }[name]
    raise AttributeError(name)
