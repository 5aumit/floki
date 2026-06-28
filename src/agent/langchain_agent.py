"""
LangGraph-based agent for MLflow Experiment Q&A
----------------------------------------------
This agent uses LangGraph to orchestrate LLM reasoning and MLflow tool calls.
A separate formatter LLM produces structured BlockResponse output for rendering.
"""


import os
import json
import sys
import time
import uuid
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from llm.inference_engine import get_llm_from_config, get_formatter_llm_from_config
from mlflow_tools import data_access
from llm.tracing import setup_langfuse, propagate_attributes
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from agent.agent_middleware import handle_tool_errors
from agent.context_memory import trim_messages_for_memory
from agent.response_formatter import format_to_block_response


from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

os.environ["MLFLOW_LOGGING_LEVEL"] = "WARNING"

from langchain.agents import create_agent


CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config.json'))
logging.info("Loading config from: %s", CONFIG_PATH)
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

llm_config = config.get('llm', {})
if not llm_config.get('gemini_api_key'):
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if gemini_api_key:
        llm_config['gemini_api_key'] = gemini_api_key

logging.info("Loading agent model: %s", llm_config.get('gemini_model', 'Not Set'))
logging.info("Loading formatter model: %s", llm_config.get('formatter_model', 'Not Set'))
llm = get_llm_from_config(llm_config)
formatter_llm = get_formatter_llm_from_config(llm_config)

langfuse_handler, fuse_client, conversation_id, lf_run, langfuse_user, FLUSH_PER_QUERY = setup_langfuse(config)
session_thread_id = conversation_id or f"session-{uuid.uuid4().hex[:8]}"

mlflow_tools = data_access.get_all_tools()
checkpointer = InMemorySaver()

AGENT_SYSTEM_PROMPT = (
    "You are a precise MLflow experiment assistant.\n"
    "1) Always use tools to fetch MLflow data. Never invent data.\n"
    "2) After tool results, write a clear markdown summary for the user.\n"
    "3) On tool errors, explain what went wrong.\n"
    "4) Ask for confirmation before destructive actions."
)

agent = create_agent(
    model=llm,
    tools=mlflow_tools,
    checkpointer=checkpointer,
    system_prompt=AGENT_SYSTEM_PROMPT,
    middleware=[handle_tool_errors],
)


def _build_invoke_config() -> dict:
    config_kwargs = {
        "configurable": {"thread_id": session_thread_id},
    }
    if langfuse_handler is not None:
        logging.info("Attaching Langfuse handler to agent invocation.")
        config_kwargs["callbacks"] = [langfuse_handler]
        if conversation_id is not None:
            meta = config_kwargs.setdefault("metadata", {})
            meta["conversation_id"] = conversation_id
            if langfuse_user:
                meta["user"] = langfuse_user
    return config_kwargs


def _persist_trimmed_memory(agent_result: dict, invoke_config: dict) -> None:
    trimmed = trim_messages_for_memory(agent_result["messages"])
    agent.update_state(
        invoke_config,
        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed]},
    )


def run_query(user_query: str) -> dict:
    messages = [{"role": "user", "content": user_query}]
    invoke_config = _build_invoke_config()
    agent_result = agent.invoke({"messages": messages}, config=invoke_config)

    structured = format_to_block_response(
        formatter_llm,
        agent_result["messages"],
        user_query,
    )
    _persist_trimmed_memory(agent_result, invoke_config)
    return {
        "structured_response": structured,
        "messages": agent_result["messages"],
    }


def _print_result(result):
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


def loading_animation(message, duration=3):
    spinner = ['|', '/', '-', '\\']
    print(message, end='', flush=True)
    print('\n', end='', flush=True)
    for i in range(duration * 4):
        print(f' {spinner[i % 4]}', end='\r', flush=True)
        time.sleep(0.25)
    print(' ' * (len(message) + 2), end='\r')


def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Initializing agent and loading tools...")
    loading_animation("Starting up, please wait...", duration=3)

    try:
        import console_ui as ui
        ui.print_welcome()
    except Exception:
        logging.info("console_ui not available for welcome banner")
    if fuse_client is not None:
        print(f"Langfuse tracing enabled. See at {os.getenv('LANGFUSE_BASE_URL', 'your Langfuse dashboard')}")

    if conversation_id is not None and fuse_client is not None:
        session_prop = propagate_attributes(session_id=conversation_id, user_id=langfuse_user)
    else:
        session_prop = None

    if session_prop is not None:
        with session_prop:
            _interactive_loop(fuse_client)
    else:
        _interactive_loop(fuse_client)


def _get_user_input():
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import HTML
        session = PromptSession()
        prompt_html = HTML('<ansiblue>You</ansiblue> ')
        return session.prompt(prompt_html)
    except Exception:
        try:
            import console_ui as ui
            return ui.console.input("\n[bold blue]You[/bold blue] ")
        except Exception:
            return input("\n> ")


def _extract_output_snippet(result) -> str:
    try:
        structured = result.get("structured_response")
        if structured is not None:
            return json.dumps(structured, default=str)
    except Exception:
        pass
    try:
        return result.get('messages')[-1].content
    except Exception:
        try:
            return str(result)
        except Exception:
            return None


def _interactive_loop(fuse_client_local):
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
            if fuse_client_local is not None and not FLUSH_PER_QUERY:
                try:
                    fuse_client_local.flush()
                except Exception:
                    pass
            print("Goodbye!")
            break

        if fuse_client_local is not None and hasattr(fuse_client_local, 'start_as_current_observation'):
            try:
                with fuse_client_local.start_as_current_observation(as_type="span", name="langchain-call") as obs_ctx:
                    try:
                        obs_ctx.update(input={"query": user_query})
                    except Exception:
                        pass
                    result = run_query(user_query)
                    _print_result(result)
                    output_snippet = _extract_output_snippet(result)
                    if output_snippet is not None:
                        try:
                            obs_ctx.update(output={"result": output_snippet})
                        except Exception:
                            pass
                    if FLUSH_PER_QUERY:
                        try:
                            fuse_client_local.flush()
                        except Exception:
                            pass
            except Exception:
                result = run_query(user_query)
                _print_result(result)
        else:
            result = run_query(user_query)
            _print_result(result)
