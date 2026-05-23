"""
LangGraph-based agent for MLflow Experiment Q&A
----------------------------------------------
This agent uses LangGraph to orchestrate LLM reasoning and MLflow tool calls.
"""


import os
import json
import sys
import time
import threading
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from llm.inference_engine import GroqEngine, get_llm_from_config
from mlflow_tools import data_access
from llm.tracing import setup_langfuse, propagate_attributes
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from agent.agent_middleware import handle_tool_errors, BLOCK_RESPONSE_SCHEMA


from dotenv import load_dotenv
load_dotenv()

import logging
# logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

os.environ["MLFLOW_LOGGING_LEVEL"] = "WARNING"

import asyncio
import json
from langchain.agents import create_agent            # correct agent factory
from langchain.tools import tool     


CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config.json'))
logging.info("Loading config from: %s", CONFIG_PATH)
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# LLM selection logic
llm_config = config.get('llm', {})
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
llm_config['groq_api_key'] = GROQ_API_KEY
logging.info("Loading model from config: %s", llm_config.get('groq_model', 'Not Set'))
llm = get_llm_from_config(llm_config)


# Langfuse/tracing setup
langfuse_handler, fuse_client, conversation_id, lf_run, langfuse_user, FLUSH_PER_QUERY = setup_langfuse(config)

mlflow_tools = data_access.get_all_tools()
checkpointer = InMemorySaver()



agent = create_agent(
    model=llm,
    tools=mlflow_tools,
    checkpointer=checkpointer,
    response_format=ToolStrategy(schema=BLOCK_RESPONSE_SCHEMA),
    system_prompt=(
        "You are a precise MLflow experiment assistant. RULES:\n"
        "1) Always use the provided tools to fetch or query MLflow data. Do not invent or guess information.\n"
        "2) Keep responses concise and focused on the user's MLflow goals.\n"
        "3) If a tool encounters an error, explain the issue in a text block. Do not raise exceptions.\n"
        "4) For destructive actions, ask for explicit confirmation in a text block before proceeding.\n"
        "\n"
        "OUTPUT SCHEMA (edit this section if needed):\n"
        "- Return JSON only, in this exact shape:\n"
        "  {\"blocks\": [{\"type\": \"text\", \"markdown\": \"...\"} | {\"type\": \"table\", \"markdown\": \"|h|...\"}]}\n"
        "- Use type=\"text\" for analysis, summaries, and next steps.\n"
        "- Use type=\"table\" only for clean Markdown pipe tables when comparisons are requested or helpful.\n"
        "- Do not use TextBlock/TableBlock or any other keys; only blocks/type/markdown are allowed."
    ),
    middleware=[handle_tool_errors],
    
)


def run_query(user_query: str):
    messages = [{"role": "user", "content": user_query}]
    config_kwargs = {}
    if langfuse_handler is not None:
        logging.info("Attaching Langfuse handler to agent invocation.")
        # attach handler
        config_kwargs['callbacks'] = [langfuse_handler]
        config_kwargs['configurable'] = {'thread_id': conversation_id or "default_thread"}
        # include conversation metadata if agent/client supports it
        if conversation_id is not None:
            meta = config_kwargs.setdefault('metadata', {})
            meta['conversation_id'] = conversation_id
            if langfuse_user: 
                meta['user'] = langfuse_user
    # Only pass config when non-empty to avoid passing None handlers
    if config_kwargs:
        result = agent.invoke({"messages": messages}, config=config_kwargs)
    else:
        result = agent.invoke({"messages": messages})
    return result


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
    print("\n==============================")
    print("  Welcome to MLflow Agent CLI  ")
    print("==============================")
    print("Initializing agent and loading tools...")
    loading_animation("Starting up, please wait...", duration=3)
    
    print("Agent is ready! Type 'exit' to quit.")
    # Try to show a colorful welcome banner (non-fatal)
    try:
        import console_ui as ui
        ui.print_welcome()
    except Exception:
        logging.info("console_ui not available for welcome banner")
    # Print tracing info if enabled
    if fuse_client is not None:
        print(f"Langfuse tracing enabled. See at {os.getenv('LANGFUSE_BASE_URL', 'your Langfuse dashboard')}")

    # Enter session-level attribute propagation for grouping traces/observations
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
    """Get user input using prompt_toolkit if available, otherwise fall back.

    Tries prompt_toolkit (rich features), then Rich Console.input, then builtin input.
    Raises EOFError up to the caller to handle termination.
    """
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
            # last fallback to builtin input; let EOFError bubble up
            return input("\n> ")


def _interactive_loop(fuse_client_local):
    while True:
        try:
            user_query = _get_user_input()
        except EOFError:
            print("\nGoodbye!")
            break
        except KeyboardInterrupt:
            # User pressed Ctrl-C; continue the loop to allow graceful exit
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

        # Create a root observation/span for this query so trace-level IO is populated
        if fuse_client_local is not None and hasattr(fuse_client_local, 'start_as_current_observation'):
            try:
                with fuse_client_local.start_as_current_observation(as_type="span", name="langchain-call") as obs_ctx:
                    try:
                        obs_ctx.update(input={"query": user_query})
                    except Exception:
                        pass
                    result = run_query(user_query)
                    _print_result(result)
                    # Try to extract a readable output snippet
                    output_snippet = None
                    try:
                        output_snippet = result.get('messages')[-1].content
                    except Exception:
                        try:
                            output_snippet = str(result)
                        except Exception:
                            output_snippet = None
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
                # Fallback to running without explicit observation context
                result = run_query(user_query)
                _print_result(result)
        else:
            # No Langfuse client or observation support; just run the query
            result = run_query(user_query)
            _print_result(result)
