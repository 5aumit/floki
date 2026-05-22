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
from agent.agent_middleware import handle_tool_errors, classify_and_set_schema


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
    system_prompt=(
        "You are a precise MLflow experiment assistant. RULES:\n"
        "1) Always use the provided tools to fetch or query MLflow data. Do not invent or guess run IDs, experiment IDs, metrics, parameters, or artifact locations.\n"
        "2) When a user requests data (experiments, runs, metrics, params, artifacts), CALL the appropriate tool and DO NOT embed raw data in your assistant message. The tool's structured output will be rendered by the UI.\n"
        "3) After a tool call, provide a short natural-language summary (no tables, no code blocks) of <=2 sentences describing the high-level result and next steps.\n"
        "4) If asked to return data directly, return valid JSON only (array or object), no Markdown, no ASCII tables.\n"
        "5) On tool errors, return a JSON object: {\"error\": <code>, \"message\": <human message>}. Do not raise exceptions.\n"
        "6) For any action that may be destructive, ask for explicit confirmation before proceeding.\n"
        "7) Keep responses concise and focused on user's goal.\n"
        "Adhere strictly to these rules."
    ),
    middleware=[classify_and_set_schema, handle_tool_errors],
    
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


def _interactive_loop(fuse_client_local):
    while True:
        try:
            user_query = input("\n> ")
        except EOFError:
            print("\nGoodbye!")
            break
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

