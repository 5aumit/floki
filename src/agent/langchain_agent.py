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
from langfuse import get_client
from langfuse.langchain import CallbackHandler


from dotenv import load_dotenv
load_dotenv()

import logging
# logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

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
logging.info("Loading model from config: %s", llm_config.get('groq_model', 'Not Set'))
llm = get_llm_from_config(llm_config)

# Langfuse setup — use environment variables only for simplicity
langfuse_handler = None
public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
host = os.getenv('LANGFUSE_BASE_URL')
try:
    if public_key:
        # Prefer passing public_key; avoid secret_key kwarg for compatibility
        try:
            fuse_client = get_client(public_key=public_key, host=host) if host else get_client(public_key=public_key)
        except TypeError:
            # Fallback: set env vars and call get_client()
            os.environ.setdefault('LANGFUSE_PUBLIC_KEY', public_key)
            if host:
                os.environ.setdefault('LANGFUSE_BASE_URL', host)
            fuse_client = get_client()
        try:
            langfuse_handler = CallbackHandler(client=fuse_client)
        except TypeError:
            langfuse_handler = CallbackHandler()
        logging.info("Langfuse handler initialized.")
    else:
        logging.info("LANGFUSE_PUBLIC_KEY not set; Langfuse tracing disabled.")
except Exception as e:
    logging.exception("Failed to initialize Langfuse handler: %s", e)
    langfuse_handler = None

mlflow_tools = data_access.get_all_tools()

agent = create_agent(
    model=llm,
    tools=mlflow_tools,
    system_prompt="You are an MLflow experiment assistant. Use tools as needed.",
)


def run_query(user_query: str):
    messages = [{"role": "user", "content": user_query}]
    config_kwargs = {}
    if langfuse_handler is not None:
        logging.info("Attaching Langfuse handler to agent invocation.")
        config_kwargs['callbacks'] = [langfuse_handler]
    # Only pass config when non-empty to avoid passing None handlers
    if config_kwargs:
        result = agent.invoke({"messages": messages}, config=config_kwargs)
    else:
        result = agent.invoke({"messages": messages})
    return result



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
    while True:
        user_query = input("\n> ")
        if user_query.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        result = run_query(user_query)
        # Print the last message content if available, else print the whole result
        try:
            print(f"\n{result['messages'][-1].content}")
        except Exception:
            print(f"\n{result}")

if __name__ == "__main__":
    main()
