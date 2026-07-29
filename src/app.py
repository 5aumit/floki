import argparse
import os
import sys

_AGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from langchain_agent import finish_langfuse, flush_langfuse_on_exit, main, main_tui


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Floki MLflow experiment assistant")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tui", action="store_true", help="Run the Textual TUI")
    group.add_argument("--repl", action="store_true", help="Run the Rich REPL (default)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        if args.tui:
            main_tui()
        else:
            main()
    except KeyboardInterrupt:
        print("\nInterrupted. Finishing Langfuse run if present...")
        finish_langfuse()
    else:
        flush_langfuse_on_exit()
