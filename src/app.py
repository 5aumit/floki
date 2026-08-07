import argparse
import os
import sys
from collections.abc import Sequence

_AGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from langchain_agent import (
    finish_langfuse,
    flush_langfuse_on_exit,
    main as run_legacy_repl,
    main_tui,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the two supported interactive interface choices."""
    parser = argparse.ArgumentParser(description="Floki MLflow experiment assistant")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tui", action="store_true", help="Run the Textual TUI (default)")
    group.add_argument("--repl", action="store_true", help="Run the Rich REPL")
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> None:
    """Start the selected interface and keep startup interrupt cleanup available."""
    args = _parse_args(argv)
    try:
        if args.repl:
            run_legacy_repl()
            # The legacy loop owns its normal exit handling. Keep the outer
            # fallback for EOF and other normal returns in that interface only.
            flush_langfuse_on_exit()
        else:
            # No flag and the compatible --tui flag both select Textual. The
            # Textual App owns its single normal shutdown cleanup path.
            main_tui()
    except KeyboardInterrupt:
        print("\nInterrupted. Finishing Langfuse run if present...")
        finish_langfuse()


if __name__ == "__main__":
    run()
