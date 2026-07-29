# Floki 🧭 — MLflow Experiment Agentic Chatbot

![image](assets/floki-banner-2.png)

> **⚠️ Work in Progress:** This project is actively being developed. Features, structure, and documentation may change frequently.

Floki is named after the legendary Viking engineer Flóki Vilgerðarson, who built innovative boats that enabled Vikings to explore new lands. This project aims to empower ML researchers to explore their experiment logs with the same spirit of discovery.

A CLI-based assistant for ML experimentation, inspired by Claude Code, that helps researchers query, analyze, and gain insights from MLflow experiment logs.

**Demo**
> One Dark TUI: welcome banner, tool-verb status animation, and structured answer blocks.

![image](assets/floki-demo.gif)

## Quick Setup

### 1) Create and activate the local environment

```bash
bash setup_env.sh
conda activate ./mlflow_agentic_env
```

`setup_env.sh` is the supported setup path. It creates or updates a project-local
Conda environment from `environment.yml` and verifies the TUI and test dependencies.

### 2) Add required API keys

Copy the safe template, then adjust the MLflow location if needed:

```bash
cp config.example.json config.json
```

Create a `.env` file in the project root or export the variables into your shell.
Only Gemini is required to run the agent. Langfuse variables are optional and enable tracing:

```
GEMINI_API_KEY=your_gemini_api_key_here
# Optional tracing
LANGFUSE_PUBLIC_KEY=your_langfuse_api_key_here
LANGFUSE_SECRET_KEY=your_langfuse_api_key_here
LANGFUSE_BASE_URL="https://us.cloud.langfuse.com"
```

Model configuration lives in `config.json` under `llm`:

- `gemini_model` — agent LLM for tool calling (default: `gemini-2.5-flash`)
- `formatter_model` — formatter LLM for structured UI output (default: `gemini-2.5-flash-lite`)

The agent runs in two phases: first it calls MLflow tools and drafts an answer, then the formatter produces a structured `BlockResponse` for rendering.

### Sample MLflow data

The repository includes a small SQLite MLflow database at `data/mlflow.db` for local
exploration. The example configuration points to it. Start the MLflow UI with:

```bash
bash src/scripts/run_mlflow.sh
```

For your own tracking server, change `mlflow.mlruns_dir` in `config.json` to its
MLflow tracking URI.

The sample database includes run metadata and metrics. Its historical artifact files
are not included; new local artifacts are stored under `data/artifacts/`.

### 3) Run the agent

From the project root:

```bash
bash run_agent.sh
```

Or directly:

```bash
python src/app.py --tui
```

Legacy Rich REPL:

```bash
python src/app.py --repl
```

Type your question and press Enter. Type `exit` or `quit` to leave.

## Development checks

Run the automated checks from the activated environment:

```bash
pytest
python -m compileall -q src
```
