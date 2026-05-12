"""
MLflow Data Access Tools (refactored)
------------------------------------
Provide both raw data-access helpers (raw_*) that return JSON-serializable
structures, and LangChain Tool wrappers (@tool) that call the raw helpers.
"""

import os
import json
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
# mlflow is optional for unit tests in minimal environments
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import MlflowException
except Exception:
    mlflow = None
    MlflowClient = None
    class MlflowException(Exception):
        pass
try:
    from langchain.tools import tool
except Exception:
    # Provide a no-op decorator when langchain isn't available (useful for tests / minimal installs)
    def tool(*args, **kwargs):
        def _decorator(f):
            return f
        return _decorator
from . import schemas

# Keep MLflow logs quieter by default
# os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "WARNING")
logging.getLogger("mlflow").setLevel(logging.WARNING)

# Load mlruns_dir from global config
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config.json'))
try:
    with open(CONFIG_PATH, 'r') as f:
        _config = json.load(f)
    mlruns_dir = _config.get('mlflow', {}).get('mlruns_dir', 'data/mlruns')
except Exception:
    mlruns_dir = 'data/mlruns'

print(f"Using mlruns directory: {mlruns_dir}")

mlflow.set_tracking_uri(mlruns_dir)
client = MlflowClient()


def _iso_from_epoch_ms(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    try:
        import datetime
        return datetime.datetime.utcfromtimestamp(ms / 1000.0).isoformat() + 'Z'
    except Exception:
        return None


# ---------- Raw helpers (callable directly & easier to unit test) ----------

def raw_list_experiments(include_deleted: bool = False, max_results: int = 100) -> List[Dict[str, Any]]:
    """List experiments with richer summary fields."""
    try:
        exps = client.search_experiments()
    except MlflowException as e:
        logging.error("Error listing experiments: %s", e)
        raise

    out = []
    for exp in exps:
        if not include_deleted and getattr(exp, 'lifecycle_stage', None) == 'deleted':
            continue
        out.append({
            'experiment_id': exp.experiment_id,
            'name': exp.name,
            'artifact_location': getattr(exp, 'artifact_location', None),
            'lifecycle_stage': getattr(exp, 'lifecycle_stage', None),
            'tags': getattr(exp, 'tags', {}) or {}
        })
    return out


def raw_list_runs(
    experiment_id: str,
    status: Optional[List[str]] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    order_by: Optional[str] = None,
    max_results: int = 100,
    include_metrics: bool = False,
) -> List[Dict[str, Any]]:
    """Return summarized runs for given experiments.

    Note: MLflowClient.search_runs accepts experiment_ids and order_by; more complex
    filtering can be added later.
    """
    if experiment_id.lower() in ["all","*"]:
        raise ValueError("Listing runs across all experiments is not supported for token economy. Please specify a single experiment ID.")
    try:
        runs = client.search_runs([experiment_id], order_by=[order_by] if order_by else None, max_results=max_results)
    except MlflowException as e:
        logging.error("Error listing runs: %s", e)
        raise

    out = []
    for run in runs:
        run_info = {
            'run_id': run.info.run_id,
            'run_name': getattr(run.info, 'run_name', None),
            'status': getattr(run.info, 'status', None),
            'start_time_iso': _iso_from_epoch_ms(getattr(run.info, 'start_time', None)),
            'end_time_iso': _iso_from_epoch_ms(getattr(run.info, 'end_time', None)),
        }
        if include_metrics:
            metrics = dict(getattr(run, 'data', SimpleNamespace()).metrics) if hasattr(run, 'data') else {}
            params = dict(getattr(run, 'data', SimpleNamespace()).params) if hasattr(run, 'data') else {}
            run_info['metrics_preview'] = {k: metrics[k] for i, k in enumerate(metrics) if i < 5}
            run_info['params_preview'] = {k: params[k] for i, k in enumerate(params) if i < 10}
        out.append(run_info)
    return out


# New: Count runs per experiment ID
def raw_count_runs_per_experiment(experiment_ids: List[str]) -> Dict[str, int]:
    """Return a dict of experiment_id -> number of runs."""
    counts = {}
    for exp_id in experiment_ids:
        try:
            runs = client.search_runs([exp_id], max_results=50000)
            counts[exp_id] = len(runs)
        except MlflowException as e:
            logging.error(f"Error counting runs for experiment {exp_id}: {e}")
            counts[exp_id] = -1
    return counts


def raw_get_run_metrics(run_id: str) -> Dict[str, float]:
    """Return a dict of metric_name -> latest_value for the run."""
    try:
        run = client.get_run(run_id)
    except MlflowException as e:
        logging.error("Error getting run metrics for %s: %s", run_id, e)
        raise
    return dict(run.data.metrics)


def raw_get_run_params(run_id: str) -> Dict[str, str]:
    """Return a dict of param_name -> value for the run."""
    try:
        run = client.get_run(run_id)
    except MlflowException as e:
        logging.error("Error getting run params for %s: %s", run_id, e)
        raise
    return dict(run.data.params)


def raw_find_best_runs_by_metric(
    experiment_ids: List[str],
    metric: str,
    mode: str = 'max',
    top_k: int = 1,
    filter_params: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Return top-k runs ordered by metric (max or min)."""

    if experiment_ids.lower() in ["all","*"]:
        raise ValueError("To search across all experiments, use a list of all experiment IDs obtained from list_experiments.")
    order = f"metrics.{metric} DESC" if mode == 'max' else f"metrics.{metric} ASC"
    try:
        runs = client.search_runs(experiment_ids, order_by=[order], max_results=top_k)
    except MlflowException as e:
        logging.error("Error searching runs for best metric %s: %s", metric, e)
        raise

    out = []
    for run in runs:
        out.append({
            'run_id': run.info.run_id,
            'run_name': getattr(run.info, 'run_name', None),
            'metrics': dict(getattr(run, 'data', SimpleNamespace()).metrics),
            'params': dict(getattr(run, 'data', SimpleNamespace()).params),
            'artifact_uri': getattr(run.info, 'artifact_uri', None) if hasattr(run.info, 'artifact_uri') else None,
        })
    return out


def raw_check_experiment_generalization(
    experiment_id_or_name: str,
    train_metric: str = 'train_loss',
    test_metric: str = 'test_loss',
    threshold_abs: Optional[float] = None,
    threshold_rel: Optional[float] = 0.2,
) -> Dict[str, Any]:
    """Check runs for generalization failures. Returns a simple report dict."""
    # Resolve experiment
    try:
        exp = None
        # Try by id
        try:
            e = client.get_experiment(experiment_id_or_name)
            if e:
                exp = e
        except Exception:
            # ignore: get_experiment may raise if not found
            pass
        if exp is None:
            # Try by name
            exp = client.get_experiment_by_name(experiment_id_or_name)
        if not exp:
            return {'passed': True, 'failing_runs': [], 'reason': 'experiment_not_found'}
    except MlflowException as e:
        logging.error("Error resolving experiment %s: %s", experiment_id_or_name, e)
        raise

    experiment_id = getattr(exp, 'experiment_id', None)
    runs = raw_list_runs(experiment_id=experiment_id, max_results=1000)

    failing = []
    for r in runs:
        run_id = r['run_id']
        metrics = raw_get_run_metrics(run_id)
        train = metrics.get(train_metric)
        test = metrics.get(test_metric)
        if train is None or test is None:
            continue
        diff = test - train
        diff_pct = (diff / train) if train != 0 else float('inf')
        if threshold_abs is not None:
            failed = diff > threshold_abs
        else:
            failed = diff_pct > threshold_rel
        if failed:
            failing.append({'run_id': run_id, 'train': train, 'test': test, 'diff': diff, 'diff_pct': diff_pct})

    return {'passed': len(failing) == 0, 'failing_runs': failing}


# ---------- Tool wrappers (kept lightweight) ----------

@tool(description="List MLflow experiments (tool wrapper).", args_schema=schemas.ListExperimentsParams)
def list_experiments_tool(include_deleted: bool = False, max_results: int = 100):
    result = raw_list_experiments(include_deleted=include_deleted, max_results=max_results)
    try:
        return json.dumps(result, default=str)
    except Exception:
        return str(result)



# Update tool wrapper for new signature
@tool(description="List MLflow runs for a single experiment (tool wrapper).", args_schema=schemas.ListRunsParams)
def list_runs_tool(experiment_id: str, status: Optional[List[str]] = None, start_time: Optional[int] = None, end_time: Optional[int] = None, order_by: Optional[str] = None, max_results: int = 100):
    result = raw_list_runs(experiment_id=experiment_id, status=status, start_time=start_time, end_time=end_time, order_by=order_by, max_results=max_results)
    try:
        return json.dumps(result, default=str)
    except Exception:
        return str(result)


# New tool wrapper for counting runs
@tool(description="Count MLflow runs per experiment (tool wrapper).", args_schema=schemas.CountRunsPerExperimentParams)
def count_runs_per_experiment_tool(experiment_ids: List[str]):
    result = raw_count_runs_per_experiment(experiment_ids)
    try:
        return json.dumps(result, default=str)
    except Exception:
        return str(result)


@tool(description="Get run metrics (tool wrapper).", args_schema=schemas.GetRunMetricsParams)
def get_run_metrics_tool(run_id: str):
    result = raw_get_run_metrics(run_id)
    try:
        return json.dumps(result, default=str)
    except Exception:
        return str(result)


@tool(description="Get run params (tool wrapper).", args_schema=schemas.GetRunParamsParams)
def get_run_params_tool(run_id: str):
    result = raw_get_run_params(run_id)
    try:
        return json.dumps(result, default=str)
    except Exception:
        return str(result)


@tool(description="Find top runs by metric (tool wrapper).", args_schema=schemas.FindBestRunByMetricParams)
def find_best_runs_by_metric_tool(experiment_ids: List[str], metric: str, mode: str = 'max', top_k: int = 1):
    result = raw_find_best_runs_by_metric(experiment_ids=experiment_ids, metric=metric, mode=mode, top_k=top_k)
    try:
        return json.dumps(result, default=str)
    except Exception:
        return str(result)


@tool(description="Check experiment generalization (tool wrapper).", args_schema=schemas.CheckExperimentGeneralizationParams)
def check_experiment_generalization_tool(experiment_id_or_name: str, train_metric: str = 'train_loss', test_metric: str = 'test_loss', threshold_abs: Optional[float] = None, threshold_rel: Optional[float] = 0.2):
    result = raw_check_experiment_generalization(experiment_id_or_name=experiment_id_or_name, train_metric=train_metric, test_metric=test_metric, threshold_abs=threshold_abs, threshold_rel=threshold_rel)
    try:
        return json.dumps(result, default=str)
    except Exception:
        return str(result)


def get_all_tools():
    """Return all tool wrappers for agent instantiation."""
    return [
        list_experiments_tool,
        list_runs_tool,
        count_runs_per_experiment_tool,
        get_run_metrics_tool,
        get_run_params_tool,
        find_best_runs_by_metric_tool,
        check_experiment_generalization_tool,
    ]
