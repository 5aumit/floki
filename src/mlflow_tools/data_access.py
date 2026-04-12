"""
MLflow Data Access Tools
-----------------------
Provides functions for listing experiments, runs, and retrieving metrics/parameters.
"""

import mlflow
import logging
from mlflow.tracking import MlflowClient
from typing import List, Dict, Any, Optional
from .schemas import (
    ListExperimentsParams,
    ListRunsParams,
    GetRunMetricsParams,
    GetRunParamsParams,
    FindBestRunByMetricParams,
    CheckExperimentGeneralizationParams
)
import os
import json
from langchain.tools import tool



# logging.getLogger("mlflow").setLevel(logging.ERROR)
os.environ["MLFLOW_LOGGING_LEVEL"] = "WARNING"
# Load mlruns_dir from global config
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config.json'))
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    mlruns_dir = config.get('mlflow', {}).get('mlruns_dir', 'data/mlruns')
except Exception:
    mlruns_dir = 'data/mlruns'

print(f"Using mlruns directory: {mlruns_dir}")

mlflow.set_tracking_uri(mlruns_dir)
client = MlflowClient()

@tool(description="List all MLflow experiments.", args_schema=ListExperimentsParams)
def list_experiments() -> List[Dict[str, Any]]:
    """Return a list of all MLflow experiments."""
    # No parameters, but validate for consistency
    ListExperimentsParams()
    experiments = client.search_experiments()
    return [
        {
            'name': exp.name,
            'experiment_id': exp.experiment_id,
            'description': getattr(exp, 'description', None)
        }
        for exp in experiments
    ]

@tool(description="List runs for a given experiment_id.", args_schema=ListRunsParams)
def list_runs(experiment_ids: List[str], max_results: int = 20) -> List[Dict[str, Any]]:
    """Return a list of runs for a given experiment ID."""
    runs = client.search_runs(experiment_ids, max_results=max_results)
    return [
        {
            'run_id': run.info.run_id,
            'run_name': run.info.run_name,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time
        }
        for run in runs
    ]

@tool(description="Get all metrics for a given run_id.", args_schema=GetRunMetricsParams)
def get_run_metrics(run_id: str) -> Dict[str, Any]:
    """Return all metrics for a given run ID."""
    run = client.get_run(run_id)
    return dict(run.data.metrics)

@tool(description="Get all parameters for a given run_id.", args_schema=GetRunParamsParams)
def get_run_params(run_id: str) -> Dict[str, Any]:
    """Return all parameters for a given run ID."""
    run = client.get_run(run_id)
    return dict(run.data.params)

@tool(description="Find the best run for a metric in an experiment.", args_schema=FindBestRunByMetricParams)
def find_best_run_by_metric(experiment_ids: List[str], metric: str, mode: str = 'max') -> Optional[Dict[str, Any]]:
    """Find the run with the best value for a given metric (max or min)."""
    order = f"metrics.{metric} DESC" if mode == 'max' else f"metrics.{metric} ASC"
    runs = client.search_runs(experiment_ids, order_by=[order], max_results=1)
    if runs:
        run = runs[0]
        return {
            'run_id': run.info.run_id,
            'run_name': run.info.run_name,
            'metrics': dict(run.data.metrics),
            'params': dict(run.data.params)
        }
    return None


@tool(description="Check if any runs in an experiment failed to generalize (test metric much worse than train metric).", args_schema=CheckExperimentGeneralizationParams)
def check_experiment_generalization(experiment_name: str, metric: str = "loss", threshold: float = None) -> str:
    """
    Checks if any runs in the given experiment failed to generalize (e.g., test metric much worse than train metric).
    Args:
        experiment_name: Name of the MLflow experiment.
        metric: Metric to compare (default: 'loss').
        threshold: Optional threshold for difference (if None, use heuristic).
    Returns:
        Summary string.
    """
    experiments = list_experiments.invoke()
    exp = next((e for e in experiments if e['name'] == experiment_name), None)
    if not exp:
        return f"Experiment '{experiment_name}' not found."
    runs = list_runs.invoke(exp['experiment_id'], max_results=1000)
    failed_runs = []
    for run in runs:
        metrics = get_run_metrics.invoke(run['run_id'])
        train = metrics.get(f"train_{metric}")
        test = metrics.get(f"test_{metric}")
        if train is not None and test is not None:
            diff = abs(test - train)
            # Heuristic: test metric 20% worse than train metric
            if threshold is not None:
                if (test - train) > threshold:
                    failed_runs.append(run['run_id'])
            elif test > train * 1.2:
                failed_runs.append(run['run_id'])
    if failed_runs:
        return f"Runs that failed to generalize in '{experiment_name}': {failed_runs}"
    else:
        return f"All runs in '{experiment_name}' generalized well."

def get_all_tools():
    """
    Return a list of all tool functions defined in this module.
    Useful for agent instantiation.
    """
    return [
        list_experiments,
        list_runs,
        get_run_metrics,
        get_run_params,
        find_best_run_by_metric,
        check_experiment_generalization
    ]
