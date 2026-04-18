from typing import (
    Any, Dict, List, Literal, Optional
)
from pydantic import BaseModel, Field

# Tool argument schemas
class ListExperimentsParams(BaseModel):
    include_deleted: bool = Field(False, description="Include deleted experiments")
    max_results: int = Field(100, description="Maximum number of experiments to return")

class ListRunsParams(BaseModel):
    experiment_ids: List[str] = Field(..., description="IDs of the experiments to list runs for.")
    status: Optional[List[str]] = Field(None, description="Filter by run status, e.g. ['FINISHED']")
    start_time: Optional[int] = Field(None, description="Only runs started after this epoch ms")
    end_time: Optional[int] = Field(None, description="Only runs started before this epoch ms")
    order_by: Optional[str] = Field(None, description="order_by clause for MLflow search_runs")
    max_results: int = Field(100, description="Maximum number of runs to return.")

class GetRunMetricsParams(BaseModel):
    run_id: str = Field(..., description="ID of the run to get metrics for.")

class GetRunParamsParams(BaseModel):
    run_id: str = Field(..., description="ID of the run to get parameters for.")

class FindBestRunByMetricParams(BaseModel):
    experiment_ids: List[str] = Field(..., description="IDs of the experiments to search in.")
    metric: str = Field(..., description="Metric name to optimize.")
    mode: Literal['max', 'min'] = Field('max', description="Whether to maximize or minimize the metric.")
    top_k: int = Field(1, description="Return top-k runs")

class CheckExperimentGeneralizationParams(BaseModel):
    experiment_id_or_name: str = Field(..., description="Experiment id or name to check.")
    train_metric: str = Field('train_loss', description="Train metric name")
    test_metric: str = Field('test_loss', description="Test metric name")
    threshold_abs: Optional[float] = Field(None, description="Absolute threshold for (test - train)")
    threshold_rel: Optional[float] = Field(0.2, description="Relative threshold fraction for test/train difference")

# Response models (simple, JSON-serializable)
class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    artifact_location: Optional[str]
    lifecycle_stage: Optional[str]
    tags: Optional[Dict[str, str]]

class RunSummary(BaseModel):
    run_id: str
    run_name: Optional[str]
    status: Optional[str]
    start_time_iso: Optional[str]
    end_time_iso: Optional[str]
    metrics_preview: Optional[Dict[str, float]]
    params_preview: Optional[Dict[str, str]]

class RunDetail(BaseModel):
    run_id: str
    run_name: Optional[str]
    metrics: Dict[str, float]
    params: Dict[str, str]
    artifact_uri: Optional[str]

class GeneralizationReport(BaseModel):
    passed: bool
    failing_runs: List[Dict[str, Any]]
