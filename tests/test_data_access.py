from types import SimpleNamespace
import pytest
from src.mlflow_tools import data_access

class FakeRunInfo(SimpleNamespace):
    def __init__(self, run_id='r1', run_name='run1', status='FINISHED', start_time=0, end_time=1, artifact_uri='file:///tmp/artifacts'):
        super().__init__(run_id=run_id, run_name=run_name, status=status, start_time=start_time, end_time=end_time, artifact_uri=artifact_uri)

class FakeRunData(SimpleNamespace):
    def __init__(self, metrics=None, params=None):
        super().__init__(metrics=metrics or {}, params=params or {})

class FakeClient:
    def __init__(self):
        self._exps = [SimpleNamespace(name='exp1', experiment_id='1', artifact_location='file:///tmp', lifecycle_stage='active', tags={})]
    def search_experiments(self, *args, **kwargs):
        return self._exps
    def search_runs(self, experiment_ids, order_by=None, max_results=50):
        info = FakeRunInfo()
        data = FakeRunData(metrics={'accuracy': 0.9, 'train_loss': 0.2, 'test_loss': 0.3}, params={'lr': '0.01'})
        run = SimpleNamespace(info=info, data=data)
        return [run]
    def get_run(self, run_id):
        info = FakeRunInfo(run_id=run_id)
        data = FakeRunData(metrics={'accuracy': 0.9, 'train_loss': 0.2, 'test_loss': 0.3}, params={'lr':'0.01'})
        return SimpleNamespace(info=info, data=data)
    def get_experiment_by_name(self, name):
        if name == 'exp1':
            return SimpleNamespace(experiment_id='1')
        return None


def test_list_experiments(monkeypatch):
    monkeypatch.setattr(data_access, 'client', FakeClient())
    exps = data_access.raw_list_experiments()
    assert isinstance(exps, list)
    assert exps[0]['experiment_id'] == '1'


def test_get_run_metrics(monkeypatch):
    monkeypatch.setattr(data_access, 'client', FakeClient())
    m = data_access.raw_get_run_metrics('r1')
    assert m['accuracy'] == 0.9


def test_check_experiment_generalization(monkeypatch):
    monkeypatch.setattr(data_access, 'client', FakeClient())
    report = data_access.raw_check_experiment_generalization('exp1')
    # In FakeClient, train_loss=0.2 test_loss=0.3 -> 50% worse so should fail
    assert report['passed'] is False
    assert 'r1' in [r['run_id'] for r in report['failing_runs']]

def test_list_runs(monkeypatch):
    monkeypatch.setattr(data_access, 'client', FakeClient())
    runs = data_access.raw_list_runs('1')
    assert isinstance(runs, list)
    assert runs[0]['run_id'] == 'r1'
    # Should not include metrics/params by default
    assert 'metrics_preview' not in runs[0]
    # With metrics
    runs_with_metrics = data_access.raw_list_runs('1', include_metrics=True)
    assert 'metrics_preview' in runs_with_metrics[0]
    assert 'params_preview' in runs_with_metrics[0]


def test_count_runs_per_experiment(monkeypatch):
    monkeypatch.setattr(data_access, 'client', FakeClient())
    counts = data_access.raw_count_runs_per_experiment(['1', '2'])
    assert isinstance(counts, dict)
    assert counts['1'] == 1
    assert counts['2'] == 1  # FakeClient returns 1 run for any exp id
