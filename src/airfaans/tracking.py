"""Optional MLflow experiment tracking."""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def tracked_run(run_name: str, parameters: dict[str, object]):
    try:
        import mlflow
    except ImportError:
        yield None
        return
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(parameters)
        yield mlflow
