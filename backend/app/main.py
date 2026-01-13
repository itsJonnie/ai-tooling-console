import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import List

from app.models import CreateRunRequest, Run, LogEntry, MetricPoint
from app.storage import storage
from app.runner import simulate_training

app = FastAPI(title="AI Tooling Console API", version="0.1.0")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/runs", response_model=Run, status_code=201)
def create_run(request: CreateRunRequest, background_tasks: BackgroundTasks) -> Run:
    """Create a new training run."""
    run = storage.create_run(
        dataset_name=request.dataset_name,
        model_name=request.model_name,
        epochs=request.epochs,
        notes=request.notes
    )
    # Start training simulation in background
    background_tasks.add_task(simulate_training, run.id)
    return run


@app.get("/runs", response_model=List[Run])
def list_runs() -> List[Run]:
    """List all runs."""
    return storage.list_runs()


@app.get("/runs/{run_id}", response_model=Run)
def get_run(run_id: str) -> Run:
    """Get a single run by ID."""
    run = storage.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/runs/{run_id}/logs", response_model=List[LogEntry])
def get_run_logs(run_id: str) -> List[LogEntry]:
    """Get logs for a specific run."""
    run = storage.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return storage.get_logs(run_id)


@app.get("/runs/{run_id}/metrics", response_model=List[MetricPoint])
def get_run_metrics(run_id: str) -> List[MetricPoint]:
    """Get metrics for a specific run."""
    run = storage.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return storage.get_metrics(run_id)
