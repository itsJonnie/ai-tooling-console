import pytest
import asyncio
from fastapi.testclient import TestClient
from app.main import app
from app.storage import storage
from app.models import RunState

client = TestClient(app)


def test_create_run():
    """Test creating a new run."""
    # Clear storage for clean test
    storage.runs.clear()
    storage.logs.clear()
    storage.metrics.clear()
    
    response = client.post(
        "/runs",
        json={
            "dataset_name": "test_dataset",
            "model_name": "test_model",
            "epochs": 3,
            "notes": "Test run"
        }
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["dataset_name"] == "test_dataset"
    assert data["model_name"] == "test_model"
    assert data["epochs"] == 3
    assert data["notes"] == "Test run"
    assert data["state"] == RunState.QUEUED.value
    assert "id" in data
    assert "created_at" in data


def test_list_runs():
    """Test listing all runs."""
    # Clear storage
    storage.runs.clear()
    storage.logs.clear()
    storage.metrics.clear()
    
    # Create a few runs
    run1 = client.post(
        "/runs",
        json={
            "dataset_name": "dataset1",
            "model_name": "model1",
            "epochs": 2
        }
    ).json()
    
    run2 = client.post(
        "/runs",
        json={
            "dataset_name": "dataset2",
            "model_name": "model2",
            "epochs": 2
        }
    ).json()
    
    # List runs
    response = client.get("/runs")
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 2
    assert any(r["id"] == run1["id"] for r in runs)
    assert any(r["id"] == run2["id"] for r in runs)


def test_get_run():
    """Test getting a single run."""
    # Clear storage
    storage.runs.clear()
    storage.logs.clear()
    storage.metrics.clear()
    
    # Create a run
    create_response = client.post(
        "/runs",
        json={
            "dataset_name": "test_dataset",
            "model_name": "test_model",
            "epochs": 1
        }
    )
    run_id = create_response.json()["id"]
    
    # Get the run immediately (should work regardless of state)
    response = client.get(f"/runs/{run_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == run_id
    assert data["dataset_name"] == "test_dataset"


def test_get_run_not_found():
    """Test getting a non-existent run returns 404."""
    response = client.get("/runs/non-existent-id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_run_state_transitions():
    """Test that run state transitions from QUEUED to RUNNING to SUCCEEDED/FAILED."""
    # Clear storage
    storage.runs.clear()
    storage.logs.clear()
    storage.metrics.clear()
    
    # Create a run with 1 epoch (quick test)
    response = client.post(
        "/runs",
        json={
            "dataset_name": "test_dataset",
            "model_name": "test_model",
            "epochs": 1
        }
    )
    run_id = response.json()["id"]
    
    # Wait for training to complete (1 epoch * 1 second = 1 second, plus buffer)
    import time
    time.sleep(2.5)
    
    # Should be SUCCEEDED or FAILED
    run = storage.get_run(run_id)
    assert run is not None
    assert run.state in [RunState.SUCCEEDED, RunState.FAILED]
    if run.state == RunState.FAILED:
        assert run.error_message is not None
        assert run.root_cause is not None
    else:
        # If succeeded, should have finished_at
        assert run.finished_at is not None


def test_get_run_logs():
    """Test getting logs for a run."""
    # Clear storage
    storage.runs.clear()
    storage.logs.clear()
    storage.metrics.clear()
    
    # Create a run
    response = client.post(
        "/runs",
        json={
            "dataset_name": "test_dataset",
            "model_name": "test_model",
            "epochs": 2
        }
    )
    run_id = response.json()["id"]
    
    # Wait for some logs to be generated
    import time
    time.sleep(2)
    
    # Get logs
    logs_response = client.get(f"/runs/{run_id}/logs")
    assert logs_response.status_code == 200
    logs = logs_response.json()
    assert isinstance(logs, list)
    # Should have at least one log entry after training starts
    if len(logs) > 0:
        log = logs[0]
        assert "timestamp" in log
        assert "level" in log
        assert "message" in log


def test_get_run_metrics():
    """Test getting metrics for a run."""
    # Clear storage
    storage.runs.clear()
    storage.logs.clear()
    storage.metrics.clear()
    
    # Create a run
    response = client.post(
        "/runs",
        json={
            "dataset_name": "test_dataset",
            "model_name": "test_model",
            "epochs": 2
        }
    )
    run_id = response.json()["id"]
    
    # Wait for some metrics to be generated
    import time
    time.sleep(2)
    
    # Get metrics
    metrics_response = client.get(f"/runs/{run_id}/metrics")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert isinstance(metrics, list)
    # Should have at least one metric after training starts
    if len(metrics) > 0:
        metric = metrics[0]
        assert "timestamp" in metric
        assert "epoch" in metric
        assert "loss" in metric
        assert "accuracy" in metric
        assert isinstance(metric["loss"], (int, float))
        assert isinstance(metric["accuracy"], (int, float))


def test_get_logs_not_found():
    """Test getting logs for non-existent run returns 404."""
    response = client.get("/runs/non-existent-id/logs")
    assert response.status_code == 404


def test_get_metrics_not_found():
    """Test getting metrics for non-existent run returns 404."""
    response = client.get("/runs/non-existent-id/metrics")
    assert response.status_code == 404


def test_create_run_validation():
    """Test that creating a run with invalid data returns 422."""
    # Clear storage
    storage.runs.clear()
    storage.logs.clear()
    storage.metrics.clear()
    
    # Missing required fields
    response = client.post("/runs", json={})
    assert response.status_code == 422
    
    # Invalid epochs (must be > 0)
    response = client.post(
        "/runs",
        json={
            "dataset_name": "test",
            "model_name": "test",
            "epochs": 0
        }
    )
    assert response.status_code == 422
