from typing import Dict, List, Optional
from datetime import datetime
import uuid

from app.models import Run, RunState, LogEntry, MetricPoint


class Storage:
    """In-memory storage for runs, logs, and metrics."""
    
    def __init__(self):
        self.runs: Dict[str, Run] = {}
        self.logs: Dict[str, List[LogEntry]] = {}
        self.metrics: Dict[str, List[MetricPoint]] = {}
    
    def create_run(
        self,
        dataset_name: str,
        model_name: str,
        epochs: int,
        notes: Optional[str] = None
    ) -> Run:
        """Create a new run and return it."""
        run_id = str(uuid.uuid4())
        run = Run(
            id=run_id,
            dataset_name=dataset_name,
            model_name=model_name,
            epochs=epochs,
            notes=notes,
            state=RunState.QUEUED,
            created_at=datetime.now()
        )
        self.runs[run_id] = run
        self.logs[run_id] = []
        self.metrics[run_id] = []
        return run
    
    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID."""
        return self.runs.get(run_id)
    
    def list_runs(self) -> List[Run]:
        """List all runs."""
        return list(self.runs.values())
    
    def update_run(self, run: Run) -> None:
        """Update a run."""
        if run.id in self.runs:
            self.runs[run.id] = run
    
    def append_log(self, run_id: str, log_entry: LogEntry) -> None:
        """Append a log entry to a run."""
        if run_id not in self.logs:
            self.logs[run_id] = []
        self.logs[run_id].append(log_entry)
    
    def get_logs(self, run_id: str) -> List[LogEntry]:
        """Get all logs for a run."""
        return self.logs.get(run_id, [])
    
    def append_metric(self, run_id: str, metric: MetricPoint) -> None:
        """Append a metric point to a run."""
        if run_id not in self.metrics:
            self.metrics[run_id] = []
        self.metrics[run_id].append(metric)
    
    def get_metrics(self, run_id: str) -> List[MetricPoint]:
        """Get all metrics for a run."""
        return self.metrics.get(run_id, [])


# Global storage instance
storage = Storage()
