from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from .models import Run, RunState, LogEntry, MetricPoint


@dataclass
class InMemoryStore:
    runs: Dict[str, Run] = field(default_factory=dict)
    logs: Dict[str, List[LogEntry]] = field(default_factory=dict)
    metrics: Dict[str, List[MetricPoint]] = field(default_factory=dict)

    def create_run(
        self,
        dataset_name: str,
        model_name: str,
        epochs: int,
        notes: Optional[str] = None
    ) -> Run:
        run_id = str(uuid4())
        run = Run(
            id=run_id,
            dataset_name=dataset_name,
            model_name=model_name,
            epochs=epochs,
            notes=notes,
            created_at=datetime.now(timezone.utc),
            state=RunState.QUEUED,
        )
        self.runs[run_id] = run
        self.logs[run_id] = []
        self.metrics[run_id] = []
        return run

    def list_runs(self) -> List[Run]:
        return sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)

    def get_run(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def update_run(self, run: Run) -> None:
        """Update a run in storage."""
        if run.id in self.runs:
            self.runs[run.id] = run

    def append_log(self, run_id: str, log_entry: LogEntry) -> None:
        if run_id not in self.logs:
            self.logs[run_id] = []
        self.logs[run_id].append(log_entry)

    def append_metric(self, run_id: str, metric: MetricPoint) -> None:
        if run_id not in self.metrics:
            self.metrics[run_id] = []
        self.metrics[run_id].append(metric)

    def get_logs(self, run_id: str) -> List[LogEntry]:
        """Get logs for a run, newest first."""
        items = self.logs.get(run_id, [])
        return list(reversed(items))

    def get_metrics(self, run_id: str) -> List[MetricPoint]:
        """Get metrics for a run, newest first."""
        items = self.metrics.get(run_id, [])
        return list(reversed(items))


storage = InMemoryStore()
