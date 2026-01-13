from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RunState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class CreateRunRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset")
    model_name: str = Field(..., description="Name of the model")
    epochs: int = Field(..., gt=0, description="Number of training epochs")
    notes: Optional[str] = Field(None, description="Optional notes about the run")


class Run(BaseModel):
    id: str
    dataset_name: str
    model_name: str
    epochs: int
    notes: Optional[str] = None
    state: RunState
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None
    root_cause: Optional[str] = None


class LogEntry(BaseModel):
    timestamp: datetime
    level: str = Field(..., description="Log level (INFO, WARNING, ERROR)")
    message: str = Field(..., description="Log message")


class MetricPoint(BaseModel):
    timestamp: datetime
    epoch: int
    loss: float = Field(..., description="Training loss (decreasing)")
    accuracy: float = Field(..., description="Training accuracy (increasing)")
