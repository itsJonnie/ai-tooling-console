import asyncio
import random
from datetime import datetime
from typing import List

from app.models import Run, RunState, LogEntry, MetricPoint
from app.storage import storage

# Failure root causes
FAILURE_ROOT_CAUSES = [
    "data schema mismatch",
    "out of memory",
    "gradient explosion",
    "learning rate too high",
    "invalid input dimensions",
    "model architecture incompatible",
    "training data corrupted",
    "checkpoint loading failed",
    "loss function divergence",
    "optimizer state corruption"
]


async def simulate_training(run_id: str) -> None:
    """
    Simulate a training run in the background.
    Updates run state, generates logs and metrics every second.
    Has ~10% chance of failure.
    """
    run = storage.get_run(run_id)
    if not run:
        return
    
    # Transition to RUNNING
    run.state = RunState.RUNNING
    run.started_at = datetime.now()
    storage.update_run(run)
    
    # Initial values
    initial_loss = 1.0
    initial_accuracy = 0.5
    target_loss = 0.1
    target_accuracy = 0.95
    
    # Check for failure (10% chance)
    should_fail = random.random() < 0.10
    
    for epoch in range(1, run.epochs + 1):
        # Calculate progress (0.0 to 1.0)
        progress = epoch / run.epochs
        
        # Interpolate loss (decreasing) and accuracy (increasing)
        current_loss = initial_loss - (initial_loss - target_loss) * progress
        current_accuracy = initial_accuracy + (target_accuracy - initial_accuracy) * progress
        
        # Add some randomness to make it realistic
        current_loss += random.uniform(-0.02, 0.02)
        current_accuracy += random.uniform(-0.01, 0.01)
        
        # Clamp values
        current_loss = max(target_loss, min(initial_loss, current_loss))
        current_accuracy = max(initial_accuracy, min(target_accuracy, current_accuracy))
        
        # Create log entry
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level="INFO",
            message=f"Epoch {epoch}/{run.epochs}: loss={current_loss:.4f}, accuracy={current_accuracy:.4f}"
        )
        storage.append_log(run_id, log_entry)
        
        # Create metric point
        metric = MetricPoint(
            timestamp=datetime.now(),
            epoch=epoch,
            loss=current_loss,
            accuracy=current_accuracy
        )
        storage.append_metric(run_id, metric)
        
        # Check for failure (only check once, around epoch 3-5 to make it realistic)
        # Only fail if we have enough epochs (at least 3)
        failure_epoch = random.randint(3, min(5, run.epochs)) if run.epochs >= 3 else None
        if should_fail and failure_epoch is not None and epoch == failure_epoch:
            run.state = RunState.FAILED
            run.finished_at = datetime.now()
            run.error_message = f"Training failed at epoch {epoch}"
            run.root_cause = random.choice(FAILURE_ROOT_CAUSES)
            storage.update_run(run)
            
            # Add error log
            error_log = LogEntry(
                timestamp=datetime.now(),
                level="ERROR",
                message=f"Training failed: {run.root_cause}"
            )
            storage.append_log(run_id, error_log)
            return
        
        # Wait 1 second before next epoch
        await asyncio.sleep(1)
    
    # If we made it through all epochs, mark as succeeded
    if run.state == RunState.RUNNING:
        run.state = RunState.SUCCEEDED
        run.finished_at = datetime.now()
        storage.update_run(run)
        
        # Add success log
        success_log = LogEntry(
            timestamp=datetime.now(),
            level="INFO",
            message=f"Training completed successfully after {run.epochs} epochs"
        )
        storage.append_log(run_id, success_log)


