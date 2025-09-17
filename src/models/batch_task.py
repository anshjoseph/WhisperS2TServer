from pydantic import BaseModel, Field, computed_field
from typing import List, Optional, Callable, Any
from enum import Enum
import time


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStats(BaseModel):
    """Basic processing statistics."""
    start_time: float
    end_time: Optional[float] = None
    processing_time: Optional[float] = None
    file_size: Optional[int] = None


class PerformanceMetrics(BaseModel):
    """Essential performance metrics."""
    total_processed: int = 0
    total_failed: int = 0
    avg_processing_time: float = 0.0
    is_busy: bool = False
    throughput_per_hour: float = 0.0


class ProcessorStats(BaseModel):
    """Processor statistics class."""
    is_running: bool
    processed_batches: int
    failed_batches: int
    input_queue_size: int
    output_queue_size: int
    uptime: float
    performance_metrics: PerformanceMetrics


class Task(BaseModel):
    task_id: str
    file_path: str
    task: str
    lang: str
    prompt: Optional[str] = None
    output: Optional[List[dict]] = None
    status: TaskStatus = TaskStatus.PENDING
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    processing_started_at: Optional[float] = None
    processed_at: Optional[float] = None
    stats: Optional[ProcessingStats] = None

    @property
    def processing_time(self) -> Optional[float]:
        """Time spent in actual processing."""
        if self.processing_started_at and self.processed_at:
            return self.processed_at - self.processing_started_at
        return None


class BatchTask(BaseModel):
    batch_id: str
    start_time: float
    end_time: Optional[float] = None
    tasks: List[Task]
    status: TaskStatus = TaskStatus.PENDING
    completed_tasks: int = 0
    failed_tasks: int = 0
    processing_started_at: Optional[float] = None

    @computed_field
    @property
    def total_tasks(self) -> int:
        """Total number of tasks in the batch."""
        return len(self.tasks)

    @property
    def progress(self) -> float:
        """Returns processing progress as a percentage."""
        return (self.completed_tasks + self.failed_tasks) / self.total_tasks * 100

    @property
    def is_complete(self) -> bool:
        """Returns True if all tasks are completed or failed."""
        return (self.completed_tasks + self.failed_tasks) == self.total_tasks

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100

    