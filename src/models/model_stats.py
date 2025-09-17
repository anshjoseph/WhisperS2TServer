from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class ProcessingStats(BaseModel):
    batches_processed: int
    tasks_processed: int
    errors: int
    processing_times: List[float]
    processing: bool
    last_used: Optional[datetime] = None  # default None if not set
