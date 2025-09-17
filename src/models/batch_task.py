from pydantic import BaseModel
from typing import List
from typing import Optional

class Task(BaseModel):
    task_id :str
    file_path : str
    task : str
    lang : str
    prompt: Optional[str] = None
    output : List[dict] = None

class BatchTask(BaseModel):
    batch_id : str
    start_time : float
    end_time : float
    tasks : List[Task]
    