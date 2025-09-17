from models.batch_task import BatchTask, Task
from typing import List, Dict, Optional
from services.whisper_model_pool import WhisperModelPool
from threading import Thread, Event, Lock, Condition
from uuid import uuid4
import time
import asyncio
from queue import Queue, Empty
from config import get_config
from utils.log import get_configure_logger


logger = get_configure_logger(__file__)


class ThreadSafeTaskStorage:
    """Thread-safe storage for completed tasks with notification support."""
    
    def __init__(self):
        self.__tasks: Dict[str, Task] = {}
        self.__lock = Lock()
        self.__condition = Condition(self.__lock)
        self.__completed_task_ids = set()

    def set(self, task: Task):
        """Store a completed task and notify waiting threads."""
        with self.__condition:
            self.__tasks[task.task_id] = task
            self.__completed_task_ids.add(task.task_id)
            self.__condition.notify_all()
            logger.debug(f"Task {task.task_id} stored and notifications sent")

    def get(self, task_id: str) -> Optional[Task]:
        """Get a task by ID if it exists."""
        with self.__lock:
            return self.__tasks.get(task_id)

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[Task]:
        """Wait for a specific task to complete with optional timeout."""
        with self.__condition:
            # Check if task is already completed
            if task_id in self.__completed_task_ids:
                return self.__tasks.get(task_id)
            
            # Wait for task completion
            if self.__condition.wait_for(
                lambda: task_id in self.__completed_task_ids, 
                timeout=timeout
            ):
                return self.__tasks.get(task_id)
            return None

    def remove(self, task_id: str):
        """Remove a task from storage."""
        with self.__lock:
            self.__tasks.pop(task_id, None)
            self.__completed_task_ids.discard(task_id)

    def cleanup_old_tasks(self, max_age_seconds: float = 3600):
        """Clean up old completed tasks to prevent memory leaks."""
        current_time = time.time()
        removed_count = 0
        
        with self.__lock:
            tasks_to_remove = []
            for task_id, task in self.__tasks.items():
                # Assuming tasks have a completion_time or using current logic
                task_age = current_time - getattr(task, 'completion_time', current_time)
                if task_age > max_age_seconds:
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                self.__tasks.pop(task_id, None)
                self.__completed_task_ids.discard(task_id)
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old completed tasks")
        
        return removed_count


class TaskManager:
    """
    Manages task batching and processing through WhisperModelPool.
    Handles async task submission and result retrieval.
    """
    
    def __init__(self):
        self.__pool = WhisperModelPool(self._batch_completion_hook)
        self.__completed_tasks = ThreadSafeTaskStorage()
        self.__pending_queue = Queue()
        self.__batching_thread: Optional[Thread] = None
        self.__cleanup_thread: Optional[Thread] = None
        self.__stop_event = Event()
        self.__is_running = False
        self.__config = get_config()
        
        # Statistics
        self.__total_tasks_submitted = 0
        self.__total_batches_created = 0
        
        logger.info(f"TaskManager initialized with batch size: {self.__config.file_batch_size}")

    def __batching_task(self):
        """Background thread that groups tasks into batches and submits them to the pool."""
        logger.info("Batching thread started")
        batch_timeout = 0.3  # Max time to wait before submitting partial batch
        
        while not self.__stop_event.is_set():
            try:
                tasks:List[Task] = []
                batch_start_time = time.time()
                
                # Collect tasks for batching
                while (len(tasks) < self.__config.file_batch_size and 
                       (time.time() - batch_start_time) < batch_timeout and
                       not self.__stop_event.is_set()):
                    
                    try:
                        # Use timeout to avoid blocking indefinitely
                        task:Task = self.__pending_queue.get(timeout=0.1)
                       
                        tasks.append(task)
                        logger.debug(f"Added task {task.task_id} to batch (batch size: {len(tasks)})")
                    except Empty:
                        continue
                
                # Submit batch if we have tasks
                if tasks:
                    batch_id = str(uuid4())
                    # for i in range(len(tasks)):
                    #     tasks[i].metadate
                    batch = BatchTask(
                        batch_id=batch_id,
                        start_time=time.time(),
                        tasks=tasks
                    )
                    
                    logger.info(f"Submitting batch {batch_id} with {len(tasks)} tasks")
                    
                    if self.__pool.add_batch(batch):
                        self.__total_batches_created += 1
                        logger.debug(f"Batch {batch_id} successfully submitted to pool")
                    else:
                        logger.error(f"Failed to submit batch {batch_id} to pool")
                        # TODO: Handle batch submission failure (retry, dead letter queue, etc.)
                
            except Exception as e:
                logger.error(f"Error in batching thread: {str(e)}")
                time.sleep(1)  # Brief pause before continuing
        
        logger.info("Batching thread stopped")

    def __cleanup_task(self):
        """Background thread for periodic cleanup of old completed tasks."""
        logger.info("Cleanup thread started")
        cleanup_interval = 300  # 5 minutes
        
        while not self.__stop_event.is_set():
            try:
                self.__completed_tasks.cleanup_old_tasks()
                self.__stop_event.wait(cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup thread: {str(e)}")
                self.__stop_event.wait(60)  # Wait 1 minute before retry
        
        logger.info("Cleanup thread stopped")

    def _batch_completion_hook(self, batch: BatchTask):
        """Hook called when a batch completes processing."""
        logger.info(f"Batch {batch.batch_id} completed with {batch.completed_tasks} successful tasks")
        
        try:
            # Store all completed tasks
            for task in batch.tasks:
                self.__completed_tasks.set(task)
                logger.debug(f"Task {task.task_id} marked as completed")
            
        except Exception as e:
            logger.error(f"Error processing batch completion: {str(e)}")

    async def submit_task(self, task: Task) -> Task:
        """
        Submit a task for processing and wait for completion.
        
        Args:
            task: The task to process
            
        Returns:
            The completed task with results
            
        Raises:
            RuntimeError: If TaskManager is not running
            TimeoutError: If task doesn't complete within reasonable time
        """
        if not self.__is_running:
            raise RuntimeError("TaskManager is not running. Call start() first.")
        
        logger.info(f"Submitting task {task.task_id}")
        
        # Add task to pending queue
        self.__pending_queue.put(task)
        self.__total_tasks_submitted += 1
        
        # Wait for task completion with timeout
        timeout_seconds = 300  # 5 minutes timeout
        completed_task = self.__completed_tasks.wait_for_task(task.task_id, timeout_seconds)
        
        if completed_task is None:
            logger.error(f"Task {task.task_id} timed out after {timeout_seconds} seconds")
            raise TimeoutError(f"Task {task.task_id} did not complete within {timeout_seconds} seconds")
        
        logger.info(f"Task {task.task_id} completed successfully")
        
        # Clean up the task from storage to prevent memory leaks
        self.__completed_tasks.remove(task.task_id)
        
        return completed_task

    def start(self) -> bool:
        """Start the TaskManager and its background threads."""
        if self.__is_running:
            logger.warning("TaskManager is already running")
            return True
        
        logger.info("Starting TaskManager")
        
        try:
            # Start the model pool
            if not self.__pool.start():
                logger.error("Failed to start WhisperModelPool")
                return False
            
            # Reset stop event
            self.__stop_event.clear()
            
            # Start background threads
            self.__batching_thread = Thread(target=self.__batching_task, name="TaskManager-Batching")
            self.__batching_thread.daemon = True
            self.__batching_thread.start()
            
            self.__cleanup_thread = Thread(target=self.__cleanup_task, name="TaskManager-Cleanup")
            self.__cleanup_thread.daemon = True
            self.__cleanup_thread.start()
            
            self.__is_running = True
            logger.info("TaskManager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start TaskManager: {str(e)}")
            self.stop()  # Clean up any partial initialization
            return False

    def stop(self, timeout: float = 30.0) -> bool:
        """Stop the TaskManager and all background threads."""
        if not self.__is_running:
            logger.warning("TaskManager is not running")
            return True
        
        logger.info("Stopping TaskManager")
        
        try:
            # Signal threads to stop
            self.__stop_event.set()
            
            # Wait for batching thread to finish
            if self.__batching_thread and self.__batching_thread.is_alive():
                self.__batching_thread.join(timeout=timeout/2)
                if self.__batching_thread.is_alive():
                    logger.warning("Batching thread did not stop cleanly")
            
            # Wait for cleanup thread to finish
            if self.__cleanup_thread and self.__cleanup_thread.is_alive():
                self.__cleanup_thread.join(timeout=timeout/4)
                if self.__cleanup_thread.is_alive():
                    logger.warning("Cleanup thread did not stop cleanly")
            
            # Stop the model pool
            pool_stopped = self.__pool.stop(timeout=timeout/2)
            if not pool_stopped:
                logger.warning("Model pool did not stop cleanly")
            
            self.__is_running = False
            logger.info("TaskManager stopped")
            return pool_stopped
            
        except Exception as e:
            logger.error(f"Error stopping TaskManager: {str(e)}")
            self.__is_running = False
            return False

    def get_stats(self) -> Dict:
        """Get comprehensive TaskManager statistics."""
        pool_stats = self.__pool.get_pool_stats()
        
        return {
            "is_running": self.__is_running,
            "total_tasks_submitted": self.__total_tasks_submitted,
            "total_batches_created": self.__total_batches_created,
            "pending_queue_size": self.__pending_queue.qsize(),
            "pool_stats": pool_stats
        }

    def is_running(self) -> bool:
        """Check if the TaskManager is currently running."""
        return self.__is_running