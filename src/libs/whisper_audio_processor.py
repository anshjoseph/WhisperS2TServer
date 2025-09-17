import whisper_s2t
import time
import traceback
import psutil
import gc
import os
from collections import deque
from multiprocessing import Process, Queue, Value
from threading import Thread, Event, Lock
from config import get_config
from utils.log import get_configure_logger
from pydantic import BaseModel, Field, computed_field
from typing import List, Optional, Callable, Any, Dict
from enum import Enum
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import mean, median


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingStats:
    """Detailed processing statistics."""
    start_time: float
    end_time: Optional[float] = None
    queue_time: Optional[float] = None  # Time spent in queue
    processing_time: Optional[float] = None  # Actual processing time
    file_size: Optional[int] = None  # File size in bytes
    audio_duration: Optional[float] = None  # Audio duration in seconds


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    total_processed: int = 0
    total_failed: int = 0
    total_queued: int = 0
    avg_processing_time: float = 0.0
    median_processing_time: float = 0.0
    avg_queue_time: float = 0.0
    throughput_per_minute: float = 0.0
    throughput_per_hour: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    is_busy: bool = False
    utilization_percentage: float = 0.0
    processing_times: List[float] = None
    queue_times: List[float] = None
    
    def __post_init__(self):
        if self.processing_times is None:
            self.processing_times = []
        if self.queue_times is None:
            self.queue_times = []


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
    queued_at: Optional[float] = None
    processing_started_at: Optional[float] = None
    processed_at: Optional[float] = None
    stats: Optional[ProcessingStats] = None

    @property
    def queue_time(self) -> Optional[float]:
        """Time spent in queue before processing."""
        if self.queued_at and self.processing_started_at:
            return self.processing_started_at - self.queued_at
        return None

    @property
    def processing_time(self) -> Optional[float]:
        """Time spent in actual processing."""
        if self.processing_started_at and self.processed_at:
            return self.processed_at - self.processing_started_at
        return None

    @property
    def total_time(self) -> Optional[float]:
        """Total time from creation to completion."""
        if self.processed_at:
            return self.processed_at - self.created_at
        return None


class BatchTask(BaseModel):
    batch_id: str
    start_time: float
    end_time: Optional[float] = None
    tasks: List[Task]
    status: TaskStatus = TaskStatus.PENDING
    completed_tasks: int = 0
    failed_tasks: int = 0
    queued_at: Optional[float] = None
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
    def avg_processing_time(self) -> Optional[float]:
        """Average processing time for completed tasks."""
        times = [task.processing_time for task in self.tasks 
                if task.processing_time is not None]
        return mean(times) if times else None

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100


logger = get_configure_logger(__file__)


class WhisperAudioProcessor:
    """
    Enhanced audio processor with comprehensive statistics and performance monitoring.
    """
    
    def __init__(self, hook: Callable[[BatchTask], None], max_retries: int = 3, 
                 stats_window_size: int = 100):
        self._hook = hook
        self._max_retries = max_retries
        self._stats_window_size = stats_window_size
        
        # Queues with size limits to prevent memory issues
        self._output_queue = Queue(maxsize=100)
        self._input_queue = Queue(maxsize=50)
        
        # Process and thread management
        self._worker_process: Optional[Process] = None
        self._hook_thread: Optional[Thread] = None
        self._stats_thread: Optional[Thread] = None
        self._stop_event = Event()
        
        # Shared state for worker process
        self._is_processing = Value('b', False)  # Boolean shared between processes
        
        # Statistics tracking - using simple data structures that can be pickled
        self._metrics = PerformanceMetrics()
        self._processing_times = deque(maxlen=stats_window_size)
        self._queue_times = deque(maxlen=stats_window_size)
        self._batch_history = deque(maxlen=50)  # Keep recent batch history
        
        # Monitoring
        self._is_running = False
        self._processed_batches = 0
        self._failed_batches = 0
        self._start_time: Optional[float] = None
        self._last_activity_time: Optional[float] = None
        
        # Performance tracking
        self._total_audio_duration = 0.0
        self._total_processing_time = 0.0
        
        # Memory management
        self._memory_cleanup_interval = 60.0  # Cleanup every 60 seconds
        self._last_memory_cleanup = time.time()
        self._memory_threshold_mb = 1024  # Force cleanup if memory usage exceeds 1GB
    
    def _cleanup_memory(self, force: bool = False):
        """Cleanup memory and garbage collect."""
        current_time = time.time()
        
        # Check if cleanup is needed
        time_since_cleanup = current_time - self._last_memory_cleanup
        memory_usage = psutil.virtual_memory().percent
        
        should_cleanup = (
            force or 
            time_since_cleanup >= self._memory_cleanup_interval or
            memory_usage > 80  # Force cleanup if memory usage is high
        )
        
        if should_cleanup:
            try:
                # Log memory usage before cleanup
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Garbage collection
                collected = gc.collect()
                
                # Update last cleanup time
                self._last_memory_cleanup = current_time
                
                # Log memory usage after cleanup
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_freed = memory_before - memory_after
                
                if memory_freed > 10:  # Only log if significant memory was freed
                    logger.info(f"[MEMORY] Cleanup: freed {memory_freed:.1f}MB, "
                              f"collected {collected} objects, memory: {memory_after:.1f}MB")
                else:
                    logger.debug(f"[MEMORY] Cleanup: {memory_after:.1f}MB used")
                    
            except Exception as e:
                logger.warning(f"[MEMORY] Cleanup error: {str(e)}")

    @contextmanager
    def _error_handling(self, context: str):
        """Context manager for consistent error handling."""
        try:
            yield
        except Exception as e:
            logger.error(f"[{context}] Error: {str(e)}")
            logger.debug(f"[{context}] Traceback: {traceback.format_exc()}")
            raise
    
    def _run_stats_thread(self):
        """Background thread for updating statistics and memory cleanup."""
        logger.info("[STATS] Statistics thread started")
        
        while not self._stop_event.is_set():
            try:
                self._update_system_metrics()
                self._update_performance_metrics()
                
                # Periodic memory cleanup
                self._cleanup_memory()
                
                time.sleep(1.0)  # Update stats every second
                
            except Exception as e:
                logger.error(f"[STATS] Error updating statistics: {str(e)}")
                time.sleep(1.0)
        
        logger.info("[STATS] Statistics thread stopped")
    
    def _update_system_metrics(self):
        """Update system resource usage metrics."""
        try:
            # CPU usage
            self._metrics.cpu_usage = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self._metrics.memory_usage = memory.percent
            
            # Process-specific stats if available
            if self._worker_process and self._worker_process.is_alive():
                try:
                    worker_process = psutil.Process(self._worker_process.pid)
                    worker_cpu = worker_process.cpu_percent()
                    worker_memory = worker_process.memory_info()
                    
                    # Store additional process-specific metrics
                    self._metrics.worker_cpu = worker_cpu
                    self._metrics.worker_memory_mb = worker_memory.rss / 1024 / 1024
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
        except Exception as e:
            logger.debug(f"[STATS] Error getting system metrics: {str(e)}")
    
    def _update_performance_metrics(self):
        """Update performance and utilization metrics."""
        current_time = time.time()
        
        # Update basic counters
        self._metrics.total_queued = self._input_queue.qsize()
        self._metrics.is_busy = bool(self._is_processing.value)
        
        # Calculate processing times
        if self._processing_times:
            self._metrics.avg_processing_time = mean(self._processing_times)
            self._metrics.median_processing_time = median(self._processing_times)
            self._metrics.processing_times = list(self._processing_times)
        
        # Calculate queue times
        if self._queue_times:
            self._metrics.avg_queue_time = mean(self._queue_times)
            self._metrics.queue_times = list(self._queue_times)
        
        # Calculate throughput
        if self._start_time:
            elapsed_time = current_time - self._start_time
            elapsed_minutes = elapsed_time / 60
            elapsed_hours = elapsed_time / 3600
            
            if elapsed_minutes > 0:
                self._metrics.throughput_per_minute = self._processed_batches / elapsed_minutes
            if elapsed_hours > 0:
                self._metrics.throughput_per_hour = self._processed_batches / elapsed_hours
        
        # Calculate utilization percentage
        if self._start_time and self._total_processing_time > 0:
            total_uptime = current_time - self._start_time
            self._metrics.utilization_percentage = (self._total_processing_time / total_uptime) * 100
    
    def _record_batch_completion(self, batch: BatchTask):
        """Record batch completion for statistics."""
        # No lock needed here since we're in the main thread
        self._batch_history.append(batch)
        
        # Update counters
        if batch.status == TaskStatus.COMPLETED:
            self._processed_batches += 1
            self._metrics.total_processed += batch.completed_tasks
        else:
            self._failed_batches += 1
        
        self._metrics.total_failed += batch.failed_tasks
        self._last_activity_time = time.time()
        
        # Record timing statistics
        for task in batch.tasks:
            if task.processing_time:
                self._processing_times.append(task.processing_time)
                self._total_processing_time += task.processing_time
            
            if task.queue_time:
                self._queue_times.append(task.queue_time)
        
        # Cleanup memory after batch completion
        self._cleanup_memory()

    def _run_hook_thread(self):
        """Enhanced hook thread with better error handling and monitoring."""
        logger.info("[HOOK] Hook thread started")
        
        while not self._stop_event.is_set():
            try:
                # Use timeout to allow periodic checking of stop event
                if not self._output_queue.empty():
                    batch = self._output_queue.get(timeout=1.0)
                    
                    # Record completion statistics
                    self._record_batch_completion(batch)
                    
                    with self._error_handling("HOOK"):
                        self._hook(batch)
                        logger.debug(f"[HOOK] Processed batch {batch.batch_id}")
                
                # Small sleep to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"[HOOK] Error in hook thread: {str(e)}")
                # Continue running even if hook fails
                continue
        
        logger.info("[HOOK] Hook thread stopped")
    
    @staticmethod
    def _worker_process_main(input_queue: Queue, output_queue: Queue, is_processing: Value):
        """Enhanced worker process with better resource management and timing."""
        logger.info("[WORKER] Worker process started")
        
        model = None
        last_cleanup = time.time()
        cleanup_interval = 300  # Cleanup every 5 minutes
        
        try:
            # Load model once at startup
            config = get_config()
            logger.info(f"[WORKER] Loading model: {config.model_name}")
            model = whisper_s2t.load_model(model_identifier=config.model_name)
            logger.info("[WORKER] Model loaded successfully")
            
            # Force garbage collection after model loading
            gc.collect()
            
            while True:
                try:
                    # Periodic memory cleanup in worker process
                    current_time = time.time()
                    if current_time - last_cleanup > cleanup_interval:
                        try:
                            process = psutil.Process(os.getpid())
                            memory_before = process.memory_info().rss / 1024 / 1024  # MB
                            
                            collected = gc.collect()
                            
                            memory_after = process.memory_info().rss / 1024 / 1024  # MB
                            memory_freed = memory_before - memory_after
                            
                            if memory_freed > 10:
                                logger.info(f"[WORKER] Memory cleanup: freed {memory_freed:.1f}MB, "
                                          f"collected {collected} objects")
                            
                            last_cleanup = current_time
                        except Exception as e:
                            logger.debug(f"[WORKER] Memory cleanup error: {str(e)}")
                    
                    # Use timeout to allow graceful shutdown
                    if not input_queue.empty():
                        batch: BatchTask = input_queue.get(timeout=1.0)
                        
                        # Mark as busy
                        with is_processing.get_lock():
                            is_processing.value = True
                        
                        try:
                            WhisperAudioProcessor._process_batch_static(batch, model, output_queue)
                        finally:
                            # Mark as not busy
                            with is_processing.get_lock():
                                is_processing.value = False
                            
                            # Force garbage collection after processing
                            gc.collect()
                    
                    time.sleep(0.1)  # Prevent busy waiting
                    
                except Exception as e:
                    logger.error(f"[WORKER] Error in main loop: {str(e)}")
                    # Ensure we're marked as not busy on error
                    with is_processing.get_lock():
                        is_processing.value = False
                    continue
                    
        except Exception as e:
            logger.error(f"[WORKER] Fatal error in worker process: {str(e)}")
        finally:
            # Cleanup
            logger.info("[WORKER] Cleaning up worker process")
            try:
                if model:
                    del model
                    logger.debug("[WORKER] Model deleted")
                
                # Force final garbage collection
                collected = gc.collect()
                logger.debug(f"[WORKER] Final cleanup: collected {collected} objects")
                
            except Exception as e:
                logger.warning(f"[WORKER] Cleanup error: {str(e)}")
            finally:
                # Ensure we're marked as not busy
                with is_processing.get_lock():
                    is_processing.value = False
                logger.info("[WORKER] Worker process stopped")
    
    @staticmethod
    def _process_batch_static(batch: BatchTask, model: Any, output_queue: Queue):
        """Static method to process a single batch with detailed timing and error handling."""
        batch.status = TaskStatus.PROCESSING
        batch.processing_started_at = time.time()
        
        logger.info(f"[WORKER] Processing batch {batch.batch_id} with {len(batch.tasks)} tasks")
        
        try:
            # Prepare batch data and set task timing
            files = []
            lang_codes = []
            tasks = []
            initial_prompts = []
            valid_indices = []
            
            for idx, task in enumerate(batch.tasks):
                if WhisperAudioProcessor._validate_task_static(task):
                    task.processing_started_at = time.time()
                    files.append(task.file_path)
                    lang_codes.append(task.lang)
                    tasks.append(task.task)
                    initial_prompts.append(task.prompt)
                    valid_indices.append(idx)
                    task.status = TaskStatus.PROCESSING
                else:
                    task.status = TaskStatus.FAILED
                    task.error = "Task validation failed"
                    task.processed_at = time.time()
                    batch.failed_tasks += 1
            
            if not files:
                logger.warning(f"[WORKER] No valid tasks in batch {batch.batch_id}")
                batch.status = TaskStatus.FAILED
                batch.end_time = time.time()
                output_queue.put(batch)
                return
            
            # Process with whisper
            config = get_config()
            logger.debug(f"[WORKER] Starting transcription for {len(files)} files")
            
            processing_start = time.time()
            results = model.transcribe_with_vad(
                files,
                lang_codes=lang_codes,
                tasks=tasks,
                initial_prompts=initial_prompts,
                batch_size=config.batch_size
            )
            processing_end = time.time()
            
            # Update tasks with results and timing
            for result_idx, batch_idx in enumerate(valid_indices):
                task = batch.tasks[batch_idx]
                task.output = results[result_idx]
                task.status = TaskStatus.COMPLETED
                task.processed_at = processing_end
                
                # Create detailed stats
                task.stats = ProcessingStats(
                    start_time=task.processing_started_at,
                    end_time=task.processed_at,
                    queue_time=task.queue_time,
                    processing_time=task.processing_time
                )
                
                batch.completed_tasks += 1
                logger.debug(f"[WORKER] Task {task.task_id} completed in {task.processing_time:.2f}s")
            
            batch.status = TaskStatus.COMPLETED
            batch.end_time = processing_end
            
            duration = batch.end_time - batch.processing_started_at
            avg_per_task = duration / len(valid_indices) if valid_indices else 0
            logger.info(f"[WORKER] Batch {batch.batch_id} completed in {duration:.2f}s "
                       f"(avg: {avg_per_task:.2f}s per task)")
            
            # Clear large variables to help garbage collection
            del results
            del files
            del lang_codes
            del tasks
            del initial_prompts
            
        except Exception as e:
            logger.error(f"[WORKER] Error processing batch {batch.batch_id}: {str(e)}")
            logger.debug(f"[WORKER] Traceback: {traceback.format_exc()}")
            
            # Mark all remaining tasks as failed with timing
            current_time = time.time()
            for task in batch.tasks:
                if task.status == TaskStatus.PROCESSING:
                    task.status = TaskStatus.FAILED
                    task.error = f"Batch processing failed: {str(e)}"
                    task.processed_at = current_time
                    
                    if task.processing_started_at:
                        task.stats = ProcessingStats(
                            start_time=task.processing_started_at,
                            end_time=current_time,
                            queue_time=task.queue_time
                        )
                    
                    batch.failed_tasks += 1
            
            batch.status = TaskStatus.FAILED
            batch.end_time = current_time
        
        finally:
            # Always send batch back
            output_queue.put(batch)
    
    @staticmethod
    def _validate_task_static(task: Task) -> bool:
        """Static method to validate a task before processing."""
        if not task.file_path:
            logger.warning(f"[VALIDATION] Task {task.task_id}: Missing file path")
            return False
        
        if not task.lang:
            logger.warning(f"[VALIDATION] Task {task.task_id}: Missing language")
            return False
        
        if not task.task:
            logger.warning(f"[VALIDATION] Task {task.task_id}: Missing task type")
            return False
        
        # Add file existence check if needed
        # if not os.path.exists(task.file_path):
        #     logger.warning(f"[VALIDATION] Task {task.task_id}: File not found: {task.file_path}")
        #     return False
        
        return True
    
    def start(self) -> bool:
        """Start the processor with proper error handling."""
        if self._is_running:
            logger.warning("[PROCESSOR] Already running")
            return False
        
        try:
            logger.info("[PROCESSOR] Starting audio processor")
            self._start_time = time.time()
            self._stop_event.clear()
            
            # Reset shared state
            with self._is_processing.get_lock():
                self._is_processing.value = False
            
            # Start worker process
            self._worker_process = Process(
                target=self._worker_process_main, 
                args=(self._input_queue, self._output_queue, self._is_processing)
            )
            self._worker_process.start()
            
            # Start hook thread
            self._hook_thread = Thread(target=self._run_hook_thread)
            self._hook_thread.daemon = True
            self._hook_thread.start()
            
            # Start statistics thread
            self._stats_thread = Thread(target=self._run_stats_thread)
            self._stats_thread.daemon = True
            self._stats_thread.start()
            
            self._is_running = True
            logger.info("[PROCESSOR] Audio processor started successfully")
            return True
            
        except Exception as e:
            logger.error(f"[PROCESSOR] Failed to start: {str(e)}")
            self.stop()
            return False
    
    def stop(self, timeout: float = 10.0) -> bool:
        """Graceful shutdown with timeout and memory cleanup."""
        if not self._is_running:
            logger.warning("[PROCESSOR] Not running")
            return True
        
        logger.info("[PROCESSOR] Stopping audio processor")
        
        try:
            # Signal threads to stop
            self._stop_event.set()
            
            # Wait for hook thread
            if self._hook_thread and self._hook_thread.is_alive():
                self._hook_thread.join(timeout=timeout/2)
                if self._hook_thread.is_alive():
                    logger.warning("[PROCESSOR] Hook thread did not stop gracefully")
            
            # Wait for stats thread
            if self._stats_thread and self._stats_thread.is_alive():
                self._stats_thread.join(timeout=timeout/4)
                if self._stats_thread.is_alive():
                    logger.warning("[PROCESSOR] Stats thread did not stop gracefully")
            
            # Terminate worker process
            if self._worker_process and self._worker_process.is_alive():
                logger.info("[PROCESSOR] Terminating worker process")
                self._worker_process.terminate()
                self._worker_process.join(timeout=timeout/4)
                
                if self._worker_process.is_alive():
                    logger.warning("[PROCESSOR] Force killing worker process")
                    self._worker_process.kill()
                    self._worker_process.join()
            
            # Clean up queues
            try:
                while not self._input_queue.empty():
                    self._input_queue.get_nowait()
            except:
                pass
            
            try:
                while not self._output_queue.empty():
                    self._output_queue.get_nowait()
            except:
                pass
            
            self._is_running = False
            
            # Log statistics
            if self._start_time:
                uptime = time.time() - self._start_time
                logger.info(f"[PROCESSOR] Stopped. Uptime: {uptime:.2f}s, "
                          f"Processed: {self._processed_batches}, Failed: {self._failed_batches}")
            
            # Final memory cleanup
            self._cleanup_memory(force=True)
            
            return True
            
        except Exception as e:
            logger.error(f"[PROCESSOR] Error during shutdown: {str(e)}")
            return False
    
    def add_batch(self, batch: BatchTask) -> bool:
        """Add a batch for processing with queue management."""
        if not self._is_running:
            logger.error("[PROCESSOR] Cannot add batch: processor not running")
            return False
        
        try:
            # Check queue capacity
            if self._input_queue.full():
                logger.warning(f"[PROCESSOR] Input queue full, cannot add batch {batch.batch_id}")
                return False
            
            # Set queued timestamp
            batch.queued_at = time.time()
            for task in batch.tasks:
                task.queued_at = batch.queued_at
            
            self._input_queue.put_nowait(batch)
            logger.info(f"[PROCESSOR] Added batch {batch.batch_id} to queue")
            return True
            
        except Exception as e:
            logger.error(f"[PROCESSOR] Failed to add batch {batch.batch_id}: {str(e)}")
            return False
    
    def get_stats(self) -> dict:
        """Get comprehensive processor statistics."""
        current_time = time.time()
        
        base_stats = {
            "is_running": self._is_running,
            "processed_batches": self._processed_batches,
            "failed_batches": self._failed_batches,
            "input_queue_size": self._input_queue.qsize() if self._is_running else 0,
            "output_queue_size": self._output_queue.qsize() if self._is_running else 0,
            "uptime": current_time - self._start_time if self._start_time else 0,
            "last_activity": current_time - self._last_activity_time if self._last_activity_time else None
        }
        
        # Add performance metrics
        metrics_dict = {
            "total_processed": self._metrics.total_processed,
            "total_failed": self._metrics.total_failed,
            "total_queued": self._metrics.total_queued,
            "avg_processing_time": self._metrics.avg_processing_time,
            "median_processing_time": self._metrics.median_processing_time,
            "avg_queue_time": self._metrics.avg_queue_time,
            "throughput_per_minute": self._metrics.throughput_per_minute,
            "throughput_per_hour": self._metrics.throughput_per_hour,
            "cpu_usage": self._metrics.cpu_usage,
            "memory_usage": self._metrics.memory_usage,
            "is_busy": self._metrics.is_busy,
            "utilization_percentage": self._metrics.utilization_percentage
        }
        
        return {**base_stats, **metrics_dict}
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get detailed performance metrics object."""
        return self._metrics
    
    def force_memory_cleanup(self) -> dict:
        """Force immediate memory cleanup and return memory stats."""
        try:
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Force cleanup
            self._cleanup_memory(force=True)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = memory_before - memory_after
            
            return {
                "memory_before_mb": memory_before,
                "memory_after_mb": memory_after,
                "memory_freed_mb": memory_freed,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"[MEMORY] Force cleanup error: {str(e)}")
            return {"error": str(e)}
    
    def get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            
            # Process memory
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
            
            # Worker process memory (if available)
            worker_memory = None
            if self._worker_process and self._worker_process.is_alive():
                try:
                    worker_process = psutil.Process(self._worker_process.pid)
                    worker_memory = worker_process.memory_info()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            stats = {
                "system_memory": {
                    "total_gb": system_memory.total / (1024**3),
                    "available_gb": system_memory.available / (1024**3),
                    "used_percent": system_memory.percent
                },
                "main_process": {
                    "rss_mb": process_memory.rss / (1024**2),
                    "vms_mb": process_memory.vms / (1024**2)
                },
                "worker_process": {
                    "rss_mb": worker_memory.rss / (1024**2) if worker_memory else None,
                    "vms_mb": worker_memory.vms / (1024**2) if worker_memory else None
                } if worker_memory else None,
                "last_cleanup": self._last_memory_cleanup,
                "cleanup_interval": self._memory_cleanup_interval
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"[MEMORY] Stats error: {str(e)}")
            return {"error": str(e)}
    
    def __enter__(self):
        """Context manager support."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.stop()