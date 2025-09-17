import whisper_s2t
import time
import traceback
import psutil
import gc
import os
from collections import deque
from multiprocessing import Process, Queue, Value
from threading import Thread, Event
from config import get_config
from utils.log import get_configure_logger
from typing import Optional, Callable, Any
from models.batch_task import Task, TaskStatus, BatchTask, PerformanceMetrics, ProcessingStats, ProcessorStats

logger = get_configure_logger(__file__)


class WhisperAudioProcessor:
    """
    Simplified audio processor with essential statistics and performance monitoring.
    """
    
    def __init__(self, hook: Callable[[BatchTask], None], max_retries: int = 3):
        self._hook = hook
        self._max_retries = max_retries
        
        # Queues
        self._output_queue = Queue(maxsize=100)
        self._input_queue = Queue(maxsize=50)
        
        # Process and thread management
        self._worker_process: Optional[Process] = None
        self._hook_thread: Optional[Thread] = None
        self._stats_thread: Optional[Thread] = None
        self._stop_event = Event()
        
        # Shared state for worker process
        self._is_processing = Value('b', False)
        
        # Essential statistics only
        self._metrics = PerformanceMetrics()
        self._processing_times = deque(maxlen=50)  # Reduced size
        
        # Basic monitoring
        self._is_running = False
        self._processed_batches = 0
        self._failed_batches = 0
        self._start_time: Optional[float] = None
        
        # Memory management - simplified
        self._last_memory_cleanup = time.time()
        self._memory_cleanup_interval = 120.0  # Cleanup every 2 minutes
    
    def _cleanup_memory(self):
        """Simple memory cleanup."""
        current_time = time.time()
        
        if current_time - self._last_memory_cleanup >= self._memory_cleanup_interval:
            try:
                collected = gc.collect()
                self._last_memory_cleanup = current_time
                logger.debug(f"[MEMORY] Cleanup: collected {collected} objects")
            except Exception as e:
                logger.warning(f"[MEMORY] Cleanup error: {str(e)}")
    
    def _run_stats_thread(self):
        """Background thread for updating basic statistics."""
        logger.info("[STATS] Statistics thread started")
        
        while not self._stop_event.is_set():
            try:
                # Basic metrics
                self._metrics.is_busy = bool(self._is_processing.value)
                
                # Calculate throughput
                if self._start_time:
                    elapsed_hours = (time.time() - self._start_time) / 3600
                    if elapsed_hours > 0:
                        self._metrics.throughput_per_hour = self._processed_batches / elapsed_hours
                
                # Calculate average processing time
                if self._processing_times:
                    self._metrics.avg_processing_time = sum(self._processing_times) / len(self._processing_times)
                
                # Memory cleanup
                self._cleanup_memory()
                
                time.sleep(2.0)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"[STATS] Error updating statistics: {str(e)}")
                time.sleep(2.0)
        
        logger.info("[STATS] Statistics thread stopped")
    
    def _record_batch_completion(self, batch: BatchTask):
        """Record batch completion for statistics."""
        if batch.status == TaskStatus.COMPLETED:
            self._processed_batches += 1
            self._metrics.total_processed += batch.completed_tasks
        else:
            self._failed_batches += 1
        
        self._metrics.total_failed += batch.failed_tasks
        
        # Record processing times
        for task in batch.tasks:
            if task.processing_time:
                self._processing_times.append(task.processing_time)

    def _run_hook_thread(self):
        """Hook thread with basic error handling."""
        logger.info("[HOOK] Hook thread started")
        
        while not self._stop_event.is_set():
            try:
                if not self._output_queue.empty():
                    batch = self._output_queue.get(timeout=1.0)
                    self._record_batch_completion(batch)
                    self._hook(batch)
                    logger.debug(f"[HOOK] Processed batch {batch.batch_id}")
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"[HOOK] Error in hook thread: {str(e)}")
                continue
        
        logger.info("[HOOK] Hook thread stopped")
    
    @staticmethod
    def _worker_process_main(input_queue: Queue, output_queue: Queue, is_processing):
        """Simplified worker process."""
        logger.info("[WORKER] Worker process started")
        
        model = None
        
        try:
            # Load model
            config = get_config()
            logger.info(f"[WORKER] Loading model: {config.model_name}")
            model = whisper_s2t.load_model(model_identifier=config.model_name)
            logger.info("[WORKER] Model loaded successfully")
            
            while True:
                try:
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
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"[WORKER] Error in main loop: {str(e)}")
                    with is_processing.get_lock():
                        is_processing.value = False
                    continue
                    
        except Exception as e:
            logger.error(f"[WORKER] Fatal error in worker process: {str(e)}")
        finally:
            logger.info("[WORKER] Cleaning up worker process")
            try:
                if model:
                    del model
                gc.collect()
            except Exception as e:
                logger.warning(f"[WORKER] Cleanup error: {str(e)}")
            finally:
                with is_processing.get_lock():
                    is_processing.value = False
                logger.info("[WORKER] Worker process stopped")
    
    @staticmethod
    def _process_batch_static(batch: BatchTask, model: Any, output_queue: Queue):
        """Process a single batch with basic timing."""
        batch.status = TaskStatus.PROCESSING
        batch.processing_started_at = time.time()
        
        logger.info(f"[WORKER] Processing batch {batch.batch_id} with {len(batch.tasks)} tasks")
        
        try:
            # Prepare batch data
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
            
            # Update tasks with results
            for result_idx, batch_idx in enumerate(valid_indices):
                task = batch.tasks[batch_idx]
                task.output = results[result_idx]
                task.status = TaskStatus.COMPLETED
                task.processed_at = processing_end
                
                # Create basic stats
                task.stats = ProcessingStats(
                    start_time=task.processing_started_at,
                    end_time=task.processed_at,
                    processing_time=task.processing_time
                )
                
                batch.completed_tasks += 1
            
            batch.status = TaskStatus.COMPLETED
            batch.end_time = processing_end
            
            duration = batch.end_time - batch.processing_started_at
            logger.info(f"[WORKER] Batch {batch.batch_id} completed in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"[WORKER] Error processing batch {batch.batch_id}: {str(e)}")
            
            # Mark all remaining tasks as failed
            current_time = time.time()
            for task in batch.tasks:
                if task.status == TaskStatus.PROCESSING:
                    task.status = TaskStatus.FAILED
                    task.error = f"Batch processing failed: {str(e)}"
                    task.processed_at = current_time
                    batch.failed_tasks += 1
            
            batch.status = TaskStatus.FAILED
            batch.end_time = current_time
        
        finally:
            output_queue.put(batch)
    
    @staticmethod
    def _validate_task_static(task: Task) -> bool:
        """Validate a task before processing."""
        if not task.file_path:
            logger.warning(f"[VALIDATION] Task {task.task_id}: Missing file path")
            return False
        
        if not task.lang:
            logger.warning(f"[VALIDATION] Task {task.task_id}: Missing language")
            return False
        
        if not task.task:
            logger.warning(f"[VALIDATION] Task {task.task_id}: Missing task type")
            return False
        
        return True
    
    def start(self) -> bool:
        """Start the processor."""
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
        """Stop the processor."""
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
            
            # Wait for stats thread
            if self._stats_thread and self._stats_thread.is_alive():
                self._stats_thread.join(timeout=timeout/4)
            
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
                while not self._output_queue.empty():
                    self._output_queue.get_nowait()
            except:
                pass
            
            self._is_running = False
            
            # Log final stats
            if self._start_time:
                uptime = time.time() - self._start_time
                logger.info(f"[PROCESSOR] Stopped. Uptime: {uptime:.2f}s, "
                          f"Processed: {self._processed_batches}, Failed: {self._failed_batches}")
            
            return True
            
        except Exception as e:
            logger.error(f"[PROCESSOR] Error during shutdown: {str(e)}")
            return False
    
    def add_batch(self, batch: BatchTask) -> bool:
        """Add a batch for processing."""
        if not self._is_running:
            logger.error("[PROCESSOR] Cannot add batch: processor not running")
            return False
        
        try:
            if self._input_queue.full():
                logger.warning(f"[PROCESSOR] Input queue full, cannot add batch {batch.batch_id}")
                return False
            
            self._input_queue.put_nowait(batch)
            logger.info(f"[PROCESSOR] Added batch {batch.batch_id} to queue")
            return True
            
        except Exception as e:
            logger.error(f"[PROCESSOR] Failed to add batch {batch.batch_id}: {str(e)}")
            return False
    
    def get_stats(self) -> ProcessorStats:
        """Get processor statistics as a class object."""
        current_time = time.time()
        
        return ProcessorStats(
            is_running=self._is_running,
            processed_batches=self._processed_batches,
            failed_batches=self._failed_batches,
            input_queue_size=self._input_queue.qsize() if self._is_running else 0,
            output_queue_size=self._output_queue.qsize() if self._is_running else 0,
            uptime=current_time - self._start_time if self._start_time else 0,
            performance_metrics=self._metrics
        )
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics object."""
        return self._metrics