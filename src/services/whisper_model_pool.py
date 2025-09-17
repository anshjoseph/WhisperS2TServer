from libs.whisper_audio_processor import WhisperAudioProcessor
from models.batch_task import BatchTask, Task
from config import get_config
from utils.log import get_configure_logger
from typing import Dict, Optional, List
import time
import threading
import uuid
from dataclasses import dataclass
from enum import Enum

logger = get_configure_logger(__file__)


class ModelStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ModelInstance:
    """Represents a model instance in the pool."""
    model_id: str
    processor: WhisperAudioProcessor
    status: ModelStatus
    created_at: float
    last_used_at: float
    ttl_seconds: float
    queue_size: int = 0
    total_processed: int = 0
    total_failed: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if the model instance has expired based on TTL."""
        return time.time() - self.last_used_at > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get the age of the model instance in seconds."""
        return time.time() - self.created_at

    def renew_ttl(self):
        """Renew the TTL by updating last_used_at."""
        self.last_used_at = time.time()


class WhisperModelPool:
    """
    Manages a pool of WhisperAudioProcessor instances with auto-scaling and TTL management.
    """
    
    def __init__(self, hook, ttl_seconds: float = 300.0, queue_threshold: int = 5):
        self.__pool: Dict[str, ModelInstance] = {}
        self.__config = get_config()
        self.__hook = hook
        self.__ttl_seconds = ttl_seconds
        self.__queue_threshold = queue_threshold
        self.__lock = threading.RLock()
        self.__cleanup_thread: Optional[threading.Thread] = None
        self.__stop_cleanup = threading.Event()
        self.__is_running = False
        
        # Statistics
        self.__total_batches_routed = 0
        self.__total_models_created = 0
        self.__total_models_destroyed = 0
        
        logger.info(f"[POOL] Initialized with TTL: {ttl_seconds}s, Queue threshold: {queue_threshold}")

    def _create_model_id(self) -> str:
        """Generate a unique model ID."""
        return f"model_{uuid.uuid4().hex[:8]}_{int(time.time())}"

    def _batch_completed_hook(self, model_id: str):
        """Create a batch completion hook for a specific model."""
        def hook(batch: BatchTask):
            with self.__lock:
                if model_id in self.__pool:
                    instance = self.__pool[model_id]
                    instance.renew_ttl()
                    instance.total_processed += batch.completed_tasks
                    instance.total_failed += batch.failed_tasks
                    instance.queue_size = max(0, instance.queue_size - 1)
                    
                    # Update status based on queue
                    if instance.queue_size == 0:
                        instance.status = ModelStatus.IDLE
                    
                    self.__hook(batch)
                    
                    logger.debug(f"[POOL] Model {model_id} completed batch {batch.batch_id}. "
                               f"Queue size: {instance.queue_size}")
        return hook

    def _add_model(self) -> Optional[str]:
        """Add a new model instance to the pool."""
        try:
            model_id = self._create_model_id()
            logger.info(f"[POOL] Creating new model instance: {model_id}")
            
            # Create processor with model-specific hook
            hook = self._batch_completed_hook(model_id)
            processor = WhisperAudioProcessor(hook)
            
            # Create model instance
            instance = ModelInstance(
                model_id=model_id,
                processor=processor,
                status=ModelStatus.STARTING,
                created_at=time.time(),
                last_used_at=time.time(),
                ttl_seconds=self.__ttl_seconds
            )
            
            # Start the processor
            if processor.start():
                instance.status = ModelStatus.IDLE
                with self.__lock:
                    self.__pool[model_id] = instance
                    self.__total_models_created += 1
                
                logger.info(f"[POOL] Model {model_id} created and started successfully")
                return model_id
            else:
                instance.status = ModelStatus.ERROR
                logger.error(f"[POOL] Failed to start processor for model {model_id}")
                return None
                
        except Exception as e:
            logger.error(f"[POOL] Error creating model: {str(e)}")
            return None

    def _remove_model(self, model_id: str) -> bool:
        """Remove a model instance from the pool."""
        try:
            with self.__lock:
                if model_id not in self.__pool:
                    logger.warning(f"[POOL] Model {model_id} not found for removal")
                    return False
                
                instance = self.__pool[model_id]
                instance.status = ModelStatus.STOPPING
                
                logger.info(f"[POOL] Stopping model {model_id} (age: {instance.age_seconds:.1f}s)")
                
                # Stop the processor
                if instance.processor.stop(timeout=10.0):
                    del self.__pool[model_id]
                    self.__total_models_destroyed += 1
                    logger.info(f"[POOL] Model {model_id} removed successfully")
                    return True
                else:
                    logger.warning(f"[POOL] Model {model_id} did not stop cleanly")
                    # Force remove even if stop failed
                    del self.__pool[model_id]
                    self.__total_models_destroyed += 1
                    return False
                    
        except Exception as e:
            logger.error(f"[POOL] Error removing model {model_id}: {str(e)}")
            return False

    def _find_best_model(self) -> Optional[str]:
        """Find the best available model for routing a batch."""
        with self.__lock:
            if not self.__pool:
                return None
            
            # Priority 1: Idle models
            idle_models = [
                (model_id, instance) for model_id, instance in self.__pool.items()
                if instance.status == ModelStatus.IDLE
            ]
            
            if idle_models:
                # Return the idle model with the smallest queue
                return min(idle_models, key=lambda x: x[1].queue_size)[0]
            
            # Priority 2: Busy models with smallest queue
            busy_models = [
                (model_id, instance) for model_id, instance in self.__pool.items()
                if instance.status == ModelStatus.BUSY
            ]
            
            if busy_models:
                return min(busy_models, key=lambda x: x[1].queue_size)[0]
            
            return None

    def _should_scale_up(self) -> bool:
        """Determine if we should create a new model instance."""
        with self.__lock:
            if not self.__config.enable_auto_scale:
                return len(self.__pool) < self.__config.pool_config.size
            
            # Check if we've reached the auto-scale limit
            if len(self.__pool) >= self.__config.max_auto_scale_limit:
                return False
            
            # Scale up if all models are busy and have queues above threshold
            busy_models = [
                instance for instance in self.__pool.values()
                if instance.status == ModelStatus.BUSY and instance.queue_size >= self.__queue_threshold
            ]
            
            return len(busy_models) == len(self.__pool) and len(self.__pool) > 0

    def _cleanup_expired_models(self):
        """Background thread to clean up expired models."""
        logger.info("[POOL] Cleanup thread started")
        
        while not self.__stop_cleanup.is_set():
            try:
                current_time = time.time()
                models_to_remove = []
                
                with self.__lock:
                    for model_id, instance in self.__pool.items():
                        # Only remove idle models that are expired
                        if (instance.status == ModelStatus.IDLE and 
                            instance.is_expired and 
                            len(self.__pool) > self.__config.pool_config.size):
                            models_to_remove.append(model_id)
                
                # Remove expired models
                for model_id in models_to_remove:
                    logger.info(f"[POOL] Removing expired model: {model_id}")
                    self._remove_model(model_id)
                
                # Sleep for cleanup interval
                self.__stop_cleanup.wait(30.0)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"[POOL] Cleanup thread error: {str(e)}")
                self.__stop_cleanup.wait(30.0)
        
        logger.info("[POOL] Cleanup thread stopped")

    def start(self) -> bool:
        """Start the model pool with initial models."""
        if self.__is_running:
            logger.warning("[POOL] Already running")
            return True
        
        try:
            logger.info(f"[POOL] Starting pool with {self.__config.pool_config.size} initial models")
            
            # Create initial models
            created_count = 0
            for i in range(self.__config.pool_config.size):
                model_id = self._add_model()
                if model_id:
                    created_count += 1
                else:
                    logger.warning(f"[POOL] Failed to create initial model {i + 1}")
            
            if created_count == 0:
                logger.error("[POOL] Failed to create any initial models")
                return False
            
            # Start cleanup thread
            self.__cleanup_thread = threading.Thread(target=self._cleanup_expired_models)
            self.__cleanup_thread.daemon = True
            self.__cleanup_thread.start()
            
            self.__is_running = True
            logger.info(f"[POOL] Started successfully with {created_count}/{self.__config.pool_config.size} models")
            return True
            
        except Exception as e:
            logger.error(f"[POOL] Failed to start: {str(e)}")
            return False

    def stop(self, timeout: float = 30.0) -> bool:
        """Stop the model pool and all instances."""
        if not self.__is_running:
            logger.warning("[POOL] Not running")
            return True
        
        logger.info("[POOL] Stopping model pool")
        
        try:
            # Stop cleanup thread
            self.__stop_cleanup.set()
            if self.__cleanup_thread and self.__cleanup_thread.is_alive():
                self.__cleanup_thread.join(timeout=10.0)
            
            # Stop all models
            model_ids = list(self.__pool.keys())
            stopped_count = 0
            
            for model_id in model_ids:
                if self._remove_model(model_id):
                    stopped_count += 1
            
            self.__is_running = False
            logger.info(f"[POOL] Stopped. {stopped_count}/{len(model_ids)} models stopped cleanly")
            return stopped_count == len(model_ids)
            
        except Exception as e:
            logger.error(f"[POOL] Error during stop: {str(e)}")
            return False

    def add_batch(self, batch: BatchTask) -> bool:
        """Add a batch to the pool for processing."""
        if not self.__is_running:
            logger.error("[POOL] Cannot add batch: pool not running")
            return False
        
        try:
            # Find the best model for this batch
            model_id = self._find_best_model()
            
            # If no model available or should scale up, try to create a new one
            if model_id is None or self._should_scale_up():
                logger.info("[POOL] Attempting to scale up")
                new_model_id = self._add_model()
                if new_model_id:
                    model_id = new_model_id
                    logger.info(f"[POOL] Scaled up: created model {new_model_id}")
            
            if model_id is None:
                logger.error("[POOL] No available models for batch processing")
                return False
            
            with self.__lock:
                instance = self.__pool[model_id]
                
                # Add batch to the selected model
                if instance.processor.add_batch(batch):
                    instance.renew_ttl()
                    instance.queue_size += 1
                    instance.status = ModelStatus.BUSY
                    self.__total_batches_routed += 1
                    
                    logger.info(f"[POOL] Batch {batch.batch_id} routed to model {model_id} "
                              f"(queue: {instance.queue_size})")
                    return True
                else:
                    logger.error(f"[POOL] Failed to add batch to model {model_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"[POOL] Error adding batch: {str(e)}")
            return False

    def get_pool_stats(self) -> Dict:
        """Get comprehensive pool statistics."""
        with self.__lock:
            model_stats = {}
            total_queue_size = 0
            idle_count = 0
            busy_count = 0
            
            for model_id, instance in self.__pool.items():
                stats = instance.processor.get_stats()
                model_stats[model_id] = {
                    "status": instance.status.value,
                    "age_seconds": instance.age_seconds,
                    "queue_size": instance.queue_size,
                    "total_processed": instance.total_processed,
                    "total_failed": instance.total_failed,
                    "uptime": stats.uptime,
                    "is_busy": stats.performance_metrics.is_busy,
                    "avg_processing_time": stats.performance_metrics.avg_processing_time,
                    "throughput_per_hour": stats.performance_metrics.throughput_per_hour
                }
                
                total_queue_size += instance.queue_size
                if instance.status == ModelStatus.IDLE:
                    idle_count += 1
                elif instance.status == ModelStatus.BUSY:
                    busy_count += 1
            
            return {
                "pool_size": len(self.__pool),
                "idle_models": idle_count,
                "busy_models": busy_count,
                "total_queue_size": total_queue_size,
                "total_batches_routed": self.__total_batches_routed,
                "total_models_created": self.__total_models_created,
                "total_models_destroyed": self.__total_models_destroyed,
                "is_running": self.__is_running,
                "config": {
                    "initial_size": self.__config.pool_config.size,
                    "auto_scale_enabled": self.__config.enable_auto_scale,
                    "max_auto_scale_limit": self.__config.max_auto_scale_limit,
                    "queue_threshold": self.__queue_threshold,
                    "ttl_seconds": self.__ttl_seconds
                },
                "models": model_stats
            }

    def get_model_count(self) -> int:
        """Get the current number of models in the pool."""
        with self.__lock:
            return len(self.__pool)

    def force_cleanup(self) -> int:
        """Force cleanup of expired models and return count of removed models."""
        removed_count = 0
        models_to_remove = []
        
        with self.__lock:
            for model_id, instance in self.__pool.items():
                if (instance.status == ModelStatus.IDLE and 
                    instance.is_expired and 
                    len(self.__pool) > self.__config.pool_config.size):
                    models_to_remove.append(model_id)
        
        for model_id in models_to_remove:
            if self._remove_model(model_id):
                removed_count += 1
        
        logger.info(f"[POOL] Force cleanup removed {removed_count} expired models")
        return removed_count