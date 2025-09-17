from models.batch_task import BatchTask, Task
from typing import List, Dict
from libs.whisper_model_pool import WhisperModelPool
from threading import Thread, Event, Lock
from uuid import uuid4
import time
import asyncio
from config import get_config


class TaskManager:
    def __init__(self):
        self.__pool: WhisperModelPool = WhisperModelPool()
        self.__tasks: List[Task] = []
        self.__results: Dict[str, asyncio.Future] = {}
        self.__lock = Lock()

        # Threads
        self._task_batch_thread = Thread(target=self.task_batching, daemon=True)
        self._scaler_thread = Thread(target=self.auto_scaler_loop, daemon=True)
        self._stop_event = Event()

        self._config = get_config()

        # scaling cooldown
        self._last_scale_time = 0
        self._scale_cooldown = 5  # seconds

    async def add_task(self, task: Task):
        """
        Async version of add_task. Returns a future that resolves
        when the output of this task is available.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self.__lock:
            self.__tasks.append(task)
            self.__results[str(task.task_id)] = future
        return await future

    def _task_hook(self, batch: BatchTask):
        """
        Hook function called by each WhisperAudioProcessor
        when a batch completes.
        Sets the result in the corresponding futures.
        """
        for task in batch.tasks:
            self.set_task_result(str(task.task_id), task.output)

    def set_task_result(self, task_id: str, result: dict):
        """
        Sets result for a task future.
        """
        with self.__lock:
            future = self.__results.get(task_id)
            if future and not future.done():
                future.set_result(result)

    def _scheduler(self):
        """Pick a free model from pool."""
        for proc_id in self.__pool.get_models():
            stats = self.__pool.get_model_stats(proc_id)
            if not stats.processing:
                return proc_id
        return None

    def task_batching(self):
        """Continuously pick tasks, batch them, and schedule on models."""
        while not self._stop_event.is_set():
            if len(self.__tasks) >= 1:
                proc_id = self._scheduler()
                if proc_id is not None:
                    tasks = []
                    with self.__lock:
                        while len(tasks) < self._config.batch_size and self.__tasks:
                            tasks.append(self.__tasks.pop(0))

                    batch = BatchTask(
                        batch_id=uuid4(),
                        start_time=time.time(),
                        end_time=-1,
                        tasks=tasks,
                    )
                    self.__pool.add_task(proc_id, batch)

            time.sleep(0.2)

    def auto_scaler_loop(self):
        """Monitor load and scale pool."""
        while not self._stop_event.is_set():
            self._auto_scale()
            time.sleep(1)

    def _auto_scale(self):
        if not self._config.enable_auto_scale:
            return

        now = time.time()
        if now - self._last_scale_time < self._scale_cooldown:
            return

        queue_len = len(self.__tasks)
        pool_size = len(self.__pool.get_models())

        # scale up if overloaded
        if queue_len > pool_size * self._config.batch_size:
            if pool_size < self._config.max_auto_scale_limit:
                self.__pool.add_model(hook=self._task_hook)
                print(f"[Scaler] Added model instance. Pool size: {len(self.__pool.get_models())}")

        # scale down if idle
        elif queue_len == 0:
            busy_models = sum(
                1 for m in self.__pool.get_models()
                if self.__pool.get_model_stats(m).processing
            )
            if busy_models == 0 and pool_size > self._config.pool_config.size:
                victim_id = self.__pool.get_models()[-1]
                self.__pool.close_model(victim_id)
                print(f"[Scaler] Closed model instance {victim_id}. Pool size: {len(self.__pool.get_models())}")

        self._last_scale_time = now

    def start(self):
        """Preload min models, then start threads."""
        min_size = self._config.pool_config.size
        for _ in range(min_size):
            self.__pool.add_model(hook=self._task_hook)
        print(f"[TaskManager] Preloaded {min_size} models in pool.")

        self._task_batch_thread.start()
        self._scaler_thread.start()

    def stop(self):
        """Stop TaskManager and close all models in pool."""
        self._stop_event.set()
        self._task_batch_thread.join()
        self._scaler_thread.join()

        self.__pool.close_all()
        print("[TaskManager] Stopped and all models closed.")
