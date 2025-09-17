from libs.whisper_audio_processor import BatchTask, WhisperAudioProcessor, Task
import time
import threading
import uuid

def batch_completed_hook(batch: BatchTask):
    """Example hook function called when a batch is completed."""
    print(f"\nBatch {batch.batch_id} completed with status: {batch.status}")
    print(f"Progress: {batch.progress:.1f}% ({batch.completed_tasks}/{batch.total_tasks})")
    for task in batch.tasks:
        print(f"  Task {task.task_id}: {task.status} {task.output}")
        if task.error:
            print(f"    Error: {task.error}")

def auto_batch_adder(processor: WhisperAudioProcessor, interval: int = 10):
    """Automatically adds a new batch every `interval` seconds."""
    batch_count = 0
    while True:
        batch_count += 1
        batch_id = f"auto_batch_{batch_count:03}_{uuid.uuid4().hex[:4]}"
        tasks = []
        # Example: 2 tasks per batch
        for i in range(1, 3):
            task_id = f"{batch_id}_task_{i:03}"
            tasks.append(Task(
                task_id=task_id,
                file_path=f"./test/test1.wav",  # Replace with actual files
                task="transcribe",
                lang="en"
            ))
        batch = BatchTask(batch_id=batch_id, start_time=time.time(), tasks=tasks)
        if processor.add_batch(batch):
            print(f"Batch {batch_id} added successfully")
        time.sleep(interval)

if __name__ == "__main__":
    with WhisperAudioProcessor(batch_completed_hook) as processor:
        # Start auto batch adder in a separate thread
        threading.Thread(target=auto_batch_adder, args=(processor, 10), daemon=True).start()

        # Monitor processing
        try:
            while True:
                stats = processor.get_stats()
                print(f"Last activity: {stats['is_busy']}")
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping processor monitoring.")
