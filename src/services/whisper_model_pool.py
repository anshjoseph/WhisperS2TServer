from libs.whisper_audio_processor import WhisperAudioProcessor, Task, TaskStatus, BatchTask
from config import get_config
from utils.log import get_configure_logger
from typing import Dict

logger = get_configure_logger()

class WhisperModelPool:
    def __init__(self):
        self.__pool : Dict[str, WhisperAudioProcessor] = dict()
    

    def _add_model(self):
        pass

    def add_batch(self, batch:BatchTask):
        pass
    