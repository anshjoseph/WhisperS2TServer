import torch
from utils.log import get_configure_logger


logger = get_configure_logger(__file__)

# Whisper models VRAM requirements in GB (FP16, batch size=1)
WHISPER_VRAM = {
    "tiny": 1.0,
    "base": 1.5,
    "small": 2.8,
    "medium": 5.5,
    "large-v2": 10.2,
    "large-v3": 10.8
}

def check_model_fit(model_name: str, gpu_index=0, buffer_gb=0.2) -> bool:
    """
    Check if a given Whisper model fits in GPU memory.
    
    Parameters:
        model_name: str - Whisper model name (e.g., 'tiny', 'base', 'small', etc.)
        gpu_index: int - CUDA GPU index
        buffer_gb: float - safety buffer to avoid OOM (default 0.2 GB)
        
    Returns:
        bool: True if model fits, False otherwise
    """
    if model_name not in WHISPER_VRAM:
        raise ValueError(f"Unknown model name '{model_name}'. Valid names: {list(WHISPER_VRAM.keys())}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    device = torch.device(f"cuda:{gpu_index}")
    props = torch.cuda.get_device_properties(device)
    total_vram = props.total_memory / (1024 ** 3)
    used_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)
    free_vram = total_vram - used_vram - buffer_gb

    required_vram = WHISPER_VRAM[model_name]
    fits = required_vram <= free_vram

    logger.info(f"[GPU {gpu_index}] Model '{model_name}' requires {required_vram} GB VRAM, free VRAM available: {free_vram:.2f} GB -> Fits: {fits}")
    return fits


