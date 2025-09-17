import os
import tempfile
import time
import uuid
from typing import Optional, Dict
from pathlib import Path
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from services.task_manager import TaskManager
from models.batch_task import Task
from utils.log import get_configure_logger

# Configure logging
logger = get_configure_logger(__file__)

# Pydantic models for API responses
class TranscriptionResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Task] = None
    error: Optional[dict] = None
    processing_time: Optional[float] = None
    created_at: str

class SystemStats(BaseModel):
    is_running: bool
    total_tasks_submitted: int
    total_batches_created: int
    pending_queue_size: int
    pool_stats: Dict

# Global variables
app = FastAPI(
    title="Audio Transcription Service",
    description="Upload .wav files and get AI-powered transcriptions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global TaskManager instance
task_manager: Optional[TaskManager] = None

# Supported audio file extensions - only .wav files
SUPPORTED_EXTENSIONS = {'.wav'}

class TranscriptionService:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="transcription_")
        logger.info(f"Created temporary directory: {self.temp_dir}")
    
    def cleanup_temp_dir(self):
        """Clean up temporary directory"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Failed to cleanup temp directory: {str(e)}")
    
    def save_uploaded_file(self, file: UploadFile) -> str:
        """Save uploaded file to temporary directory"""
        file_extension = Path(file.filename).suffix.lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(self.temp_dir, unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved uploaded file: {file_path}")
        return file_path
    
    def validate_audio_file(self, filename: str) -> bool:
        """Validate if file is a supported audio format"""
        file_extension = Path(filename).suffix.lower()
        return file_extension in SUPPORTED_EXTENSIONS

# Global service instance
transcription_service = TranscriptionService()

@app.on_event("startup")
async def startup_event():
    """Initialize TaskManager on startup"""
    global task_manager
    try:
        task_manager = TaskManager()
        if not task_manager.start():
            raise RuntimeError("Failed to start TaskManager")
        logger.info("TaskManager started successfully")
    except Exception as e:
        logger.error(f"Failed to start TaskManager: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global task_manager
    if task_manager:
        task_manager.stop()
        logger.info("TaskManager stopped")
    
    transcription_service.cleanup_temp_dir()
    logger.info("Application shutdown complete")

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Audio Transcription Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "transcribe": "/transcribe-simple (upload .wav file, get transcription text)",
            "transcribe_detailed": "/transcribe (upload .wav file, get detailed response)",
            "stats": "/stats (get system statistics)",
            "health": "/health (health check)"
        },
        "supported_formats": ".wav files only"
    }



@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_detailed(
    file: UploadFile = File(...),
    language: str = Query(default="en", description="Language code for transcription"),
    prompt: str = Query(default="", description="Transcription of previous segment")
):
    """Detailed transcription endpoint - upload .wav file and get detailed response with metadata"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not transcription_service.validate_audio_file(file.filename):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    
    # Check file size (limit to 100MB)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > 100 * 1024 * 1024:  # 100MB
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 100MB")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    start_time = time.time()
    file_path = None
    
    try:
        # Save uploaded file
        file_path = transcription_service.save_uploaded_file(file)
        
        logger.info(f"Starting detailed transcription for file: {file.filename}")
        
        # Create Task object
        task = Task(
            task_id=task_id,
            file_path=file_path,
            task="transcribe",
            lang=language,
            prompt=prompt
        )
        
        # Submit to TaskManager and get result
        result = await task_manager.submit_task(task)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"Detailed transcription completed for file: {file.filename} in {processing_time:.2f}s")
        
        # Return detailed response
        return TranscriptionResponse(
            task_id=task_id,
            status="completed",
            result=result,
            error=result.error,
            processing_time=processing_time,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        logger.error(f"Detailed transcription failed: {str(e)}")
        
        # Return error response
        return TranscriptionResponse(
            task_id=task_id,
            status="failed",
            result=None,
            error=str(e),
            processing_time=processing_time,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        )
        
    finally:
        # Clean up temporary file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup file: {str(e)}")

@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics from TaskManager"""
    global task_manager
    
    if not task_manager:
        raise HTTPException(status_code=503, detail="TaskManager not available")
    
    try:
        stats = task_manager.get_stats()
        return SystemStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global task_manager
    
    if not task_manager:
        raise HTTPException(status_code=503, detail="TaskManager not available")
    
    try:
        stats = task_manager.get_stats()
        return {
            "status": "healthy",
            "task_manager_running": stats.get("is_running", False),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )