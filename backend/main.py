"""
FastAPI backend for parking video analytics
Detects and counts vehicles in video using YOLOv8
"""

import os
import uuid
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import aiofiles

from video_analyzer import VideoAnalyzer, AnalysisConfig
from job_manager import JobManager

# Initialize FastAPI app
app = FastAPI(title="Parking Video Analytics API")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("./uploads")
RESULTS_DIR = Path("./results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Job manager (singleton)
job_manager = JobManager()


# Pydantic models
class AnalyzeRequest(BaseModel):
    fps_sampling: int = 5
    confidence_threshold: float = 0.4
    roi: Optional[Dict[str, int]] = None  # {x, y, width, height}
    vehicle_types: List[str] = ["car", "truck", "bus", "motorcycle"]


class JobStatus(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float
    message: str
    created_at: str
    completed_at: Optional[str] = None


class AnalysisResult(BaseModel):
    job_id: str
    video_info: Dict
    statistics: Dict
    timeline: List[Dict]  # [{frame: int, timestamp: float, count: int}]
    processing_time: float


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Parking Video Analytics API",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/api/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    fps_sampling: int = 5,
    confidence_threshold: float = 0.4,
    roi_x: Optional[int] = None,
    roi_y: Optional[int] = None,
    roi_width: Optional[int] = None,
    roi_height: Optional[int] = None,
):
    """
    Upload video and start analysis
    Returns job_id for tracking progress
    """
    # Validate file
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format. Use MP4, AVI, MOV, or MKV")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    video_path = UPLOAD_DIR / f"{job_id}.mp4"
    async with aiofiles.open(video_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Parse ROI
    roi = None
    if all([roi_x is not None, roi_y is not None, roi_width, roi_height]):
        roi = {
            "x": roi_x,
            "y": roi_y,
            "width": roi_width,
            "height": roi_height
        }
    
    # Create analysis config
    config = AnalysisConfig(
        fps_sampling=fps_sampling,
        confidence_threshold=confidence_threshold,
        roi=roi,
        vehicle_types=["car", "truck", "bus", "motorcycle"]
    )
    
    # Queue job
    job_manager.create_job(job_id, str(video_path), config)
    
    return {
        "job_id": job_id,
        "message": "Video uploaded successfully. Analysis started.",
        "status": "queued"
    }


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get current job status and progress"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at")
    )


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """Get analysis results for completed job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")
    
    # Load results from file
    result_path = RESULTS_DIR / f"{job_id}.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    
    async with aiofiles.open(result_path, 'r') as f:
        content = await f.read()
        results = json.loads(content)
    
    return results


@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time progress updates
    """
    await websocket.accept()
    
    try:
        while True:
            job = job_manager.get_job(job_id)
            if not job:
                await websocket.send_json({
                    "error": "Job not found",
                    "job_id": job_id
                })
                break
            
            # Send current status
            await websocket.send_json({
                "job_id": job_id,
                "status": job["status"],
                "progress": job["progress"],
                "message": job["message"]
            })
            
            # If job is completed or failed, close connection
            if job["status"] in ["completed", "failed"]:
                break
            
            # Wait before next update
            await asyncio.sleep(0.5)
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Delete job and associated files"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete video file
    video_path = Path(job["video_path"])
    if video_path.exists():
        video_path.unlink()
    
    # Delete results file
    result_path = RESULTS_DIR / f"{job_id}.json"
    if result_path.exists():
        result_path.unlink()
    
    # Remove job from manager
    job_manager.delete_job(job_id)
    
    return {"message": "Job deleted successfully"}


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs"""
    return job_manager.list_jobs()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
