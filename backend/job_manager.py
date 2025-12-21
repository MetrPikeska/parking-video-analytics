"""
Job manager for background video processing
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from video_analyzer import VideoAnalyzer, AnalysisConfig


class JobManager:
    """Manages analysis jobs and background processing"""
    
    def __init__(self, max_workers: int = 1):
        """
        Initialize job manager
        
        Args:
            max_workers: Max concurrent jobs (1 for sequential processing)
        """
        self.jobs: Dict[str, Dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.analyzer = None
        self.results_dir = Path("./results")
        self.results_dir.mkdir(exist_ok=True)
    
    def _get_analyzer(self) -> VideoAnalyzer:
        """Get or create analyzer instance (lazy loading)"""
        if self.analyzer is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.analyzer = VideoAnalyzer(model_name="yolov8n.pt", device=device)
        return self.analyzer
    
    def create_job(self, job_id: str, video_path: str, config: AnalysisConfig):
        """Create and start a new analysis job"""
        self.jobs[job_id] = {
            "job_id": job_id,
            "video_path": video_path,
            "config": config,
            "status": "queued",
            "progress": 0.0,
            "message": "Job queued",
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }
        
        # Submit job to executor
        self.executor.submit(self._process_job, job_id)
    
    def _process_job(self, job_id: str):
        """Process a job (runs in thread pool)"""
        try:
            job = self.jobs[job_id]
            job["status"] = "processing"
            job["message"] = "Initializing..."
            
            # Progress callback
            def progress_callback(progress: float, message: str):
                self.jobs[job_id]["progress"] = round(progress, 1)
                self.jobs[job_id]["message"] = message
            
            # Run analysis
            analyzer = self._get_analyzer()
            results = analyzer.analyze_video(
                job["video_path"],
                job["config"],
                progress_callback
            )
            
            # Save results to file
            result_path = self.results_dir / f"{job_id}.json"
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Update job status
            job["status"] = "completed"
            job["progress"] = 100.0
            job["message"] = "Analysis completed successfully"
            job["completed_at"] = datetime.now().isoformat()
            job["result_path"] = str(result_path)
            
        except Exception as e:
            # Handle errors
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["message"] = f"Error: {str(e)}"
            self.jobs[job_id]["completed_at"] = datetime.now().isoformat()
            print(f"Job {job_id} failed: {e}")
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job status and info"""
        return self.jobs.get(job_id)
    
    def delete_job(self, job_id: str):
        """Delete a job"""
        if job_id in self.jobs:
            del self.jobs[job_id]
    
    def list_jobs(self) -> list:
        """List all jobs"""
        return [
            {
                "job_id": job_id,
                "status": job["status"],
                "progress": job["progress"],
                "created_at": job["created_at"]
            }
            for job_id, job in self.jobs.items()
        ]
