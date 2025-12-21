"""
Video analyzer using YOLOv8 for vehicle detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
from ultralytics import YOLO


@dataclass
class AnalysisConfig:
    """Configuration for video analysis"""
    fps_sampling: int = 5
    confidence_threshold: float = 0.4
    roi: Optional[Dict[str, int]] = None
    vehicle_types: List[str] = None
    
    def __post_init__(self):
        if self.vehicle_types is None:
            self.vehicle_types = ["car", "truck", "bus", "motorcycle"]


class VideoAnalyzer:
    """Analyzes video for vehicle detection and counting"""
    
    # COCO class names for vehicles
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle", 
        5: "bus",
        7: "truck"
    }
    
    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cuda"):
        """
        Initialize video analyzer
        
        Args:
            model_name: YOLO model name (yolov8n.pt, yolov8s.pt, etc.)
            device: Device to use ('cuda' or 'cpu')
        """
        import torch
        
        # Auto-detect CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA not available, falling back to CPU")
            device = "cpu"
        
        self.model = YOLO(model_name)
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        print(f"✓ Model loaded: {model_name}")
        print(f"✓ Device: {device}")
        if device == "cuda":
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    def get_video_info(self, video_path: str) -> Dict:
        """Extract video metadata"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "resolution": f"{width}x{height}"
        }
    
    def is_in_roi(self, bbox: Tuple[int, int, int, int], roi: Dict[str, int]) -> bool:
        """Check if bounding box center is inside ROI"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        roi_x1 = roi["x"]
        roi_y1 = roi["y"]
        roi_x2 = roi["x"] + roi["width"]
        roi_y2 = roi["y"] + roi["height"]
        
        return (roi_x1 <= center_x <= roi_x2 and 
                roi_y1 <= center_y <= roi_y2)
    
    def analyze_video(
        self, 
        video_path: str, 
        config: AnalysisConfig,
        progress_callback=None
    ) -> Dict:
        """
        Analyze video and count vehicles
        
        Args:
            video_path: Path to video file
            config: Analysis configuration
            progress_callback: Callback function(progress: float, message: str)
        
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        
        # Get video info
        video_info = self.get_video_info(video_path)
        
        cap = cv2.VideoCapture(video_path)
        fps = video_info["fps"]
        frame_count = video_info["frame_count"]
        
        # Calculate frame skip
        frame_skip = max(1, fps // config.fps_sampling)
        
        timeline = []
        frame_idx = 0
        processed_frames = 0
        
        if progress_callback:
            progress_callback(0, "Starting analysis...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for sampling
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # Run YOLO detection
            results = self.model(
                frame,
                conf=config.confidence_threshold,
                verbose=False,
                device=self.device
            )
            
            # Count vehicles
            vehicle_count = 0
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    
                    # Check if it's a vehicle class
                    if cls_id not in self.VEHICLE_CLASSES:
                        continue
                    
                    vehicle_type = self.VEHICLE_CLASSES[cls_id]
                    
                    # Check if this vehicle type should be counted
                    if vehicle_type not in config.vehicle_types:
                        continue
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    
                    # Check ROI if specified
                    if config.roi and not self.is_in_roi(bbox, config.roi):
                        continue
                    
                    # Count this vehicle
                    vehicle_count += 1
                    detections.append({
                        "type": vehicle_type,
                        "bbox": bbox,
                        "confidence": float(box.conf[0])
                    })
            
            # Record timeline data
            timestamp = frame_idx / fps
            timeline.append({
                "frame": frame_idx,
                "timestamp": round(timestamp, 2),
                "count": vehicle_count,
                "detections": detections
            })
            
            # Update progress
            processed_frames += 1
            progress = (frame_idx / frame_count) * 100
            
            if progress_callback and processed_frames % 10 == 0:
                progress_callback(
                    progress,
                    f"Processing frame {frame_idx}/{frame_count}"
                )
            
            frame_idx += 1
        
        cap.release()
        
        # Calculate statistics
        counts = [t["count"] for t in timeline]
        statistics = {
            "total_frames_analyzed": len(timeline),
            "max_vehicles": max(counts) if counts else 0,
            "min_vehicles": min(counts) if counts else 0,
            "avg_vehicles": round(sum(counts) / len(counts), 2) if counts else 0,
            "peak_frame": timeline[counts.index(max(counts))]["frame"] if counts else 0,
            "peak_timestamp": timeline[counts.index(max(counts))]["timestamp"] if counts else 0
        }
        
        processing_time = time.time() - start_time
        
        if progress_callback:
            progress_callback(100, "Analysis completed!")
        
        return {
            "video_info": video_info,
            "config": {
                "fps_sampling": config.fps_sampling,
                "confidence_threshold": config.confidence_threshold,
                "roi": config.roi,
                "vehicle_types": config.vehicle_types
            },
            "statistics": statistics,
            "timeline": timeline,
            "processing_time": round(processing_time, 2)
        }


if __name__ == "__main__":
    # Test
    analyzer = VideoAnalyzer(device="cuda")
    config = AnalysisConfig(fps_sampling=5, confidence_threshold=0.4)
    
    def progress_cb(progress, message):
        print(f"[{progress:.1f}%] {message}")
    
    results = analyzer.analyze_video("test.mp4", config, progress_cb)
    print(results["statistics"])
