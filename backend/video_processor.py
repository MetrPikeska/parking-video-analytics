"""
Video processor - creates output video with bounding boxes and vehicle count
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from ultralytics import YOLO
import torch


class VideoProcessor:
    """Process video and create output with detections overlay"""
    
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle", 
        5: "bus",
        7: "truck"
    }
    
    def __init__(self, model_name: str = "yolov8n.pt"):
        """Initialize processor with YOLO model"""
        # Auto-detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = YOLO(model_name)
        self.model.to(device)
        self.device = device
        
        print(f"✓ Model: {model_name}")
        print(f"✓ Device: {device}")
        if device == "cuda":
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        confidence: float = 0.4,
        fps_sampling: int = 1,
        roi: Optional[Dict] = None,
        progress_callback=None
    ):
        """
        Process video and create output with detections
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video
            confidence: Detection confidence threshold
            fps_sampling: Process every Nth frame (1 = all frames)
            roi: Optional region of interest {x, y, width, height}
            progress_callback: Callback function(progress, message)
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer with H.264 codec (better compatibility)
        # Try different codecs in order of preference
        codecs = ['avc1', 'H264', 'X264', 'mp4v']
        out = None
        
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"Using codec: {codec}")
                break
        
        if not out or not out.isOpened():
            raise ValueError("Cannot create video writer with any codec")
        
        frame_idx = 0
        processed_count = 0
        
        print(f"\nProcessing video: {input_path}")
        print(f"Resolution: {width}x{height} @ {fps}fps")
        print(f"Total frames: {total_frames}")
        print(f"Output: {output_path}\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            vehicle_count = 0
            
            # Process only sampled frames for detection
            if frame_idx % fps_sampling == 0:
                # Run YOLO detection
                results = self.model(
                    frame,
                    conf=confidence,
                    verbose=False,
                    device=self.device
                )
                
                # Draw detections
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        
                        # Check if vehicle
                        if cls_id not in self.VEHICLE_CLASSES:
                            continue
                        
                        # Get bbox
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Check ROI
                        if roi:
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            roi_x1 = roi["x"]
                            roi_y1 = roi["y"]
                            roi_x2 = roi["x"] + roi["width"]
                            roi_y2 = roi["y"] + roi["height"]
                            
                            if not (roi_x1 <= center_x <= roi_x2 and 
                                   roi_y1 <= center_y <= roi_y2):
                                continue
                        
                        vehicle_count += 1
                        
                        # Draw bounding box (green)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        vehicle_type = self.VEHICLE_CLASSES[cls_id]
                        conf_score = float(box.conf[0])
                        label = f"{vehicle_type} {conf_score:.2f}"
                        
                        # Label background
                        (label_w, label_h), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        cv2.rectangle(
                            frame, 
                            (x1, y1 - label_h - 10), 
                            (x1 + label_w, y1), 
                            (0, 255, 0), 
                            -1
                        )
                        cv2.putText(
                            frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                        )
            
            # Draw ROI rectangle if specified
            if roi:
                cv2.rectangle(
                    frame,
                    (roi["x"], roi["y"]),
                    (roi["x"] + roi["width"], roi["y"] + roi["height"]),
                    (255, 0, 255),
                    2
                )
            
            # Draw counter in top-left corner
            counter_text = f"Pocet aut: {vehicle_count}"
            
            # Black background for better visibility
            (text_w, text_h), _ = cv2.getTextSize(
                counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
            )
            cv2.rectangle(frame, (10, 10), (30 + text_w, 50 + text_h), (0, 0, 0), -1)
            
            # White text
            cv2.putText(
                frame, counter_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
            )
            
            # Write frame to output
            out.write(frame)
            
            # Progress update
            if progress_callback and frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                progress_callback(progress, f"Frame {frame_idx}/{total_frames}")
            
            frame_idx += 1
            if frame_idx % fps_sampling == 0:
                processed_count += 1
        
        cap.release()
        out.release()
        
        print(f"\n✓ Processing complete!")
        print(f"Processed {processed_count} frames")
        print(f"Output saved: {output_path}")


if __name__ == "__main__":
    # Example usage
    processor = VideoProcessor("yolov8n.pt")
    
    # Process video
    processor.process_video(
        input_path="input/video.mp4",
        output_path="output/result.mp4",
        confidence=0.4,
        fps_sampling=1,  # Process every frame
        roi=None  # No ROI restriction
    )
