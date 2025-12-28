"""
Advanced Video Processor for Parking Lot Monitoring
Optimized for static cameras, winter conditions, and maximum detection accuracy

Features:
- High-resolution inference
- Image tiling for small/distant objects
- ROI masking
- Optimized NMS settings
- Accuracy-first approach
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from ultralytics import YOLO
import torch
from dataclasses import dataclass


@dataclass
class DetectionConfig:
    """Configuration for advanced detection"""
    # Model settings
    model_name: str = "yolov8n.pt"  # Can use yolov8s.pt, yolov8m.pt for better accuracy
    input_size: int = 1280  # Higher resolution (default: 640, max: 1920)
    
    # Detection thresholds
    confidence: float = 0.3  # Lower threshold catches more vehicles
    iou_threshold: float = 0.3  # Lower = less aggressive NMS, keeps more overlapping boxes
    
    # Tiling settings
    enable_tiling: bool = True
    tile_size: int = 640  # Size of each tile
    tile_overlap: float = 0.2  # 20% overlap between tiles
    
    # ROI settings
    roi_mask: Optional[np.ndarray] = None  # Binary mask (1=analyze, 0=ignore)
    
    # Processing
    fps_sampling: int = 1  # Process every Nth frame
    
    # Vehicle classes (COCO dataset IDs)
    vehicle_classes: set = None
    
    def __post_init__(self):
        if self.vehicle_classes is None:
            self.vehicle_classes = {2}  # car only


class AdvancedVideoProcessor:
    """
    Advanced processor optimized for parking lot vehicle detection
    
    WHY THESE OPTIMIZATIONS WORK:
    
    1. HIGH RESOLUTION INFERENCE (1280px):
       - Small/distant cars occupy more pixels
       - Better feature extraction for YOLO
       - Critical for parking lots with deep perspective
    
    2. IMAGE TILING:
       - Splits large frame into overlapping patches
       - Each patch processed at high resolution
       - Small objects become larger relative to tile size
       - Essential for detecting cars 50+ meters away
    
    3. LOWER CONFIDENCE THRESHOLD (0.3):
       - Catches partially occluded vehicles
       - Detects low-contrast cars (white car on snow)
       - Static camera = low false positive risk
    
    4. ADJUSTED NMS (IoU 0.3):
       - Less aggressive non-maximum suppression
       - Keeps detections of closely parked cars
       - Prevents merging of adjacent vehicles
    
    5. ROI MASKING:
       - Ignores sky, trees, surrounding areas
       - Reduces false positives
       - Focuses compute on parking area
    
    6. STATIC CAMERA OPTIMIZATION:
       - No need for tracking (cars don't move much)
       - Can use aggressive detection settings
       - Temporal consistency not critical
    """
    
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }
    
    def __init__(self, config: DetectionConfig):
        """Initialize advanced processor"""
        self.config = config
        
        # Auto-detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        self.model = YOLO(config.model_name)
        self.model.to(self.device)
        
        print("="*60)
        print("  ADVANCED PARKING LOT DETECTOR")
        print("="*60)
        print(f"\n✓ Model: {config.model_name}")
        print(f"✓ Device: {self.device}")
        if self.device == "cuda":
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ Input resolution: {config.input_size}px")
        print(f"✓ Tiling: {'Enabled' if config.enable_tiling else 'Disabled'}")
        if config.enable_tiling:
            print(f"  - Tile size: {config.tile_size}px")
            print(f"  - Overlap: {config.tile_overlap*100:.0f}%")
        print(f"✓ Confidence threshold: {config.confidence}")
        print(f"✓ NMS IoU threshold: {config.iou_threshold}")
        print(f"✓ ROI mask: {'Yes' if config.roi_mask is not None else 'No'}")
        print()
    
    def create_tiles(self, image: np.ndarray) -> List[Tuple[np.ndarray, int, int]]:
        """
        Split image into overlapping tiles
        
        Returns:
            List of (tile_image, offset_x, offset_y)
        """
        h, w = image.shape[:2]
        tile_size = self.config.tile_size
        overlap = int(tile_size * self.config.tile_overlap)
        stride = tile_size - overlap
        
        tiles = []
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Handle edge cases
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)
                
                tile = image[y_start:y_end, x_start:x_end]
                tiles.append((tile, x_start, y_start))
        
        return tiles
    
    def detect_in_tile(self, tile: np.ndarray, offset_x: int, offset_y: int) -> List[Dict]:
        """
        Run detection on a single tile and convert to global coordinates
        
        Returns:
            List of detections with global coordinates
        """
        results = self.model(
            tile,
            conf=self.config.confidence,
            iou=self.config.iou_threshold,
            imgsz=self.config.input_size,
            verbose=False,
            device=self.device
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                
                # Filter by vehicle class
                if cls_id not in self.config.vehicle_classes:
                    continue
                
                # Get bbox in tile coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to global coordinates
                x1_global = int(x1 + offset_x)
                y1_global = int(y1 + offset_y)
                x2_global = int(x2 + offset_x)
                y2_global = int(y2 + offset_y)
                
                conf = float(box.conf[0])
                
                detections.append({
                    'bbox': (x1_global, y1_global, x2_global, y2_global),
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': self.VEHICLE_CLASSES.get(cls_id, 'unknown')
                })
        
        return detections
    
    def merge_detections(self, all_detections: List[Dict]) -> List[Dict]:
        """
        Merge overlapping detections from different tiles using NMS
        
        This is critical because tiling creates duplicate detections
        at tile boundaries.
        """
        if not all_detections:
            return []
        
        # Convert to format for NMS
        boxes = []
        scores = []
        
        for det in all_detections:
            x1, y1, x2, y2 = det['bbox']
            boxes.append([x1, y1, x2, y2])
            scores.append(det['confidence'])
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.config.confidence,
            self.config.iou_threshold
        )
        
        # Return filtered detections
        if len(indices) > 0:
            indices = indices.flatten()
            return [all_detections[i] for i in indices]
        
        return []
    
    def apply_roi_filter(self, detections: List[Dict]) -> List[Dict]:
        """
        Filter detections based on ROI mask
        
        Only keep detections where center point is in ROI
        """
        if self.config.roi_mask is None:
            return detections
        
        filtered = []
        mask = self.config.roi_mask
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Check if center is in ROI
            if 0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1]:
                if mask[center_y, center_x] > 0:
                    filtered.append(det)
        
        return filtered
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Main detection pipeline
        
        Process:
        1. Create tiles (if enabled)
        2. Run detection on each tile
        3. Convert to global coordinates
        4. Merge overlapping detections (NMS)
        5. Apply ROI filtering
        
        Returns:
            List of final detections
        """
        all_detections = []
        
        if self.config.enable_tiling:
            # Tiled detection
            tiles = self.create_tiles(frame)
            
            for tile, offset_x, offset_y in tiles:
                tile_detections = self.detect_in_tile(tile, offset_x, offset_y)
                all_detections.extend(tile_detections)
            
            # Merge overlapping detections from different tiles
            all_detections = self.merge_detections(all_detections)
        else:
            # Full-frame detection
            all_detections = self.detect_in_tile(frame, 0, 0)
        
        # Apply ROI filtering
        all_detections = self.apply_roi_filter(all_detections)
        
        return all_detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det['class_name']
            
            # Draw bounding box (green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            text = f"{label} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Label background
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                (0, 255, 0),
                -1
            )
            
            # Label text
            cv2.putText(
                frame, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        
        return frame
    
    def draw_counter(self, frame: np.ndarray, count: int) -> np.ndarray:
        """Draw vehicle counter overlay"""
        counter_text = f"Pocet aut: {count}"
        
        # Black background
        (text_w, text_h), _ = cv2.getTextSize(
            counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
        )
        cv2.rectangle(frame, (10, 10), (30 + text_w, 50 + text_h), (0, 0, 0), -1)
        
        # White text
        cv2.putText(
            frame, counter_text, (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
        )
        
        return frame
    
    def draw_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """Draw ROI mask overlay (semi-transparent)"""
        if self.config.roi_mask is None:
            return frame
        
        # Create colored overlay
        overlay = frame.copy()
        mask_color = (255, 0, 255)  # Magenta
        
        # Draw ROI boundary
        contours, _ = cv2.findContours(
            self.config.roi_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        cv2.drawContours(overlay, contours, -1, mask_color, 2)
        
        # Blend
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        return frame
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback=None
    ):
        """
        Process entire video with advanced detection
        
        Args:
            input_path: Input video path
            output_path: Output video path
            progress_callback: Optional callback(progress, message)
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create ROI mask if not provided (analyze full frame)
        if self.config.roi_mask is None:
            self.config.roi_mask = np.ones((height, width), dtype=np.uint8)
        
        # Create video writer
        codecs = ['avc1', 'H264', 'X264', 'mp4v']
        out = None
        
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"Using codec: {codec}\n")
                break
        
        if not out or not out.isOpened():
            raise ValueError("Cannot create video writer")
        
        print(f"Processing video: {input_path}")
        print(f"Resolution: {width}x{height} @ {fps}fps")
        print(f"Total frames: {total_frames}")
        print(f"Output: {output_path}\n")
        
        frame_idx = 0
        processed_count = 0
        last_detection_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only sampled frames
            if frame_idx % self.config.fps_sampling == 0:
                # Detect vehicles
                detections = self.detect_vehicles(frame)
                last_detection_count = len(detections)
                
                # Draw detections
                frame = self.draw_detections(frame, detections)
                frame = self.draw_counter(frame, len(detections))
                
                processed_count += 1
            
            # Progress update (always show, not just with callback)
            if frame_idx % 10 == 0:  # Update every 10 frames
                progress = (frame_idx / total_frames) * 100
                elapsed_sec = frame_idx / fps
                total_sec = total_frames / fps
                msg = f"[{progress:5.1f}%] Frame {frame_idx}/{total_frames} ({elapsed_sec:.1f}s/{total_sec:.1f}s) | Cars: {last_detection_count}"
                print(msg, end="\r")
                
                # Also call callback if provided
                if progress_callback:
                    progress_callback(progress, msg)
            
            # Write frame
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        print(f"\n\n{'='*60}")
        print("✅ PROCESSING COMPLETE")
        print("="*60)
        print(f"Processed {processed_count} frames")
        print(f"Output: {output_path}\n")


def create_roi_mask_from_polygon(frame_shape: Tuple[int, int], points: List[Tuple[int, int]]) -> np.ndarray:
    """
    Create ROI mask from polygon points
    
    Args:
        frame_shape: (height, width) of video frame
        points: List of (x, y) polygon vertices
        
    Returns:
        Binary mask (1=ROI, 0=ignore)
        
    Example:
        # Define parking lot area (avoiding trees, sky)
        roi_points = [
            (100, 400),   # Bottom-left
            (100, 200),   # Top-left
            (1500, 200),  # Top-right
            (1500, 800),  # Bottom-right
        ]
        mask = create_roi_mask_from_polygon((1080, 1920), roi_points)
    """
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    points_array = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 1)
    
    return mask


# Example usage
if __name__ == "__main__":
    # Create advanced configuration
    config = DetectionConfig(
        model_name="yolov8n.pt",  # Use yolov8s.pt or yolov8m.pt for better accuracy
        input_size=1280,  # High resolution
        confidence=0.3,  # Lower threshold
        iou_threshold=0.3,  # Less aggressive NMS
        enable_tiling=True,  # Enable tiling for small objects
        tile_size=640,
        tile_overlap=0.2,
        fps_sampling=1,  # Process every frame
        vehicle_classes={2}  # Car only
    )
    
    # Optional: Create ROI mask to ignore sky/trees/fields
    # Example for 1920x1080 frame
    # roi_points = [
    #     (0, 300),      # Top-left (ignore sky)
    #     (1920, 300),   # Top-right
    #     (1920, 1080),  # Bottom-right
    #     (0, 1080)      # Bottom-left
    # ]
    # config.roi_mask = create_roi_mask_from_polygon((1080, 1920), roi_points)
    
    # Create processor
    processor = AdvancedVideoProcessor(config)
    
    # Process video
    processor.process_video(
        input_path="input/opalena.mov",
        output_path="output/opalena_advanced.mp4"
    )
