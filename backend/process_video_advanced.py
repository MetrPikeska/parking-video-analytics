"""
Run advanced parking lot detection
Usage: python process_video_advanced.py
"""

from pathlib import Path
from advanced_video_processor import AdvancedVideoProcessor, DetectionConfig, create_roi_mask_from_polygon


def main():
    """Process video with advanced detection pipeline"""
    
    input_dir = Path("input")
    output_dir = Path("output")
    
    # Create directories
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Find first video file
    video_files = (
        list(input_dir.glob("*.[mM][pP]4")) + 
        list(input_dir.glob("*.[aA][vV][iI]")) + 
        list(input_dir.glob("*.[mM][oO][vV]")) + 
        list(input_dir.glob("*.[mM][kK][vV]"))
    )
    
    if not video_files:
        print("❌ Žádné video nenalezeno ve složce 'input/'")
        print("Podporované formáty: MP4, AVI, MOV, MKV")
        return
    
    input_video = video_files[0]
    output_video = output_dir / f"{input_video.stem}_advanced.mp4"
    
    print("\n" + "="*60)
    print("  ADVANCED PARKING LOT DETECTION")
    print("="*60)
    print(f"\nVstupní video: {input_video}")
    print(f"Výstupní video: {output_video}\n")
    
    # ========================================================================
    # CONFIGURATION - Adjust these settings for your parking lot
    # ========================================================================
    
    config = DetectionConfig(
        # Model: yolov8n.pt (fast) | yolov8s.pt (balanced) | yolov8m.pt (accurate)
        model_name="yolov8n.pt",
        
        # Input resolution: Higher = better detection of small/distant cars
        # 640 (default) | 1280 (recommended) | 1920 (max, slower)
        input_size=1280,
        
        # Confidence threshold: Lower = more detections (including uncertain ones)
        # 0.2-0.4 recommended for parking lots
        confidence=0.3,
        
        # IoU threshold for NMS: Lower = keeps more close detections
        # 0.3-0.4 recommended for dense parking
        iou_threshold=0.3,
        
        # TILING: Essential for detecting small/distant vehicles
        enable_tiling=True,
        tile_size=640,  # Size of each tile
        tile_overlap=0.2,  # 20% overlap to avoid missing cars at boundaries
        
        # FPS sampling: 1 = every frame | 2 = every 2nd frame (faster)
        fps_sampling=1,
        
        # Vehicle classes: {2} = car only | {2,3,5,7} = car, motorcycle, bus, truck
        vehicle_classes={2}  # Car only
    )
    
    # ========================================================================
    # OPTIONAL: ROI MASK
    # Uncomment and adjust to ignore sky, trees, fields, etc.
    # ========================================================================
    
    # Example: Ignore top 300 pixels (sky/trees)
    # Get first frame to determine resolution
    # import cv2
    # cap = cv2.VideoCapture(str(input_video))
    # ret, frame = cap.read()
    # cap.release()
    # 
    # if ret:
    #     h, w = frame.shape[:2]
    #     roi_points = [
    #         (0, 300),      # Top-left (start below sky)
    #         (w, 300),      # Top-right
    #         (w, h),        # Bottom-right
    #         (0, h)         # Bottom-left
    #     ]
    #     config.roi_mask = create_roi_mask_from_polygon((h, w), roi_points)
    #     print(f"✓ ROI mask applied: ignoring top 300px\n")
    
    # ========================================================================
    # PROCESS VIDEO
    # ========================================================================
    
    processor = AdvancedVideoProcessor(config)
    
    processor.process_video(
        input_path=str(input_video),
        output_path=str(output_video)
    )
    
    print("\n" + "="*60)
    print("✅ HOTOVO!")
    print("="*60)
    print(f"\nVýstupní video: {output_video}")
    print("\nVýhody pokročilé detekce:")
    print("  • Vyšší rozlišení inference (1280px)")
    print("  • Image tiling pro malé/vzdálené objekty")
    print("  • Nižší confidence threshold (0.3)")
    print("  • Optimalizované NMS nastavení")
    print("  • Volitelné ROI maskování")
    print("\nOčekávejte VÍCE detekcí než u základního procesu!\n")


if __name__ == "__main__":
    main()
