"""
Simple script to process parking video and create output with detections
"""

from video_processor import VideoProcessor
from pathlib import Path


def main():
    """Process video from input folder and save to output folder"""
    
    # Setup paths
    input_dir = Path("input")
    output_dir = Path("output")
    
    # Create directories
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Find first video file in input directory
    video_files = list(input_dir.glob("*.[mM][pP]4")) + list(input_dir.glob("*.[aA][vV][iI]")) + list(input_dir.glob("*.[mM][oO][vV]")) + list(input_dir.glob("*.[mM][kK][vV]"))
    
    if not video_files:
        print("❌ Žádné video nenalezeno ve složce 'input/'")
        print("Umístěte video soubor do složky 'backend/input/'")
        return
    
    input_video = video_files[0]
    output_video = output_dir / f"{input_video.stem}_analyzed.mp4"
    
    print("=" * 60)
    print("  PARKING VIDEO ANALYTICS - Procesování")
    print("=" * 60)
    print(f"\nVstupní video: {input_video}")
    print(f"Výstupní video: {output_video}\n")
    
    # Initialize processor
    processor = VideoProcessor(model_name="yolov8n.pt")
    
    # Progress callback
    def show_progress(progress, message):
        print(f"[{progress:5.1f}%] {message}", end="\r")
    
    # Process video
    processor.process_video(
        input_path=str(input_video),
        output_path=str(output_video),
        confidence=0.5,  # Vyšší confidence pro lepší detekci aut
        fps_sampling=1,  # Analyze every frame (change to 2-5 for faster processing)
        roi=None,  # No ROI filter (analyze whole frame)
        progress_callback=show_progress
    )
    
    print("\n" + "=" * 60)
    print("✅ HOTOVO!")
    print("=" * 60)
    print(f"\nVýstupní video s detekcemi: {output_video}")
    print("\nVideo obsahuje:")
    print("  • Zelené obdélníky kolem detekovaných aut")
    print("  • Počet aut v levém horním rohu")
    print("  • Labels s typem vozidla a confidence")


if __name__ == "__main__":
    main()
