"""
Video trimmer - Shorten video to specified duration
"""

import cv2
from pathlib import Path


def trim_video(input_path: str, output_path: str, duration_seconds: int = 30):
    """
    Trim video to specified duration
    
    Args:
        input_path: Path to input video
        output_path: Path to save trimmed video
        duration_seconds: Desired duration in seconds (default: 120 = 2 minutes)
    """
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    
    # Calculate target frames
    target_frames = duration_seconds * fps
    
    print(f"\n{'='*60}")
    print(f"  VIDEO TRIMMER")
    print(f"{'='*60}")
    print(f"\nVstup: {input_path}")
    print(f"Rozlišení: {width}x{height} @ {fps}fps")
    print(f"Původní délka: {total_duration:.1f}s ({total_frames} snímků)")
    print(f"Nová délka: {duration_seconds}s ({target_frames} snímků)")
    print(f"Výstup: {output_path}\n")
    
    # Create video writer
    codecs = ['avc1', 'H264', 'X264', 'mp4v']
    out = None
    
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"Použitý kodek: {codec}\n")
            break
    
    if not out or not out.isOpened():
        raise ValueError("Nelze vytvořit video writer")
    
    # Copy frames
    frame_idx = 0
    
    while frame_idx < target_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"\n⚠ Video skončilo u snímku {frame_idx}")
            break
        
        out.write(frame)
        
        if frame_idx % (fps * 5) == 0:  # Update every 5 seconds
            progress = (frame_idx / target_frames) * 100
            elapsed = frame_idx / fps
            print(f"[{progress:5.1f}%] {elapsed:.1f}s / {duration_seconds}s", end="\r")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    actual_duration = frame_idx / fps
    
    print(f"\n\n{'='*60}")
    print(f"✅ HOTOVO!")
    print(f"{'='*60}")
    print(f"\nZkrácené video: {output_path}")
    print(f"Délka: {actual_duration:.1f}s")
    print(f"Velikost: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB\n")


def main():
    """Process video from to_trim folder"""
    
    input_dir = Path("to_trim")
    output_dir = Path("input")  # Output goes to input folder for processing
    
    # Create directories
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Find first video file (MP4, AVI, MOV, MKV)
    video_files = list(input_dir.glob("*.[mM][pP]4")) + list(input_dir.glob("*.[aA][vV][iI]")) + list(input_dir.glob("*.[mM][oO][vV]")) + list(input_dir.glob("*.[mM][kK][vV]"))
    
    if not video_files:
        print("❌ Žádné video nenalezeno ve složce 'to_trim/'")
        print("Podporované formáty: MP4, AVI, MOV, MKV")
        print("\nUmístěte dlouhé video do složky 'backend/to_trim/'")
        return
    
    input_video = video_files[0]
    output_video = output_dir / f"{input_video.stem}_1min.mp4"
    
    # Trim to 1 minute (60 seconds)
    trim_video(
        input_path=str(input_video),
        output_path=str(output_video),
        duration_seconds=60  # 1 minute
    )


if __name__ == "__main__":
    main()
