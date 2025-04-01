#!/usr/bin/env python3
import cv2
import os
import platform
import numpy as np
import subprocess

def main():
    print("OpenCV Codec and Video Writer Diagnosis")
    print("=======================================")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Python Version: {platform.python_version()}")
    
    # Check if ffmpeg is installed
    print("\nChecking FFmpeg installation:")
    try:
        ffmpeg_version = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT).decode().split('\n')[0]
        print(f"  FFmpeg installed: {ffmpeg_version}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("  FFmpeg not found. Consider installing with: sudo apt-get install ffmpeg")
    
    # Try different codecs
    print("\nTesting codecs:")
    codecs = ['mp4v', 'avc1', 'XVID', 'X264', 'H264', 'MJPG', 'DIVX']
    
    # Create a simple test frame
    test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Draw a red rectangle
    cv2.rectangle(test_frame, (50, 50), (150, 150), (0, 0, 255), -1)
    
    results = []
    
    for codec in codecs:
        try:
            # Create a test file with appropriate extension based on codec
            if codec in ['XVID', 'DIVX']:
                test_file = f"test_{codec}.avi"
            else:
                test_file = f"test_{codec}.mp4"
                
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(test_file, fourcc, 30.0, (200, 200))
            
            if writer.isOpened():
                # Write a few frames to make sure it's properly created
                for _ in range(10):
                    writer.write(test_frame)
                writer.release()
                
                # Verify the file exists and is not empty
                if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
                    status = "SUCCESS"
                    # Try to read it back to verify it's valid
                    cap = cv2.VideoCapture(test_file)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            status += " (readable)"
                        else:
                            status += " (not readable)"
                        cap.release()
                    else:
                        status += " (not readable)"
                else:
                    status = "FAILED (empty file)"
                
                file_size = os.path.getsize(test_file) if os.path.exists(test_file) else 0
            else:
                status = "FAILED (writer not opened)"
                file_size = 0
        except Exception as e:
            status = f"ERROR: {str(e)}"
            file_size = 0
        
        results.append((codec, status, file_size))
        print(f"  - {codec}: {status} ({file_size} bytes)")
    
    # Also try raw uncompressed
    try:
        test_file = "test_raw.avi"
        fourcc = 0  # Uncompressed
        writer = cv2.VideoWriter(test_file, fourcc, 30.0, (200, 200))
        
        if writer.isOpened():
            writer.write(test_frame)
            writer.release()
            
            if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
                status = "SUCCESS"
                file_size = os.path.getsize(test_file)
            else:
                status = "FAILED (empty file)"
                file_size = 0
        else:
            status = "FAILED (writer not opened)"
            file_size = 0
    except Exception as e:
        status = f"ERROR: {str(e)}"
        file_size = 0
    
    results.append(("RAW (0)", status, file_size))
    print(f"  - RAW (0): {status} ({file_size} bytes)")
    
    # Clean up test files
    print("\nCleaning up test files...")
    for codec in codecs:
        ext = ".avi" if codec in ['XVID', 'DIVX'] else ".mp4"
        test_file = f"test_{codec}{ext}"
        if os.path.exists(test_file):
            os.remove(test_file)
    
    if os.path.exists("test_raw.avi"):
        os.remove("test_raw.avi")
    
    # Print recommendations
    print("\nRecommendations based on test results:")
    working_codecs = [r[0] for r in results if "SUCCESS" in r[1]]
    
    if working_codecs:
        print(f"  - Working codecs: {', '.join(working_codecs)}")
        print(f"  - Recommended codec: {working_codecs[0]}")
        print(f"  - Command line usage: --codec {working_codecs[0]}")
        
        # Specific recommendations
        if "XVID" in working_codecs:
            print("\nRecommended setup for Linux:")
            print("  ./person_tracker.py -i your_video.mp4 -o output.avi --codec XVID")
        elif "mp4v" in working_codecs:
            print("\nRecommended setup:")
            print("  ./person_tracker.py -i your_video.mp4 -o output.mp4 --codec mp4v")
        elif "RAW (0)" in working_codecs:
            print("\nRecommended setup (will create large files):")
            print("  ./person_tracker.py -i your_video.mp4 -o output.avi --codec 0")
    else:
        print("  - No working codecs found. Try installing ffmpeg:")
        print("    sudo apt-get update && sudo apt-get install ffmpeg libavcodec-dev libavformat-dev libswscale-dev")
        print("    Then reinstall opencv-python-headless:")
        print("    pip uninstall -y opencv-python-headless && pip install opencv-python-headless")
        print("    Or use uncompressed output as a last resort:")
        print("    ./person_tracker.py -i your_video.mp4 -o output.avi --codec 0")

if __name__ == "__main__":
    main() 