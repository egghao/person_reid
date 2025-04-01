#!/usr/bin/env python3
import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import torch
from collections import defaultdict
import platform

def parse_arguments():
    parser = argparse.ArgumentParser(description="Person tracking using Ultralytics")
    parser.add_argument(
        "--input_video", "-i", 
        type=str, 
        required=True, 
        help="Path to input video file"
    )
    parser.add_argument(
        "--output_video", "-o", 
        type=str, 
        default=None, 
        help="Path to output video file (default: input_filename_tracked.mp4)"
    )
    parser.add_argument(
        "--model", "-m", 
        type=str, 
        default="yolov8n.pt", 
        help="Path to YOLOv8 model or model name (default: yolov8n.pt). YOLOv3 to YOLOv12 are also supported."
    )
    parser.add_argument(
        "--confidence", "-c", 
        type=float, 
        default=0.3, 
        help="Detection confidence threshold (default: 0.3)"
    )
    parser.add_argument(
        "--iou", 
        type=float, 
        default=0.5, 
        help="IOU threshold for NMS (default: 0.5)"
    )
    parser.add_argument(
        "--device", "-d", 
        type=str, 
        default=None, 
        help="Device to run inference on (cuda device or cpu)"
    )
    parser.add_argument(
        "--classes", 
        type=int, 
        nargs="+", 
        default=[0], 
        help="Filter by class (0 for person in COCO, default: 0)"
    )
    parser.add_argument(
        "--tracker", "-t",
        type=str,
        default="bytetrack.yaml",
        help="Tracker config file (options: bytetrack.yaml, botsort.yaml) (default: bytetrack.yaml)"
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display the tracking results in real-time (not recommended for headless environments)"
    )
    parser.add_argument(
        "--draw_tracks",
        action="store_true",
        help="Draw track trails for each object"
    )
    parser.add_argument(
        "--trail_length",
        type=int,
        default=30,
        help="Maximum length of tracking trails (default: 30)"
    )
    parser.add_argument(
        "--codec",
        type=str,
        default=None,
        help="Video codec to use (e.g., 'avc1', 'XVID', 'mp4v'). Default is auto-selected based on platform."
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="Save individual frames as images instead of a video file. Useful for headless environments."
    )
    parser.add_argument(
        "--frame_dir",
        type=str,
        default=None,
        help="Directory to save individual frames when using --save_frames (default: output_frames/)"
    )
    parser.add_argument(
        "--frame_format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Format to save individual frames (jpg or png) (default: jpg)"
    )
    parser.add_argument(
        "--every_nth_frame",
        type=int,
        default=1,
        help="Save every Nth frame when using --save_frames (default: 1, save all frames)"
    )
    parser.add_argument(
        "--crop_boxes",
        action="store_true",
        help="Crop and save person bounding boxes from frames"
    )
    parser.add_argument(
        "--crop_dir",
        type=str,
        default=None,
        help="Directory to save cropped person boxes (default: input_filename_crops/)"
    )
    parser.add_argument(
        "--crop_format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Format to save cropped person boxes (jpg or png) (default: jpg)"
    )
    parser.add_argument(
        "--crop_padding",
        type=float,
        default=0.1,
        help="Padding around bounding box as fraction of box size (default: 0.1)"
    )
    return parser.parse_args()

def process_video(args):
    # Determine device
    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Open video file
    video_path = args.input_video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video properties: {width}x{height} at {fps} FPS, {total_frames} frames")
    
    # Set up frame saving if requested
    if args.save_frames:
        # Determine frame directory
        if args.frame_dir is None:
            base, _ = os.path.splitext(args.input_video)
            frame_dir = f"{base}_frames"
        else:
            frame_dir = args.frame_dir
        
        # Create directory if it doesn't exist
        os.makedirs(frame_dir, exist_ok=True)
        print(f"Will save frames to: {frame_dir}")
        
        # Determine frame format
        if args.frame_format == "jpg":
            frame_ext = ".jpg"
            frame_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        else:
            frame_ext = ".png"
            frame_quality = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        
        writer = None
    
    # Set up crop saving if requested
    if args.crop_boxes:
        # Determine crop directory
        if args.crop_dir is None:
            base, _ = os.path.splitext(args.input_video)
            crop_dir = f"{base}_crops"
        else:
            crop_dir = args.crop_dir
        
        # Create directory if it doesn't exist
        os.makedirs(crop_dir, exist_ok=True)
        print(f"Will save person crops to: {crop_dir}")
        
        # Determine crop format
        if args.crop_format == "jpg":
            crop_ext = ".jpg"
            crop_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        else:
            crop_ext = ".png"
            crop_quality = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    
    # Initialize tracking history for drawing trails
    track_history = defaultdict(lambda: [])
    
    # Process frames
    progress_bar = tqdm(total=total_frames, desc="Processing video")
    frame_count = 0
    total_crops = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            progress_bar.update(1)
            
            # Skip frames if needed when saving individual frames
            if args.save_frames and args.every_nth_frame > 1:
                if frame_count % args.every_nth_frame != 0:
                    continue
            
            # Run inference with tracking
            results = model.track(
                source=frame,
                conf=args.confidence,
                iou=args.iou,
                classes=args.classes,
                tracker=args.tracker,
                persist=True,
                device=device
            )[0]
            
            # Create annotated frame
            annotated_frame = results.plot()
            
            # Draw tracking trails if requested
            if args.draw_tracks and results.boxes.id is not None:
                boxes = results.boxes.xywh.cpu()
                track_ids = results.boxes.id.int().cpu().tolist()
                
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    # Store center coordinates
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    # Remove old points
                    if len(track) > args.trail_length:
                        track.pop(0)
                    
                    # Draw tracking line
                    if len(track) >= 2:
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
            
            # Crop and save person boxes if requested
            if args.crop_boxes and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu()
                track_ids = results.boxes.id.int().cpu().tolist()
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    # Add padding to the bounding box
                    w = x2 - x1
                    h = y2 - y1
                    pad_w = int(w * args.crop_padding)
                    pad_h = int(h * args.crop_padding)
                    
                    # Ensure coordinates stay within frame boundaries
                    x1 = max(0, int(x1) - pad_w)
                    y1 = max(0, int(y1) - pad_h)
                    x2 = min(width, int(x2) + pad_w)
                    y2 = min(height, int(y2) + pad_h)
                    
                    # Crop the person
                    person_crop = frame[y1:y2, x1:x2]
                    
                    # Save the crop
                    if person_crop.size > 0:  # Check if crop is valid
                        crop_file = os.path.join(crop_dir, f"frame_{frame_count:06d}_id_{track_id:06d}{crop_ext}")
                        cv2.imwrite(crop_file, person_crop, crop_quality)
                        total_crops += 1
            
            # Show the frame if requested and not in headless environment
            if args.show:
                try:
                    cv2.imshow("Person Tracking", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"Warning: Could not display frame: {e}")
                    print("Continuing without display...")
                    args.show = False
            
            # Save the frame
            if args.save_frames:
                # Generate frame filename with padding for proper sorting
                frame_file = os.path.join(frame_dir, f"frame_{frame_count:06d}{frame_ext}")
                cv2.imwrite(frame_file, annotated_frame, frame_quality)
                if frame_count % 100 == 0:
                    print(f"Saved frame {frame_count}/{total_frames}")
            else:
                # Write to video file
                writer.write(annotated_frame)
            
    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        # Cleanup
        progress_bar.close()
        cap.release()
        if not args.save_frames and writer is not None:
            writer.release()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except:
                pass
                
        # Report output
        if args.save_frames:
            print(f"Saved frames to directory: {frame_dir}")
            print(f"Total frames saved: {frame_count // args.every_nth_frame}")
        if args.crop_boxes:
            print(f"Saved person crops to directory: {crop_dir}")
            print(f"Total person crops saved: {total_crops}")
        if not args.save_frames:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Output video saved to: {output_path}")
                print(f"Output file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            else:
                print(f"Error: Output video file is empty or missing: {output_path}")

def main():
    args = parse_arguments()
    process_video(args)

if __name__ == "__main__":
    main() 