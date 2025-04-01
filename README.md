# Person Tracking with Ultralytics YOLOv8

This project provides a simple and efficient way to track people in videos using Ultralytics YOLOv8 and ByteTrack. It processes input videos, detects and tracks people, and outputs an annotated video with bounding boxes and tracking IDs.

## Features

- Person detection using YOLOv8
- Multiple object tracking with ByteTrack
- Support for GPU acceleration
- Progress bar for tracking processing status
- Customizable confidence thresholds and model selection
- ID-based tracking with persistent tracking across frames
- Visual tracking history with trail visualization
- Support for headless environments

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics
- CUDA-capable GPU (optional, for faster processing)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/person-tracking.git
   cd person-tracking
   ```

2. Run the setup script to create a virtual environment and install dependencies:
   ```
   chmod +x setup_and_run.sh
   ./setup_and_run.sh
   ```

Alternatively, you can manually install the required packages:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Basic usage:

```
./person_tracker.py -i path/to/your/video.mp4
```

This will process the video and save the output as `path/to/your/video_tracked.mp4`.

### Command Line Arguments

- `-i, --input_video`: Path to input video file (required)
- `-o, --output_video`: Path to output video file (optional)
- `-m, --model`: YOLOv8 model to use (default: yolov8n.pt)
- `-c, --confidence`: Detection confidence threshold (default: 0.3)
- `--iou`: IOU threshold for NMS (default: 0.5)
- `-d, --device`: Device to run on ('cpu' or 'cuda:0', etc.)
- `-t, --tracker`: Tracker type to use (default: bytetrack.yaml)
- `-s, --show`: Display tracking results in real-time (not for headless environments)
- `--draw_tracks`: Draw tracking trails for each person
- `--trail_length`: Maximum length of tracking trails (default: 30)
- `--classes`: Class IDs to track (default: 0 for person in COCO dataset)
- `--codec`: Video codec to use (e.g., 'XVID', 'mp4v')
- `--save_frames`: Save individual frames as images instead of a video
- `--frame_dir`: Directory to save frames when using --save_frames
- `--frame_format`: Format to save frames (jpg or png)
- `--every_nth_frame`: Save every N-th frame when using --save_frames
- `--crop_boxes`: Crop and save person bounding boxes from frames
- `--crop_dir`: Directory to save cropped person boxes (default: input_filename_crops/)
- `--crop_format`: Format to save cropped person boxes (jpg or png)
- `--crop_padding`: Padding around bounding box as fraction of box size (default: 0.1)

### Examples

Track people with higher confidence threshold:
```
./person_tracker.py -i video.mp4 -c 0.5
```

Use a different YOLOv8 model:
```
./person_tracker.py -i video.mp4 -m yolov8s.pt
```

Force CPU usage:
```
./person_tracker.py -i video.mp4 -d cpu
```

Custom output path:
```
./person_tracker.py -i input.mp4 -o output_tracked.mp4
```

With visual tracking trails:
```
./person_tracker.py -i video.mp4 --draw_tracks --trail_length 50
```

Using a specific video codec:
```
./person_tracker.py -i video.mp4 --codec XVID
```

Save frames instead of video (useful for headless environments):
```
./person_tracker.py -i video.mp4 --save_frames
```

Save cropped person boxes for re-identification:
```
./person_tracker.py -i video.mp4 --crop_boxes
```

Save cropped person boxes with custom padding:
```
./person_tracker.py -i video.mp4 --crop_boxes --crop_padding 0.2
```

Save cropped person boxes in PNG format:
```
./person_tracker.py -i video.mp4 --crop_boxes --crop_format png
```

## Troubleshooting

### Video Output Issues

If you encounter issues with the output video (cannot open, corrupted file, etc.):

1. Run the diagnostic tool to check which video codecs work on your system:
   ```
   ./check_opencv_codecs.py
   ```

2. Use a compatible codec based on the diagnostic results:
   ```
   ./person_tracker.py -i video.mp4 --codec XVID
   ```

3. For headless environments, use the frame saving option:
   ```
   ./person_tracker.py -i video.mp4 --save_frames
   ```

4. Ensure ffmpeg is installed:
   ```
   sudo apt-get update && sudo apt-get install ffmpeg
   ```

### Common Issues

- **Blank or corrupted output video**: Use the `--codec` option with a compatible codec like XVID
- **Qt/GUI errors**: Use `--save_frames` option instead of creating a video
- **CUDA errors**: Use the `-d cpu` option to force CPU processing
- **Memory errors**: Use a smaller model like yolov8n.pt or reduce the frame resolution

## Models

The script supports multiple YOLO versions from YOLOv3 to YOLOv12. You can specify the version using the `--yolo_version` argument:

```
./person_tracker.py -i video.mp4 --yolo_version 8 -m yolov8n.pt
```

Available YOLO versions and their models:

### YOLOv3
- yolov3.pt (original YOLOv3)
- yolov3-spp.pt (Spatial Pyramid Pooling)
- yolov3-tiny.pt (lightweight version)

### YOLOv4
- yolov4.pt (original YOLOv4)
- yolov4-tiny.pt (lightweight version)
- yolov4-csp.pt (Cross Stage Partial Network)

### YOLOv5
- yolov5n.pt (nano)
- yolov5s.pt (small)
- yolov5m.pt (medium)
- yolov5l.pt (large)
- yolov5x.pt (extra large)

### YOLOv6
- yolov6n.pt (nano)
- yolov6s.pt (small)
- yolov6m.pt (medium)
- yolov6l.pt (large)

### YOLOv7
- yolov7.pt (base)
- yolov7-tiny.pt (tiny)
- yolov7x.pt (extra large)

### YOLOv8
- yolov8n.pt (nano)
- yolov8s.pt (small)
- yolov8m.pt (medium)
- yolov8l.pt (large)
- yolov8x.pt (extra large)

### YOLOv9
- yolov9c.pt (compact)
- yolov9e.pt (efficient)
- yolov9m.pt (medium)

### YOLOv10
- yolov10n.pt (nano)
- yolov10s.pt (small)
- yolov10m.pt (medium)

### YOLOv11
- yolov11n.pt (nano)
- yolov11s.pt (small)
- yolov11m.pt (medium)

### YOLOv12
- yolov12n.pt (nano)
- yolov12s.pt (small)
- yolov12m.pt (medium)

The first time you specify a model, it will be automatically downloaded from the Ultralytics HUB.

### Model Selection Guidelines

1. **Speed vs. Accuracy Trade-off**:
   - For real-time applications: Use nano (n) or small (s) models
   - For high accuracy: Use large (l) or extra large (x) models
   - For balanced performance: Use medium (m) models

2. **Hardware Considerations**:
   - GPU memory: Larger models require more VRAM
   - CPU usage: Smaller models are more efficient on CPU
   - Mobile/Edge: Use tiny or nano versions

3. **Use Cases**:
   - Surveillance: YOLOv8 or newer for better tracking
   - Real-time tracking: YOLOv5 or newer for better speed
   - High accuracy: Latest version (YOLOv12) for best performance

## License

This project is licensed under the MIT License - see the LICENSE file for details. 