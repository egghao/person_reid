#!/bin/bash

# Display header
echo "========================================================"
echo "      Person Tracking with Ultralytics YOLOv8"
echo "========================================================"

# Setup virtual environment
echo -e "\n[1] Setting up virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "\n[2] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo -e "\n[3] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download pretrained model if needed
echo -e "\n[4] Checking YOLOv8 model..."
if [ ! -f "yolov8n.pt" ]; then
    echo "Downloading YOLOv8n model..."
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
fi

# Make person_tracker.py executable
chmod +x person_tracker.py

# Display usage instructions
echo -e "\n========================================================"
echo "                 Setup Complete!"
echo "========================================================"
echo -e "\nUsage:"
echo "  ./person_tracker.py -i /path/to/video.mp4 [options]"
echo -e "\nOptions:"
echo "  -i, --input_video     Path to input video file (required)"
echo "  -o, --output_video    Path to output video file (optional)"
echo "  -m, --model           YOLOv8 model to use (default: yolov8n.pt)"
echo "  -c, --confidence      Detection confidence threshold (default: 0.3)"
echo "  --iou                 IOU threshold for NMS (default: 0.5)"
echo "  -d, --device          Device to run on ('cpu' or 'cuda:0', etc.)"
echo "  -t, --tracker         Tracker type to use (default: bytetrack.yaml)"
echo "  -s, --show            Display tracking results in real-time"
echo "  --draw_tracks         Draw tracking trails for each person"
echo "  --trail_length        Maximum length of tracking trails (default: 30)"
echo "  --classes             Class IDs to track (default: 0 for person)"
echo -e "\nExamples:"
echo "  ./person_tracker.py -i sample.mp4 -o tracked_output.mp4 -c 0.4"
echo "  ./person_tracker.py -i sample.mp4 -s --draw_tracks"
echo "  ./person_tracker.py -i sample.mp4 -t botsort.yaml"
echo -e "\nNote: Make sure to remain in the virtual environment (venv)"
echo "      or activate it using 'source venv/bin/activate'"
echo "========================================================\n" 