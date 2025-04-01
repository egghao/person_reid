#!/bin/bash

# URL for Market-1501 dataset (v15.09.15)
# Note: This URL might be unofficial or change. Check for official sources if possible.
URL="http://188.138.128.150/Market/Market-1501-v15.09.15.zip"
FILENAME="Market-1501-v15.09.15.zip"
DEST_DIR="Market-1501-v15.09.15"

echo "Downloading Market-1501 dataset..."
echo "URL: $URL"
echo "Destination file: $FILENAME"

# Download using wget
wget -O "$FILENAME" "$URL"

# Check if download was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to download the dataset. Please check the URL or your internet connection."
  # Try curl as an alternative
  echo "Trying with curl..."
  curl -L -o "$FILENAME" "$URL"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to download with curl as well."
    exit 1
  fi
fi

echo "Download complete."

# Check if unzip is installed
if ! command -v unzip &> /dev/null
then
    echo "Error: 'unzip' command not found. Please install unzip (e.g., 'sudo apt install unzip' or 'brew install unzip')."
    exit 1
fi

# Extract the dataset
echo "Extracting the dataset to $DEST_DIR..."
unzip "$FILENAME" -d "$DEST_DIR"

# Check if extraction was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to extract the dataset."
  exit 1
fi

echo "Extraction complete."
echo "Dataset downloaded and extracted to: $DEST_DIR"

# Optional: Remove the zip file after extraction
# echo "Removing the zip file: $FILENAME"
# rm "$FILENAME"

echo "Script finished." 