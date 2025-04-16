#!/bin/bash

INPUT_FILE=$1
ROOT=$2

if [ -z "$INPUT_FILE" ] || [ -z "$ROOT" ]; then
  echo "Usage: $0 youtube_ids.txt $ROOT"
  exit 1
fi

# Create root directories
VIDEO_DIR="$ROOT/video"
IMAGE_DIR="$ROOT/image"
DEPTH_DIR="$ROOT/depth"
mkdir -p "$VIDEO_DIR" "$IMAGE_DIR" "$DEPTH_DIR"

# Loop over each YouTube ID in the input file
while read -r yid; do
  echo "Processing video ID: $yid"
  
  for i in $(seq 0 99); do
    start_sec=$((i * 30 + 25))
    end_sec=$((i * 30 + 30))

    # Format time as HH:MM:SS
    start_ts=$(printf "%02d:%02d:%02d" $((start_sec/3600)) $(((start_sec%3600)/60)) $((start_sec%60)))
    end_ts=$(printf "%02d:%02d:%02d" $((end_sec/3600)) $(((end_sec%3600)/60)) $((end_sec%60)))

    outname="${yid}_${i}.mp4"
    outdir="${yid}_${i}"

    echo "  Segment $i: $start_ts to $end_ts"

    # Download video clip
    yt-dlp -f 298 \
      --download-sections "*$start_ts-$end_ts" \
      --force-keyframes-at-cuts \
      -o "$VIDEO_DIR/$outname" "$yid"

    # Skip if download failed
    if [ ! -f "$VIDEO_DIR/$outname" ]; then
      echo "    Skipped: could not download segment."
      break
    fi

    # Create image output directory
    mkdir -p "$IMAGE_DIR/$outdir"

    # Extract frames
    ffmpeg -i "$VIDEO_DIR/$outname" -vf "fps=30" -q:v 1 "$IMAGE_DIR/$outdir/%06d.jpg" || {
      echo "    FFmpeg failed on $outname, continuing..."
      continue
    }

    python preprocess/inference_model.py $IMAGE_DIR/$outdir

  done
done < "$INPUT_FILE"
