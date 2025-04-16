#!/bin/bash

ROOT=$1

cd third_party/WHAM

WHAM_DIR="$ROOT/wham"
mkdir -p "$WHAM_DIR"

for vid_dir in $ROOT/image/*/; do
    video_id=$(basename "$vid_dir")
    python demo.py --video $ROOT/image/$video_id/%6d.jpg --save_pkl \
    --output_pth $WHAM_DIR/$video_id
done

python aggregate_label.py $ROOT