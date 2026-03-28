#!/bin/bash

WEIGHTS_DIR="./runs/detect/runs/detect/road_damage_v1_yolo26l-freeze/weights"
BACKUP_DIR="./pt_backup"

mkdir -p "$BACKUP_DIR"

for f in "$WEIGHTS_DIR"/*.pt; do
    fname=$(basename "$f")
    if [[ "$fname" != "best.pt" && "$fname" != "last.pt" ]]; then
        mv "$f" "$BACKUP_DIR/$fname"
        echo "moved: $fname"
    fi
done

echo "done. backup in $BACKUP_DIR"

# 移动回来
mv ./pt_backup/*.pt ./runs/detect/runs/detect/road_damage_v1_yolo26l-freeze/weights/