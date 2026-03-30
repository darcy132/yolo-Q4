#!/bin/bash

SRC="//home/forge/workspace/yolo-Q4/dataset"
DST="/home/forge/workspace/yolo-Q4/dataset_aug"

cp -r "$SRC/images/val"  "$DST/images/val"
cp -r "$SRC/labels/val"  "$DST/labels/val"
cp    "$SRC/dataset.yaml" "$DST/dataset.yaml"

echo "Done: val + yaml copied to $DST"