#!/bin/bash

SRC="//home/featurize/data/dataset"
DST="/home/featurize/data/dataset_aug"

cp -r "$SRC/images/val"  "$DST/images/val"
cp -r "$SRC/labels/val"  "$DST/labels/val"
cp    "$SRC/dataset.yaml" "$DST/dataset.yaml"

echo "Done: val + yaml copied to $DST"