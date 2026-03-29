"""
YOLO Dataset Visualizer — ultralytics-based
用法:
    python visualize_dataset.py --data data.yaml --split train --n 16 --out vis_output/
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics.utils.plotting import Annotator, colors


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Visualize YOLO-format dataset bboxes")
    p.add_argument("--data",  required=True, help="Path to data.yaml")
    p.add_argument("--split", default="val", choices=["train", "val", "test"],
                   help="Which split to visualize (default: val)")
    p.add_argument("--n",     type=int, default=200,
                   help="Number of images to visualize (default: 16)")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--out",   default="vis_output",
                   help="Output directory for annotated images (default: vis_output/)")
    p.add_argument("--grid",  action="store_true",
                   help="Also save a single grid mosaic image")
    p.add_argument("--grid-size", type=int, default=4,
                   help="Grid columns (default: 4)")
    p.add_argument("--conf",  action="store_true",
                   help="Show label counts per image in title (printed to stdout)")
    return p.parse_args()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Normalised xywh → absolute xyxy."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def find_label_path(img_path: Path) -> Path:
    """Resolve images/… → labels/… path."""
    parts = img_path.parts
    try:
        idx = next(i for i, p in enumerate(parts) if p == "images")
        label_parts = list(parts)
        label_parts[idx] = "labels"
        return Path(*label_parts).with_suffix(".txt")
    except StopIteration:
        # Fallback: same dir, .txt extension
        return img_path.with_suffix(".txt")


def annotate_image(img_bgr: np.ndarray, label_path: Path, class_names: list) -> np.ndarray:
    annotator = Annotator(img_bgr.copy(), line_width=2, font_size=10)
    h, w = img_bgr.shape[:2]

    if not label_path.exists():
        return annotator.result()

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)
            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            annotator.box_label(
                [x1, y1, x2, y2],
                label=label,
                color=colors(cls_id, bgr=True),
            )

    return annotator.result()


def make_grid(images: list, cols: int) -> np.ndarray:
    """Tile a list of BGR images into a grid (zero-pad to uniform size)."""
    max_h = max(im.shape[0] for im in images)
    max_w = max(im.shape[1] for im in images)

    padded = []
    for im in images:
        ph = max_h - im.shape[0]
        pw = max_w - im.shape[1]
        padded.append(np.pad(im, ((0, ph), (0, pw), (0, 0)), constant_values=20))

    rows = [np.hstack(padded[i:i+cols]) for i in range(0, len(padded), cols)]
    # pad last row if needed
    last = rows[-1]
    if last.shape[1] < rows[0].shape[1]:
        pw = rows[0].shape[1] - last.shape[1]
        rows[-1] = np.pad(last, ((0, 0), (0, pw), (0, 0)), constant_values=20)
    return np.vstack(rows)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)

    cfg = load_yaml(args.data)
    class_names = cfg.get("names", [])
    if isinstance(class_names, dict):          # ultralytics allows dict form
        class_names = [class_names[i] for i in sorted(class_names)]

    dataset_root = Path(args.data).parent

    # Resolve split path
    split_key = args.split
    split_val = cfg.get(split_key)
    if split_val is None:
        raise ValueError(f"Key '{split_key}' not found in {args.data}. "
                         f"Available keys: {list(cfg.keys())}")

    img_dir = (dataset_root / split_val).resolve()
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    all_imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]

    if not all_imgs:
        raise RuntimeError(f"No images found under {img_dir}")

    sample = random.sample(all_imgs, min(args.n, len(all_imgs)))
    print(f"[info] dataset root : {dataset_root}")
    print(f"[info] split        : {split_key}  ({len(all_imgs)} images total)")
    print(f"[info] visualising  : {len(sample)} images")
    print(f"[info] classes      : {class_names}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated = []
    for img_path in sample:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[warn] cannot read {img_path}, skipping")
            continue

        label_path = find_label_path(img_path)
        vis = annotate_image(bgr, label_path, class_names)

        if args.conf:
            n_boxes = sum(1 for _ in open(label_path)) if label_path.exists() else 0
            print(f"  {img_path.name:40s}  boxes={n_boxes}")

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), vis)
        annotated.append(vis)

    print(f"[info] saved {len(annotated)} annotated images → {out_dir}/")

    if args.grid and annotated:
        grid = make_grid(annotated, cols=args.grid_size)
        grid_path = out_dir / "mosaic.jpg"
        cv2.imwrite(str(grid_path), grid)
        print(f"[info] mosaic saved  → {grid_path}")


if __name__ == "__main__":
    main()