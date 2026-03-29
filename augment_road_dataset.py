"""
离线道路路面检测数据集增强脚本
支持 YOLO 格式（txt bbox）的图像+标注同步增强
针对 8 类路面缺陷设计：lmlj/hbgdf/hxlf/zxlf/jl/kc/ssf/cz

使用方法：
    python augment_road_dataset.py \
        --src_dir /home/forge/workspace/yolo-Q4/dataset \
        --out_dir /home/forge/workspace/yolo-Q4/dataset_aug \
        --split train \
        --seed 42
"""

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────
# albumentations（延迟 import，方便 CI 检查）
# ──────────────────────────────────────────────
try:
    import albumentations as A
    from albumentations.core.bbox_utils import convert_bboxes_from_albumentations
except ImportError:
    raise ImportError("请先安装：pip install albumentations")

# ══════════════════════════════════════════════
# 1. 类别配置
# ══════════════════════════════════════════════

CLASS_NAMES = ['lmlj', 'hbgdf', 'hxlf', 'zxlf', 'jl', 'kc', 'ssf', 'cz']
CLASS_COUNTS = {
    0: 1000,   # lmlj  路面垃圾
    1: 47,     # hbgdf 红白杆倒伏  ← 严重不足
    2: 1330,   # hxlf  横向裂缝
    3: 1631,   # zxlf  纵向裂缝
    4: 702,    # jl    龟裂
    5: 492,    # kc    坑槽
    6: 256,    # ssf   伸缩缝破损  ← 不足
    7: 285,    # cz    路面车辙    ← 不足
}

# 目标增强倍率（最终每类图像数 ≈ 原始 × multiplier）
# 小样本类别自动拉高，避免训练时class imbalance
BASE_MULTIPLIER = 0          # 普通类别增强3倍
SMALL_THRESH    = 300        # 低于此数量视为小样本
SMALL_MULT      = 0         # 小样本增强10倍（hbgdf/ssf/cz）
TINY_THRESH     = 100        # 极小样本
TINY_MULT       = 8         # hbgdf 极小样本增强20倍


def get_multiplier(class_id: int) -> int:
    count = CLASS_COUNTS.get(class_id, 1000)
    if count < TINY_THRESH:
        return TINY_MULT
    if count < SMALL_THRESH:
        return SMALL_MULT
    return BASE_MULTIPLIER


# ══════════════════════════════════════════════
# 2. Albumentations Pipeline 定义
# ══════════════════════════════════════════════
def build_pipeline_mild() -> A.Compose:
    """轻度增强：用于多数类，保持分布稳定"""
    return A.Compose(
        [
            # ── 几何 ──────────────────────────────
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.85, 1.15),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.Perspective(scale=(0.03, 0.07), p=0.3),

            # ── 光照/颜色 ─────────────────────────
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=20, p=0.4),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),

            # ── 天气/环境 ─────────────────────────
            A.RandomShadow(
                shadow_roi=(0, 0.3, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=4,
                p=0.25,
            ),
            A.RandomFog(fog_coef_range=(0.05, 0.2), p=0.15),
            A.RandomRain(
                slant_range=(-5, 5),
                drop_length=8, drop_width=1,
                drop_color=(180, 180, 180),
                blur_value=2, brightness_coefficient=0.9,
                p=0.1,
            ),

            # ── 噪声/模糊/压缩 ───────────────────
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.11)),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.2)),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MedianBlur(blur_limit=3),
                ],
                p=0.2,
            ),
            A.ImageCompression(quality_range=(70, 95), p=0.2),

            # ── 遮挡 ──────────────────────────────
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=128,
                p=0.15,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_area=100,
            min_visibility=0.3,
        ),
    )


def build_pipeline_heavy() -> A.Compose:
    """重度增强：用于小样本类，最大化数据多样性"""
    return A.Compose(
        [
            # ── 几何（更激进）─────────────────────
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent=(-0.1, 0.1),
                scale=(0.75, 1.25),
                rotate=(-20, 20),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7,
            ),
            A.Perspective(scale=(0.05, 0.12), p=0.5),
            A.ElasticTransform(
                alpha=30, sigma=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3,
            ),

            # ── 光照/颜色（更激进）───────────────
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=35, val_shift_limit=35, p=0.6),
            A.CLAHE(clip_limit=5.0, tile_grid_size=(8, 8), p=0.4),
            A.RandomGamma(gamma_limit=(70, 140), p=0.4),
            A.ToGray(p=0.1),

            # ── 天气/环境 ─────────────────────────
            A.RandomShadow(
                shadow_roi=(0, 0.2, 1, 1),
                num_shadows_limit=(1, 3),
                shadow_dimension=5,
                p=0.4,
            ),
            A.RandomFog(fog_coef_range=(0.05, 0.3), p=0.25),
            A.RandomRain(p=0.15),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_range=(0, 1),
                num_flare_circles_range=(3, 6),
                p=0.1,
            ),

            # ── 噪声/模糊/压缩 ───────────────────
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.04, 0.24)),
                    A.ISONoise(color_shift=(0.02, 0.08), intensity=(0.1, 0.35)),
                    A.MultiplicativeNoise(multiplier=(0.85, 1.15)),
                ],
                p=0.4,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=7),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.Defocus(radius=(1, 3)),
                ],
                p=0.3,
            ),
            A.ImageCompression(quality_range=(50, 90), p=0.3),

            # ── 遮挡 ──────────────────────────────
            A.CoarseDropout(
                num_holes_range=(2, 8),
                hole_height_range=(12, 48),
                hole_width_range=(12, 48),
                fill=128,
                p=0.25,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_area=100,
            min_visibility=0.3,
        ),
    )

# ══════════════════════════════════════════════
# 3. YOLO 标注 IO
# ══════════════════════════════════════════════

def load_yolo_labels(label_path: Path):
    """读取 YOLO txt，返回 (class_ids, bboxes)，bbox 格式 [cx, cy, w, h]"""
    class_ids, bboxes = [], []
    if not label_path.exists():
        return class_ids, bboxes
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            # 过滤无效框
            if w <= 0 or h <= 0:
                continue
            class_ids.append(cid)
            bboxes.append([cx, cy, w, h])
    return class_ids, bboxes


def save_yolo_labels(label_path: Path, class_ids, bboxes):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for cid, (cx, cy, w, h) in zip(class_ids, bboxes):
            # 截断到 [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w  = max(0.001, min(1.0, w))
            h  = max(0.001, min(1.0, h))
            f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ══════════════════════════════════════════════
# 4. 单张图像增强
# ══════════════════════════════════════════════

def augment_one(
    img_path: Path,
    label_path: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    pipeline: A.Compose,
    n_copies: int,
    stem_prefix: str = "",
):
    """对单张图生成 n_copies 份增强结果，写入输出目录"""
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    class_ids, bboxes = load_yolo_labels(label_path)
    if not class_ids:
        return 0  # 无标注则跳过

    success = 0
    for i in range(n_copies):
        try:
            result = pipeline(
                image=img_rgb,
                bboxes=bboxes,
                class_labels=class_ids,
            )
        except Exception as e:
            # albumentations 偶发错误，跳过这次
            continue

        aug_bboxes = result["bboxes"]
        aug_labels = [int(x) for x in result["class_labels"]]
        if not aug_labels:
            continue  # 增强后所有框都消失了，放弃

        stem = f"{stem_prefix}{img_path.stem}_aug{i:03d}"
        out_img_path = out_img_dir / f"{stem}{img_path.suffix}"
        out_lbl_path = out_lbl_dir / f"{stem}.txt"

        aug_bgr = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_img_path), aug_bgr)
        save_yolo_labels(out_lbl_path, aug_labels, aug_bboxes)
        success += 1

    return success


# ══════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════

def collect_images(img_dir: Path, label_dir: Path):
    """收集有对应标注的图像，按主导类别分组"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]

    by_class = defaultdict(list)   # {dominant_class_id: [img_path, ...]}
    no_label = 0

    for img_path in img_paths:
        lbl_path = label_dir / (img_path.stem + ".txt")
        class_ids, _ = load_yolo_labels(lbl_path)
        if not class_ids:
            no_label += 1
            continue
        # 以出现频次最高的类别作为该图的"主导类别"
        dominant = max(set(class_ids), key=class_ids.count)
        by_class[dominant].append(img_path)

    if no_label:
        print(f"  [警告] {no_label} 张图没有对应标注，已跳过")
    return by_class


def run_augmentation(
    src_dir: Path,
    out_dir: Path,
    split: str,
    seed: int,
    copy_original: bool,
):
    random.seed(seed)
    np.random.seed(seed)

    src_img_dir = src_dir / "images" / split
    src_lbl_dir = src_dir / "labels" / split
    out_img_dir = out_dir / "images" / split
    out_lbl_dir = out_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 1) 可选：复制原始文件
    if copy_original:
        print("→ 复制原始图像...")
        for img in tqdm(sorted(src_img_dir.iterdir())):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                shutil.copy2(img, out_img_dir / img.name)
                lbl = src_lbl_dir / (img.stem + ".txt")
                if lbl.exists():
                    shutil.copy2(lbl, out_lbl_dir / lbl.name)

    # 2) 按类别分组
    print("→ 扫描标注文件...")
    by_class = collect_images(src_img_dir, src_lbl_dir)
    total_imgs = sum(len(v) for v in by_class.values())
    print(f"  有效图像: {total_imgs} 张，涉及 {len(by_class)} 个主导类别")

    mild_pipeline  = build_pipeline_mild()
    heavy_pipeline = build_pipeline_heavy()

    total_generated = 0

    for class_id, img_list in sorted(by_class.items()):
        mult = get_multiplier(class_id)
        cname = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
        pipeline = heavy_pipeline if mult >= SMALL_MULT else mild_pipeline

        print(
            f"  [class {class_id}] {cname:8s}  "
            f"{len(img_list):4d} 张  ×{mult}  "
            f"{'(heavy)' if mult >= SMALL_MULT else '(mild) '}"
        )

        for img_path in tqdm(img_list, desc=f"  {cname}", leave=False):
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            n = augment_one(
                img_path, lbl_path,
                out_img_dir, out_lbl_dir,
                pipeline, n_copies=mult,
                stem_prefix=f"c{class_id}_",
            )
            total_generated += n

    print(f"\n✓ 共生成 {total_generated} 张增强图像")
    print(f"  输出目录: {out_dir}")
    return total_generated


# ══════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════
# /home/featurize/data/dataset
def parse_args():
    p = argparse.ArgumentParser(description="道路数据集离线增强")
    p.add_argument("--src_dir",       default="/home/featurize/data/dataset/dataset",
                   help="原始数据集根目录（含 images/labels 子目录）")
    p.add_argument("--out_dir",       default="/home/featurize/data/dataset/dataset_aug",
                   help="输出数据集根目录")
    p.add_argument("--split",         default="train",
                   choices=["train", "val", "test"],
                   help="处理哪个 split（建议只增强 train）")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--no_copy_orig",  action="store_true",
                   help="不复制原始文件（只写增强结果）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_augmentation(
        src_dir=Path(args.src_dir),
        out_dir=Path(args.out_dir),
        split=args.split,
        seed=args.seed,
        copy_original=not args.no_copy_orig,
    )