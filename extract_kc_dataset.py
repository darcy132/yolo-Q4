"""
提取 hbgdf（红白杆倒伏，class 1）的二分类数据集

逻辑：
1. 扫描所有 images/train + images/val 及对应 labels
2. 过滤：只保留标注中 含有 class 5 的图片
3. 清洗：把保留图片的标注文件里非 class 5 的行去掉，class 5 重映射为 0
4. 按 8:2 划分 train/val
5. 写出到 dataset_kc/ 目录
"""

import os
import shutil
import random
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────
SOURCE_DATASET = "./dataset"          # 原始数据集根目录
OUTPUT_DATASET = "./dataset_kc"       # 输出目录
TARGET_CLASS   = 5                    # kc 坑槽
NEW_CLASS_ID   = 0                    # 二分类映射到 0
TRAIN_RATIO    = 0.8
RANDOM_SEED    = 42

IMAGE_EXTS = {".jpg"}
# ──────────────────────────────────────────────────────────────────────


def find_image_label_pairs(dataset_root: Path):
    """遍历 images/train 和 images/val，返回 (image_path, label_path) 对列表"""
    pairs = []
    for split in ["train", "val"]:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split
        if not img_dir.exists():
            print(f"[WARN] 找不到目录: {img_dir}，跳过")
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            pairs.append((img_path, lbl_path))
    print(f"[INFO] 共找到 {len(pairs)} 个图片-标注对")
    return pairs


def has_target_class(label_path: Path, target: int) -> bool:
    """判断标注文件中是否存在目标类别"""
    if not label_path.exists():
        return False
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(parts[0]) == target:
                return True
    return False


def filter_label(label_path: Path, target: int, new_id: int) -> list[str]:
    """只保留 target 类的行，并重映射类别 ID"""
    lines = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if int(parts[0]) == target:
                parts[0] = str(new_id)
                lines.append(" ".join(parts))
    return lines


def write_yaml(output_root: Path):
    yaml_content = f"""path: {output_root.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: kc
"""
    (output_root / "dataset.yaml").write_text(yaml_content)
    (output_root / "classes.txt").write_text("cz\n")


def main():
    random.seed(RANDOM_SEED)
    src = Path(SOURCE_DATASET)
    out = Path(OUTPUT_DATASET)

    # Step 1: 找所有数据对
    all_pairs = find_image_label_pairs(src)

    # Step 2: 过滤含 class 7 的图片
    cz_pairs = [(img, lbl) for img, lbl in all_pairs
                if has_target_class(lbl, TARGET_CLASS)]
    print(f"[INFO] 含 cz 类别的图片数: {len(cz_pairs)}")

    if not cz_pairs:
        print("[ERROR] 没有找到含 cz 类别的图片，请检查数据集路径和类别 ID")
        return

    # Step 3: 按 8:2 划分
    random.shuffle(cz_pairs)
    n_train = int(len(cz_pairs) * TRAIN_RATIO)
    train_pairs = cz_pairs[:n_train]
    val_pairs   = cz_pairs[n_train:]
    print(f"[INFO] 划分 → train: {len(train_pairs)}  val: {len(val_pairs)}")

    # Step 4: 建输出目录
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Step 5: 写出文件
    stats = {"train": 0, "val": 0}
    for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_path, lbl_path in pairs:
            # 复制图片
            dst_img = out / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            # 过滤并写标注
            filtered_lines = filter_label(lbl_path, TARGET_CLASS, NEW_CLASS_ID)
            dst_lbl = out / "labels" / split / lbl_path.name
            with open(dst_lbl, "w") as f:
                f.write("\n".join(filtered_lines) + ("\n" if filtered_lines else ""))

            stats[split] += 1

    # Step 6: 写 yaml / classes.txt
    write_yaml(out)

    print(f"\n[DONE] 数据集已写出到: {out.resolve()}")
    print(f"       train: {stats['train']} 张")
    print(f"       val  : {stats['val']} 张")
    print(f"       dataset.yaml + classes.txt 已生成")


if __name__ == "__main__":
    main()