"""
从多类数据集中提取指定类别，构建子集数据集。

配置项：
  SOURCE_DATASET  原始数据集根目录
  OUTPUT_DATASET  输出目录
  CLASS_MAP       {原始类别ID: 新类别ID, ...}
  CLASS_NAMES     {新类别ID: 类别名称, ...}  用于生成 dataset.yaml
  TRAIN_RATIO     训练集比例
  RANDOM_SEED     随机种子
"""

import os
import shutil
import random
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────
SOURCE_DATASET = "./dataset"
OUTPUT_DATASET = "./dataset_sub"

# 原始类别ID → 新类别ID，按需修改
CLASS_MAP: dict[int, int] = {
    1: 0,   # hbgdf → 0
    5: 1,   # kc    → 1
}

# 新类别ID → 类别名称，用于 dataset.yaml
CLASS_NAMES: dict[int, str] = {
    0: "hbgdf",
    1: "kc",
}

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
IMAGE_EXTS  = {".jpg", ".jpeg", ".png"}
# ──────────────────────────────────────────────────────────────────────


def find_image_label_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
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


def has_any_target(label_path: Path, class_map: dict[int, int]) -> bool:
    """标注文件中是否包含任意目标类别"""
    if not label_path.exists():
        return False
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(parts[0]) in class_map:
                return True
    return False


def filter_and_remap_label(
    label_path: Path,
    class_map: dict[int, int],
) -> list[str]:
    """保留目标类别的行，并将类别 ID 重映射"""
    lines = []
    if not label_path.exists():
        return lines
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            orig_id = int(parts[0])
            if orig_id in class_map:
                parts[0] = str(class_map[orig_id])
                lines.append(" ".join(parts))
    return lines


def write_meta(output_root: Path, class_names: dict[int, str]):
    nc = len(class_names)
    # 按新 ID 排序，确保顺序正确
    names_block = "\n".join(
        f"  {k}: {class_names[k]}" for k in sorted(class_names)
    )
    yaml_content = (
        f"path: {output_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {nc}\n"
        f"names:\n{names_block}\n"
    )
    (output_root / "dataset.yaml").write_text(yaml_content)

    classes_txt = "\n".join(class_names[k] for k in sorted(class_names))
    (output_root / "classes.txt").write_text(classes_txt + "\n")


def main():
    random.seed(RANDOM_SEED)
    src = Path(SOURCE_DATASET)
    out = Path(OUTPUT_DATASET)

    if not CLASS_MAP:
        print("[ERROR] CLASS_MAP 为空，请先配置要提取的类别")
        return

    print(f"[INFO] 提取类别映射: {CLASS_MAP}")

    # Step 1: 枚举所有图片-标注对
    all_pairs = find_image_label_pairs(src)

    # Step 2: 过滤出包含至少一个目标类别的样本
    target_pairs = [
        (img, lbl) for img, lbl in all_pairs
        if has_any_target(lbl, CLASS_MAP)
    ]
    print(f"[INFO] 含目标类别的图片数: {len(target_pairs)}")

    if not target_pairs:
        print("[ERROR] 没有找到含目标类别的图片，请检查数据集路径和 CLASS_MAP")
        return

    # Step 3: 打印各原始类别命中数（便于核查）
    class_counts: dict[int, int] = {k: 0 for k in CLASS_MAP}
    for _, lbl in target_pairs:
        seen = set()
        if lbl.exists():
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cid = int(parts[0])
                        if cid in CLASS_MAP and cid not in seen:
                            class_counts[cid] += 1
                            seen.add(cid)
    for orig_id, cnt in class_counts.items():
        new_id = CLASS_MAP[orig_id]
        name   = CLASS_NAMES.get(new_id, f"class{new_id}")
        print(f"  原始 class {orig_id} ({name}): {cnt} 张图片含此类")

    # Step 4: 划分 train / val
    random.shuffle(target_pairs)
    n_train = int(len(target_pairs) * TRAIN_RATIO)
    train_pairs = target_pairs[:n_train]
    val_pairs   = target_pairs[n_train:]
    print(f"[INFO] 划分 → train: {len(train_pairs)}  val: {len(val_pairs)}")

    # Step 5: 创建输出目录
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Step 6: 写出图片与标注
    stats = {"train": 0, "val": 0}
    for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, out / "images" / split / img_path.name)

            filtered = filter_and_remap_label(lbl_path, CLASS_MAP)
            dst_lbl  = out / "labels" / split / lbl_path.name
            with open(dst_lbl, "w") as f:
                if filtered:
                    f.write("\n".join(filtered) + "\n")
            stats[split] += 1

    # Step 7: 写 dataset.yaml + classes.txt
    write_meta(out, CLASS_NAMES)

    print(f"\n[DONE] 数据集已写出到: {out.resolve()}")
    print(f"       train : {stats['train']} 张")
    print(f"       val   : {stats['val']} 张")
    print(f"       dataset.yaml + classes.txt 已生成")


if __name__ == "__main__":
    main()