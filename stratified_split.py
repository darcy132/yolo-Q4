"""
stratified_split.py
--------------------
从已转换好的 yolo_dataset (images/train, images/val, labels/train, labels/val)
读取全部数据，按 **每个类别** 做 2:8 (val:train) 重新划分，
结果写入 dataset/ 目录。

用法:
    python stratified_split.py
    # 或指定路径
    python stratified_split.py --src /path/to/yolo_dataset --dst /path/to/dataset
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

# ==================== 类别定义（与原脚本保持一致）====================
CLASSES = {
    'lmlj': 0,    # 路面垃圾
    'hbgdf': 1,   # 红白杆倒伏
    'hxlf': 2,    # 横向裂缝
    'zxlf': 3,    # 纵向裂缝
    'jl': 4,      # 龟裂
    'kc': 5,      # 坑槽
    'ssf': 6,     # 伸缩缝破损
    'cz': 7,      # 路面车辙
}
CLASS_NAMES_CN = {
    'lmlj': '路面垃圾',
    'hbgdf': '红白杆倒伏',
    'hxlf': '横向裂缝',
    'zxlf': '纵向裂缝',
    'jl': '龟裂',
    'kc': '坑槽',
    'ssf': '伸缩缝破损',
    'cz': '路面车辙',
}
ID_TO_CODE = {v: k for k, v in CLASSES.items()}

# ==================== 工具函数 ====================

def collect_all_pairs(src_dir: Path):
    """
    收集 src_dir 下 images/{train,val} 与 labels/{train,val} 的全部文件对。
    返回: list of (image_path, label_path)
    """
    pairs = []
    for split in ('train', 'val'):
        img_dir   = src_dir / 'images' / split
        label_dir = src_dir / 'labels' / split
        if not img_dir.exists() or not label_dir.exists():
            print(f"⚠️  目录不存在，跳过: {img_dir} / {label_dir}")
            continue
        for label_file in sorted(label_dir.glob('*.txt')):
            # 寻找对应图片（支持常见后缀）
            img_file = None
            for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
                candidate = img_dir / (label_file.stem + ext)
                if candidate.exists():
                    img_file = candidate
                    break
            if img_file is None:
                print(f"⚠️  找不到图片，跳过标注: {label_file}")
                continue
            pairs.append((img_file, label_file))
    return pairs


def get_classes_in_label(label_path: Path):
    """读取一个 YOLO label 文件，返回其中出现的类别 ID 集合。"""
    class_ids = set()
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    class_ids.add(int(line.split()[0]))
    except Exception as e:
        print(f"⚠️  读取标注失败 {label_path}: {e}")
    return class_ids


def stratified_split(pairs, val_ratio=0.2, seed=42):
    """
    按类别做分层划分：
    - 对每个类别，将包含该类别的图片按 val_ratio 分配到 val，其余到 train。
    - 一张图片可能含多个类别，优先保证稀少类别的 val 配额，
      用已分配状态避免重复计入。
    返回: (train_pairs, val_pairs)
    """
    random.seed(seed)

    # 构建 类别 -> [pair_index, ...] 的索引
    class_to_indices = defaultdict(list)
    for idx, (img, label) in enumerate(pairs):
        for cid in get_classes_in_label(label):
            class_to_indices[cid].append(idx)

    # 按各类别样本数从少到多排序，确保稀少类别优先得到 val 保障
    sorted_classes = sorted(class_to_indices.keys(),
                            key=lambda c: len(class_to_indices[c]))

    assigned_val   = set()   # 已分配到 val 的 pair 索引
    assigned_train = set()   # 已分配到 train 的 pair 索引

    for cid in sorted_classes:
        indices = class_to_indices[cid].copy()
        random.shuffle(indices)

        # 统计该类别尚未分配的样本
        unassigned = [i for i in indices
                      if i not in assigned_val and i not in assigned_train]

        # 需要分配到 val 的数量（向上取整，至少保证 1 张）
        total_for_class = len(indices)
        need_val = max(1, round(total_for_class * val_ratio))

        # 先从未分配的里取 val
        already_val = [i for i in indices if i in assigned_val]
        still_need  = max(0, need_val - len(already_val))

        new_val = unassigned[:still_need]
        new_train = unassigned[still_need:]

        assigned_val.update(new_val)
        assigned_train.update(new_train)

    # 未被任何类别覆盖的（空标注文件）全部归入 train
    all_indices = set(range(len(pairs)))
    unhandled = all_indices - assigned_val - assigned_train
    assigned_train.update(unhandled)

    train_pairs = [pairs[i] for i in sorted(assigned_train)]
    val_pairs   = [pairs[i] for i in sorted(assigned_val)]
    return train_pairs, val_pairs


def copy_pairs(pairs, img_dst: Path, label_dst: Path):
    """将 (image, label) 对复制到目标目录。"""
    img_dst.mkdir(parents=True, exist_ok=True)
    label_dst.mkdir(parents=True, exist_ok=True)
    for img, label in pairs:
        shutil.copy2(img,   img_dst   / img.name)
        shutil.copy2(label, label_dst / label.name)


def print_split_stats(train_pairs, val_pairs):
    """打印划分后各类别在 train / val 中的分布。"""
    def count_classes(pairs):
        counter = defaultdict(int)
        for _, label in pairs:
            for cid in get_classes_in_label(label):
                counter[cid] += 1
        return counter

    train_counts = count_classes(train_pairs)
    val_counts   = count_classes(val_pairs)
    all_cids     = sorted(set(train_counts) | set(val_counts))

    print("\n" + "="*70)
    print("📊 按类别划分结果统计（以图片数计，一图可含多类）")
    print("="*70)
    header = f"{'ID':<4} {'编码':<8} {'中文名':<10} {'Train':<8} {'Val':<8} {'Total':<8} {'Val%':<6}"
    print(header)
    print("-"*70)
    for cid in all_cids:
        code    = ID_TO_CODE.get(cid, f'cls{cid}')
        cn      = CLASS_NAMES_CN.get(code, '?')
        t       = train_counts.get(cid, 0)
        v       = val_counts.get(cid, 0)
        total   = t + v
        pct     = f"{v/total*100:.1f}%" if total else "N/A"
        print(f"{cid:<4} {code:<8} {cn:<10} {t:<8} {v:<8} {total:<8} {pct:<6}")
    print("-"*70)
    print(f"{'合计':<23} {len(train_pairs):<8} {len(val_pairs):<8} "
          f"{len(train_pairs)+len(val_pairs):<8}")
    print("="*70)


def create_yaml(dst_dir: Path):
    """在 dataset/ 下生成 dataset.yaml。"""
    class_names = [ID_TO_CODE[i] for i in range(len(CLASSES))]
    yaml_lines = [
        "# YOLO 数据集配置（按类别分层划分版）",
        f"path: {dst_dir.absolute()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(CLASSES)}",
        f"names: {class_names}",
        "",
        "# 类别说明",
    ]
    for code, cid in sorted(CLASSES.items(), key=lambda x: x[1]):
        cn = CLASS_NAMES_CN.get(code, '')
        yaml_lines.append(f"# {cid}: {code} ({cn})")

    yaml_path = dst_dir / 'dataset.yaml'
    yaml_path.write_text('\n'.join(yaml_lines) + '\n', encoding='utf-8')
    print(f"✅ dataset.yaml 已生成: {yaml_path}")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description='按类别分层重新划分 YOLO 数据集')
    parser.add_argument('--src', default='yolo_dataset',
                        help='已转换好的 yolo_dataset 目录（默认: yolo_dataset）')
    parser.add_argument('--dst', default='dataset',
                        help='输出目录（默认: dataset）')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='验证集比例（默认: 0.2）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    print("="*60)
    print("🚀 按类别分层重新划分数据集")
    print("="*60)
    print(f"源目录  : {src_dir.absolute()}")
    print(f"目标目录: {dst_dir.absolute()}")
    print(f"Val 比例: {args.val_ratio} ({int(args.val_ratio*10)}:{int((1-args.val_ratio)*10)})")
    print(f"随机种子: {args.seed}")

    # 1. 收集所有文件对
    print("\n📂 读取已有数据...")
    all_pairs = collect_all_pairs(src_dir)
    if not all_pairs:
        print("❌ 未找到任何有效的图片-标注对，请检查源目录。")
        return
    print(f"共找到 {len(all_pairs)} 对图片-标注文件")

    # 2. 分层划分
    print("\n✂️  按类别进行分层划分...")
    train_pairs, val_pairs = stratified_split(
        all_pairs, val_ratio=args.val_ratio, seed=args.seed
    )

    # 3. 打印统计
    print_split_stats(train_pairs, val_pairs)

    # 4. 复制文件到 dataset/
    print(f"\n📁 写入目标目录: {dst_dir}")
    if dst_dir.exists():
        print(f"  ⚠️  目标目录已存在，将覆盖其中文件（不删除目录）")

    copy_pairs(train_pairs,
               dst_dir / 'images' / 'train',
               dst_dir / 'labels' / 'train')
    copy_pairs(val_pairs,
               dst_dir / 'images' / 'val',
               dst_dir / 'labels' / 'val')

    # 5. 复制 classes.txt（如存在）
    src_classes = src_dir / 'classes.txt'
    if src_classes.exists():
        shutil.copy2(src_classes, dst_dir / 'classes.txt')
        print(f"✅ classes.txt 已复制")

    # 6. 生成 dataset.yaml
    create_yaml(dst_dir)

    print("\n" + "="*60)
    print("✅ 完成！")
    print(f"   训练集: {len(train_pairs)} 张图片  →  {dst_dir}/images/train")
    print(f"   验证集: {len(val_pairs)} 张图片  →  {dst_dir}/images/val")
    print("="*60)


if __name__ == '__main__':
    main()