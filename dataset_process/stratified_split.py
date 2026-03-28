"""
stratified_split.py
--------------------
从已转换好的 yolo_dataset (images/train + images/val, labels/train + labels/val)
读取全部数据，按 **每个类别** 做分层划分，
并严格保证最终：train = 3200 张，val = 800 张（共 4000 张）。

划分策略：
  1. 对每个类别按 8:2 比例分配 val 配额（少数类至少保证 1 张）。
  2. 从少到多处理各类别，优先保障稀少类在 val 中有代表性。
  3. 完成分层分配后，若 val 数量 != 800，从 train 里随机补足
     （或将 val 多余部分移回 train），保证总数严格正确。
  4. 最终 train 严格 = 3200，val 严格 = 800，且两者无重叠。

用法:
    python stratified_split.py
    python stratified_split.py --src /path/to/yolo_dataset --dst /path/to/dataset
    python stratified_split.py --train-count 3200 --val-count 800
"""

import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

# ==================== 类别定义 ====================
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
    """收集 src_dir 下 train+val 全部 (image_path, label_path) 对。"""
    pairs = []
    for split in ('train', 'val'):
        img_dir   = src_dir / 'images' / split
        label_dir = src_dir / 'labels' / split
        if not img_dir.exists() or not label_dir.exists():
            print(f"⚠️  目录不存在，跳过: {split}")
            continue
        for label_file in sorted(label_dir.glob('*.txt')):
            img_file = None
            for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
                candidate = img_dir / (label_file.stem + ext)
                if candidate.exists():
                    img_file = candidate
                    break
            if img_file is None:
                print(f"⚠️  找不到图片，跳过: {label_file.name}")
                continue
            pairs.append((img_file, label_file))
    return pairs


def get_classes_in_label(label_path: Path):
    """返回一个 label 文件中出现的类别 ID 集合。"""
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


def stratified_split_indices(pairs, val_ratio: float, seed: int):
    """
    按类别分层划分，返回 (val_indices_set, train_indices_set)。
    数量在此步可能不严格等于目标值，后续 adjust 步骤修正。
    """
    random.seed(seed)

    # 类别 -> [pair 索引]
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(pairs):
        for cid in get_classes_in_label(label):
            class_to_indices[cid].append(idx)

    # 稀少类别优先处理，保证其在 val 中有代表
    sorted_classes = sorted(class_to_indices.keys(),
                            key=lambda c: len(class_to_indices[c]))

    assigned_val   = set()
    assigned_train = set()

    for cid in sorted_classes:
        indices = class_to_indices[cid].copy()
        random.shuffle(indices)

        total_for_class = len(indices)
        need_val = max(1, round(total_for_class * val_ratio))

        already_val = [i for i in indices if i in assigned_val]
        unassigned  = [i for i in indices
                       if i not in assigned_val and i not in assigned_train]

        still_need = max(0, need_val - len(already_val))
        new_val    = unassigned[:still_need]
        new_train  = unassigned[still_need:]

        assigned_val.update(new_val)
        assigned_train.update(new_train)

    # 空标注或未被任何类别覆盖的归入 train
    all_indices = set(range(len(pairs)))
    unhandled   = all_indices - assigned_val - assigned_train
    assigned_train.update(unhandled)

    return assigned_val, assigned_train


def adjust_to_exact_counts(val_set: set, train_set: set,
                           target_val: int, target_train: int, seed: int):
    """
    精确调整到目标数量：
    - val 过多 → 随机将多余的移回 train
    - val 过少 → 随机从 train 中补入 val
    随机操作保证不破坏已有的分层均衡（微小调整）。
    """
    rng = random.Random(seed + 99)

    # val 过多，移回 train
    while len(val_set) > target_val:
        idx = rng.choice(sorted(val_set))
        val_set.remove(idx)
        train_set.add(idx)

    # val 过少，从 train 补
    while len(val_set) < target_val:
        idx = rng.choice(sorted(train_set))
        train_set.remove(idx)
        val_set.add(idx)

    return val_set, train_set


def copy_pairs(pairs, img_dst: Path, label_dst: Path):
    img_dst.mkdir(parents=True, exist_ok=True)
    label_dst.mkdir(parents=True, exist_ok=True)
    for img, label in pairs:
        shutil.copy2(img,   img_dst   / img.name)
        shutil.copy2(label, label_dst / label.name)


def print_split_stats(train_pairs, val_pairs):
    """打印各类别在 train/val 中的图片数分布。"""
    def count_classes(pairs):
        counter = defaultdict(int)
        for _, label in pairs:
            for cid in get_classes_in_label(label):
                counter[cid] += 1
        return counter

    train_counts = count_classes(train_pairs)
    val_counts   = count_classes(val_pairs)
    all_cids     = sorted(set(train_counts) | set(val_counts))

    print("\n" + "="*72)
    print("📊 按类别划分结果（图片数，一张图可含多个类别）")
    print("="*72)
    print(f"{'ID':<4} {'编码':<8} {'中文名':<10} {'Train':>7} {'Val':>6} {'Total':>7} {'Val%':>6}")
    print("-"*72)
    for cid in all_cids:
        code  = ID_TO_CODE.get(cid, f'cls{cid}')
        cn    = CLASS_NAMES_CN.get(code, '?')
        t     = train_counts.get(cid, 0)
        v     = val_counts.get(cid, 0)
        total = t + v
        pct   = f"{v/total*100:.1f}%" if total else "N/A"
        print(f"{cid:<4} {code:<8} {cn:<10} {t:>7} {v:>6} {total:>7} {pct:>6}")
    print("-"*72)
    print(f"{'图片总数合计':<23} {len(train_pairs):>7} {len(val_pairs):>6} "
          f"{len(train_pairs)+len(val_pairs):>7}")
    print("="*72)


def create_yaml(dst_dir: Path):
    class_names = [ID_TO_CODE[i] for i in range(len(CLASSES))]
    lines = [
        "# YOLO 数据集配置（按类别分层划分，严格 3200/800）",
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
        lines.append(f"# {cid}: {code}  ({cn})")
    (dst_dir / 'dataset.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"✅ dataset.yaml 已生成")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description='按类别分层 + 严格数量 重新划分 YOLO 数据集'
    )
    parser.add_argument('--src',         default='yolo_dataset',
                        help='源 yolo_dataset 目录（默认: yolo_dataset）')
    parser.add_argument('--dst',         default='dataset',
                        help='输出目录（默认: dataset/）')
    parser.add_argument('--train-count', type=int, default=3200,
                        help='训练集严格数量（默认: 3200）')
    parser.add_argument('--val-count',   type=int, default=800,
                        help='验证集严格数量（默认: 800）')
    parser.add_argument('--seed',        type=int, default=42,
                        help='随机种子（默认: 42）')
    args = parser.parse_args()

    src_dir      = Path(args.src)
    dst_dir      = Path(args.dst)
    target_train = args.train_count
    target_val   = args.val_count
    target_total = target_train + target_val

    print("="*60)
    print("🚀 按类别分层重新划分数据集（严格数量版）")
    print("="*60)
    print(f"源目录   : {src_dir.absolute()}")
    print(f"目标目录 : {dst_dir.absolute()}")
    print(f"目标数量 : train={target_train}  val={target_val}  total={target_total}")
    print(f"随机种子 : {args.seed}")

    # 1. 收集全部文件对
    print("\n📂 读取已有数据...")
    all_pairs = collect_all_pairs(src_dir)
    if not all_pairs:
        print("❌ 未找到任何有效图片-标注对，请检查源目录。")
        return

    actual_total = len(all_pairs)
    print(f"共找到 {actual_total} 对图片-标注文件")

    if actual_total != target_total:
        print(f"\n❌ 错误：数据集共 {actual_total} 张，"
              f"但目标要求恰好 {target_total} 张（{target_train} + {target_val}）。")
        print("   请确认源目录数据完整，或通过 --train-count / --val-count 调整目标数量。")
        return

    # 2. 分层划分（初步，数量可能不严格）
    print("\n✂️  按类别进行分层划分...")
    val_ratio = target_val / target_total
    val_set, train_set = stratified_split_indices(all_pairs, val_ratio, args.seed)
    print(f"   初步分层：train={len(train_set)}  val={len(val_set)}")

    # 3. 精确调整到目标数量
    if len(val_set) != target_val:
        diff = len(val_set) - target_val
        direction = "减少" if diff > 0 else "增加"
        print(f"   ⚙️  val {direction} {abs(diff)} 张以满足严格数量要求...")
        val_set, train_set = adjust_to_exact_counts(
            val_set, train_set,
            target_val=target_val, target_train=target_train,
            seed=args.seed
        )

    # 4. 严格验证
    assert len(val_set)   == target_val,   f"val 数量异常: {len(val_set)}"
    assert len(train_set) == target_train, f"train 数量异常: {len(train_set)}"
    assert len(val_set & train_set) == 0,  "val 与 train 存在重叠！"
    print(f"   ✅ 验证通过：train={len(train_set)}  val={len(val_set)}")

    # 5. 整理为列表（按索引排序保证可复现）
    train_pairs = [all_pairs[i] for i in sorted(train_set)]
    val_pairs   = [all_pairs[i] for i in sorted(val_set)]

    # 6. 打印各类别分布统计
    print_split_stats(train_pairs, val_pairs)

    # 7. 复制文件到 dataset/
    print(f"\n📁 写入目标目录: {dst_dir}")
    if dst_dir.exists():
        print(f"   ⚠️  目标目录已存在，将覆盖其中同名文件")

    copy_pairs(train_pairs, dst_dir / 'images' / 'train', dst_dir / 'labels' / 'train')
    copy_pairs(val_pairs,   dst_dir / 'images' / 'val',   dst_dir / 'labels' / 'val')

    # 8. 复制 classes.txt（如存在）
    src_classes = src_dir / 'classes.txt'
    if src_classes.exists():
        shutil.copy2(src_classes, dst_dir / 'classes.txt')
        print(f"✅ classes.txt 已复制")

    # 9. 生成 dataset.yaml
    create_yaml(dst_dir)

    print("\n" + "="*60)
    print("✅ 完成！")
    print(f"   训练集 : {len(train_pairs)} 张  →  {dst_dir}/images/train")
    print(f"   验证集 : {len(val_pairs)} 张  →  {dst_dir}/images/val")
    print("="*60)


if __name__ == '__main__':
    main()