import os
import numpy as np
from ultralytics import YOLO
import ultralytics.data.dataset as dataset_module

# ── 路径配置 ─────────────────────────────────────────────────────────────────
DATA        = '/home/forge/workspace/yolo-Q4/yolo_dataset/dataset.yaml'
PROJECT     = 'runs/detect'
NAME        = 'road_damage_v1_yolo26n_weighted'
IMGSZ       = 640
DEVICE      = 0

# yolo26l 显存占用更大，batch 需要降低
# 如果 OOM 就改成 6 或 4
BATCH       = 32

# 两阶段 epoch 分配（总计 200）
# l 模型 head 更重，Stage1 给更多时间热身
STAGE1_EPOCHS = 80
STAGE2_EPOCHS = 120

# ── 公共超参 ─────────────────────────────────────────────────────────────────
# 观察 26s 的训练：cls_loss 仍然偏高，l 模型这里加到 2.0
# val/box_loss 无过拟合，数据增强可以保持甚至略微加强
COMMON_KWARGS = dict(
    data         = DATA,
    imgsz        = IMGSZ,
    batch        = BATCH,
    device       = DEVICE,
    project      = PROJECT,
    optimizer    = 'AdamW',
    weight_decay = 5e-4,

    # 类别不平衡：26s 跑了 146 epoch cls_loss 还在 3.8，加强惩罚
    cls          = 2.0,

    # 数据增强：26s 无过拟合迹象，l 模型容量更大，保持原配置
    mosaic       = 0.8,
    mixup        = 0.1,
    copy_paste   = 0.2,
    degrees      = 5.0,
    flipud       = 0.2,
    fliplr       = 0.5,
    scale        = 0.5,
    hsv_h        = 0.01,
    hsv_s        = 0.5,
    hsv_v        = 0.3,

    close_mosaic = 20,
    patience     = 30,       # l 模型收敛慢，patience 适当放宽
    save_period  = 10,
    plots        = True,
)
# ─────────────────────────────────────────────────────────────────────────────

class WeightedYOLODataset(dataset_module.YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 计算每个类的实例数
        self.counts = np.zeros(self.nc)
        for label in self.labels:
            cls_ids = label['cls'].flatten().astype(int)
            for c in cls_ids:
                self.counts[c] += 1
                
        # 计算图像权重（一张图含多个类时，取该图中最高权重，即最稀有类的权重）
        self.image_weights = np.zeros(len(self.labels))
        total = np.sum(self.counts)
        for i, label in enumerate(self.labels):
            cls_ids = label['cls'].flatten().astype(int)
            if len(cls_ids) > 0:
                weights = total / (self.counts[cls_ids] + 1e-6)   # 逆频率
                self.image_weights[i] = np.max(weights)            # 或 np.mean / np.sqrt 更温和
        
        self.image_weights = self.image_weights / self.image_weights.sum()
        print("类别实例数:", self.counts)
        print("Weighted Sampling 已启用（少数类图像采样概率更高）")

    def __getitem__(self, index):
        # 按权重重采样
        if hasattr(self, 'image_weights') and len(self.image_weights) > 0:
            index = np.random.choice(len(self.image_weights), p=self.image_weights)
        return super().__getitem__(index)

# 替换默认数据集类（猴子补丁，一行生效）
dataset_module.YOLODataset = WeightedYOLODataset

def stage1(base_weights: str) -> str:
    """
    冻结 backbone（freeze=10），只训练 neck + head。
    l 模型 head 参数量更多，给 80 epoch 充分热身。
    """
    print("\n" + "=" * 60)
    print(f"STAGE 1 — Frozen backbone  (epochs: {STAGE1_EPOCHS})")
    print("=" * 60)

    model = YOLO(base_weights)
    model.train(
        **COMMON_KWARGS,
        name          = NAME + '_s1',
        epochs        = STAGE1_EPOCHS,
        freeze        = 10,       # 冻结前 10 层（backbone）
        lr0           = 5e-4,     # 26s 用 1e-3，l 模型适当降低防震荡
        lrf           = 0.1,
        warmup_epochs = 5,        # l 模型热身轮数适当增加
    )

    best = os.path.join(PROJECT, NAME + '_s1', 'weights', 'best.pt')
    last = os.path.join(PROJECT, NAME + '_s1', 'weights', 'last.pt')
    ckpt = best if os.path.exists(best) else last
    print(f"\n[Stage 1 done] checkpoint → {ckpt}")
    return ckpt


def stage2(s1_ckpt: str):
    """
    解冻全部网络，小 lr 精调。
    26s 的训练显示 epoch 100+ 后 mAP 仍有提升空间，
    l 模型给 120 epoch 充分精调。
    """
    print("\n" + "=" * 60)
    print(f"STAGE 2 — Full fine-tune   (epochs: {STAGE2_EPOCHS})")
    print("=" * 60)

    model = YOLO(s1_ckpt)
    results = model.train(
                **COMMON_KWARGS,
        name          = NAME + '_s1',
        epochs        = 150,
        lr0           = 5e-4,     # 26s 用 1e-3，l 模型适当降低防震荡
        lrf           = 0.1,
        warmup_epochs = 20,        # l 模型热身轮数适当增加
        # **COMMON_KWARGS,
        # name          = NAME + '_s2',
        # epochs        = STAGE2_EPOCHS,
        # freeze        = 0,        # 全网解冻
        # lr0           = 5e-5,     # 解冻后 backbone 用更小 lr，防止破坏预训练特征
        # lrf           = 0.01,
        # warmup_epochs = 3,
    )
    return results


def train():
    # ── 断点续训优先级：S2 last → S1 best/last → 全新 ──────────────────────
    s2_last = os.path.join(PROJECT, 'runs/detect/' + NAME , 'weights', 'last.pt')
    # s1_best = os.path.join(PROJECT, NAME + '_s1', 'weights', 'best.pt')
    # s1_last = os.path.join(PROJECT, NAME + '_s1', 'weights', 'last.pt')

    if os.path.exists(s2_last):
        print(f"[Resume] Stage 2 中断，从 {s2_last} 恢复")
        model   = YOLO(s2_last)
        results = model.train(resume=True)
    else:  
        print("[New] 未发现 checkpoint，从头开始两阶段训练")
        results = stage2('yolo26s.pt')
        # results = stage2(s1_ckpt)

    # elif os.path.exists(s1_best) or os.path.exists(s1_last):
    #     ckpt = s1_best if os.path.exists(s1_best) else s1_last
    #     print(f"[Resume] Stage 1 已完成，直接进入 Stage 2 → {ckpt}")
    #     results = stage2(ckpt)

    # else:
    #     print("[New] 未发现 checkpoint，从头开始两阶段训练")
    #     s1_ckpt = stage1('yolo26l.pt')
    #     results = stage2(s1_ckpt)

    # ── 最终指标 ──────────────────────────────────────────────────────────────
    if results and hasattr(results, 'results_dict'):
        rd = results.results_dict
        print(f"\nBest mAP50    : {rd.get('metrics/mAP50(B)',    'N/A')}")
        print(f"Best mAP50-95 : {rd.get('metrics/mAP50-95(B)', 'N/A')}")

    return results


if __name__ == '__main__':
    train()