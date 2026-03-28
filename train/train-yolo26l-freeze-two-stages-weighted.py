import os
import numpy as np
import wandb
import ultralytics.data.dataset as dataset_module
from ultralytics import YOLO

# ── 路径配置 ──────────────────────────────────────────────────────────────────
DATA    = '/home/forge/workspace/yolo-Q4/yolo_dataset/dataset.yaml'
PROJECT = 'runs/detect'
NAME    = 'road_damage_v1_yolo26l-freeze-stage1-weighted'
IMGSZ   = 1024
DEVICE  = 0
BATCH   = 8

STAGE1_EPOCHS = 80
BASE_WEIGHTS  = 'yolo26l.pt'

# ── W&B 配置 ──────────────────────────────────────────────────────────────────
WANDB_PROJECT = 'road-damage-detection'
WANDB_RUN     = NAME

# ── 训练超参 ──────────────────────────────────────────────────────────────────
TRAIN_KWARGS = dict(
    data         = DATA,
    imgsz        = IMGSZ,
    batch        = BATCH,
    device       = DEVICE,
    project      = PROJECT,
    name         = NAME,
    optimizer    = 'AdamW',
    weight_decay = 5e-4,
    epochs       = STAGE1_EPOCHS,
    freeze       = 10,        # 冻结前 10 层（backbone）
    lr0          = 5e-4,
    lrf          = 0.1,
    warmup_epochs= 5,

    cls          = 2.0,

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
    patience     = 40,
    save_period  = 10,
    plots        = True,
)


# ── Weighted Sampling ─────────────────────────────────────────────────────────
class WeightedYOLODataset(dataset_module.YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.counts = np.zeros(self.nc)
        for label in self.labels:
            cls_ids = label['cls'].flatten().astype(int)
            for c in cls_ids:
                self.counts[c] += 1

        self.image_weights = np.zeros(len(self.labels))
        total = np.sum(self.counts)
        for i, label in enumerate(self.labels):
            cls_ids = label['cls'].flatten().astype(int)
            if len(cls_ids) > 0:
                weights = total / (self.counts[cls_ids] + 1e-6)
                self.image_weights[i] = np.max(weights)

        self.image_weights = self.image_weights / self.image_weights.sum()
        print("类别实例数:", self.counts)
        print("Weighted Sampling 已启用（少数类图像采样概率更高）")

    def __getitem__(self, index):
        if hasattr(self, 'image_weights') and len(self.image_weights) > 0:
            index = np.random.choice(len(self.image_weights), p=self.image_weights)
        return super().__getitem__(index)

dataset_module.YOLODataset = WeightedYOLODataset


# ── W&B 初始化 ────────────────────────────────────────────────────────────────
def init_wandb():
    wandb.init(
        project = WANDB_PROJECT,
        name    = WANDB_RUN,
        resume  = 'allow',
        config  = {
            **TRAIN_KWARGS,
            'model': 'yolo26l',
            'stage': 1,
        }
    )


# ── 训练入口 ──────────────────────────────────────────────────────────────────
def train():
    # init_wandb()

    s1_last = os.path.join(PROJECT, 'runs/detect/' + NAME, 'weights', 'last.pt')
    s1_best = os.path.join(PROJECT, 'runs/detect/' + NAME, 'weights', 'best.pt')

    if os.path.exists(s1_last):
        print(f"[Resume] Stage 1 中断，从 {s1_last} 恢复")
        model   = YOLO(s1_last)
        results = model.train(resume=True)
    else:
        print(f"[New] 从 {BASE_WEIGHTS} 开始 Stage 1 训练")
        model   = YOLO(BASE_WEIGHTS)
        results = model.train(**TRAIN_KWARGS)

    # ── 最终指标 ──────────────────────────────────────────────────────────────
    if results and hasattr(results, 'results_dict'):
        rd = results.results_dict
        print(f"\nBest mAP50    : {rd.get('metrics/mAP50(B)',    'N/A')}")
        print(f"Best mAP50-95 : {rd.get('metrics/mAP50-95(B)', 'N/A')}")

    best = os.path.join(PROJECT, NAME, 'weights', 'best.pt')
    last = os.path.join(PROJECT, NAME, 'weights', 'last.pt')
    ckpt = best if os.path.exists(best) else last
    print(f"\n[Stage 1 done] checkpoint → {ckpt}")

    # wandb.finish()
    return results


if __name__ == '__main__':
    train()