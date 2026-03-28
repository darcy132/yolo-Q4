import os
from ultralytics import YOLO
import wandb

# ── 路径配置 ─────────────────────────────────────────────────────────────────
DATA        = '/home/featurize/data/dataset/dataset.yaml'
PROJECT     = 'runs/detect'
NAME        = 'road_damage_v1_yolo26l-freeze'
IMGSZ       = 1024
DEVICE      = 0
BATCH       = 64
EPOCHS      = 300

# ── 训练超参 ─────────────────────────────────────────────────────────────────
TRAIN_KWARGS = dict(
    data         = DATA,
    imgsz        = IMGSZ,
    batch        = BATCH,
    device       = DEVICE,
    project      = PROJECT,
    name         = NAME,
    epochs       = EPOCHS,
    optimizer    = 'AdamW',
    weight_decay = 5e-4,
    freeze       = 10,
    lr0          = 1e-3,
    lrf          = 0.01,
    warmup_epochs= 10,
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
    close_mosaic = 30,
    patience     = 60,
    save_period  = 10,
    plots        = True,
)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
def init_wandb(resume: bool = False):
    """初始化 W&B，支持断点续训时恢复同一个 run"""
    wandb.init(
        project   = WANDB_PROJECT,
        name      = WANDB_RUN,
        # entity  = WANDB_ENTITY,   # 指定团队时取消注释
        resume    = 'allow',        # 断点续训时自动续接同名 run
        config    = {
            **TRAIN_KWARGS,
            'model': 'yolo26l',
        }
    )

def train():
    last = os.path.join(SAVE_DIR, NAME, 'weights', 'last.pt')

    if os.path.exists(last):
        print(f"[Resume] 从断点恢复 → {last}")
        init_wandb(resume=True)
        model   = YOLO(last)
        results = model.train(resume=True)
    else:
        print("[New] 未发现 checkpoint，从头开始训练")
        init_wandb(resume=False)
        model   = YOLO('yolo26l.pt')
        results = model.train(**TRAIN_KWARGS)

    # ── 打印最终指标 ──────────────────────────────────────────────────────────
    if results and hasattr(results, 'results_dict'):
        rd = results.results_dict
        map50    = rd.get('metrics/mAP50(B)',    'N/A')
        map5095  = rd.get('metrics/mAP50-95(B)', 'N/A')
        print(f"\nBest mAP50    : {map50}")
        print(f"Best mAP50-95 : {map5095}")

        # 同步最终指标到 W&B
        wandb.summary['best_mAP50']    = map50
        wandb.summary['best_mAP50-95'] = map5095

    wandb.finish()
    return results


if __name__ == '__main__':
    train()