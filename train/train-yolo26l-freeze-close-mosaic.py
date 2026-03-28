import os
from xml.parsers.expat import model
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
TRAIN_KWARGS_FINETUNE = dict(
    data         = DATA,
    imgsz        = IMGSZ,
    batch        = BATCH,
    device       = DEVICE,
    project      = PROJECT,
    name         = NAME + '_closemosaic',
    epochs       = 30,           # 只跑 close_mosaic 阶段
    optimizer    = 'AdamW',
    weight_decay = 5e-4,
    lr0          = 1e-4,         # 更低的 lr，已经接近收敛
    lrf          = 0.01,
    cls          = 2.0,
    mosaic       = 0.0,          # 直接关掉 mosaic，模拟 close_mosaic 效果
    mixup        = 0.0,
    copy_paste   = 0.0,
    close_mosaic = 0,
    patience     = 0,
    save_period  = 5,
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
    model = YOLO('/home/featurize/work/yolo-Q4/runs/detect/runs/detect/road_damage_v1_yolo26l-freeze/weights/best.pt')
    results = model.train(**TRAIN_KWARGS_FINETUNE)
  

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