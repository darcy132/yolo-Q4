import os
from ultralytics import YOLO

# ── 路径配置 ─────────────────────────────────────────────────────────────────
DATA        = '/home/forge/workspace/yolo-Q4/yolo_dataset/dataset.yaml'
PROJECT     = 'runs/detect'
NAME        = 'road_damage_v1_yolo26l-freeze'
IMGSZ       = 1024
DEVICE      = 0
BATCH       = 4
EPOCHS      = 80

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
# ─────────────────────────────────────────────────────────────────────────────


def train():
    last = os.path.join(PROJECT, NAME, 'weights', 'last.pt')

    if os.path.exists(last):
        print(f"[Resume] 从断点恢复 → {last}")
        model   = YOLO(last)
        results = model.train(resume=True)
    else:
        print("[New] 未发现 checkpoint，从头开始训练")
        model   = YOLO('yolo26l.pt')
        results = model.train(**TRAIN_KWARGS)

    if results and hasattr(results, 'results_dict'):
        rd = results.results_dict
        print(f"\nBest mAP50    : {rd.get('metrics/mAP50(B)',    'N/A')}")
        print(f"Best mAP50-95 : {rd.get('metrics/mAP50-95(B)', 'N/A')}")

    return results


if __name__ == '__main__':
    train()