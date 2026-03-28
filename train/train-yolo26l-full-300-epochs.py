import os
from ultralytics import YOLO
import wandb

# ── 路径配置 ─────────────────────────────────────────────────────────────────
DATA         = '/home/featurize/data/dataset/dataset.yaml'
PROJECT      = 'runs/detect'
NAME         = 'road_damage_v1_yolo26l-full'
SAVE_DIR     = 'runs/detect'
IMGSZ        = 1024
DEVICE       = 0
BATCH        = 32        # 不冻结时显存压力更大，适当减小
EPOCHS       = 300

WANDB_PROJECT = 'road-damage-detection'
WANDB_RUN     = NAME

# ── 训练超参 ─────────────────────────────────────────────────────────────────
# 不冻结：backbone 参与训练，lr 要保守 + 更强正则
TRAIN_KWARGS = dict(
    data         = DATA,
    imgsz        = IMGSZ,
    batch        = BATCH,
    device       = DEVICE,
    project      = PROJECT,
    name         = NAME,
    epochs       = EPOCHS,
    optimizer    = 'AdamW',
    weight_decay = 1e-3,     # 冻结版 5e-4 → 全量训练适当增大
    # freeze     = 不设置，默认不冻结
    lr0          = 5e-4,     # 冻结版 1e-3 → 全量训练更保守
    lrf          = 0.01,
    warmup_epochs= 10,
    cls          = 2.0,
    mosaic       = 0.8,
    mixup        = 0.15,     # 略微增强，全网络对混合样本更鲁棒
    copy_paste   = 0.3,      # 小目标类别多，适当增强
    degrees      = 5.0,
    flipud       = 0.2,
    fliplr       = 0.5,
    scale        = 0.6,      # 稍加强多尺度
    hsv_h        = 0.015,
    hsv_s        = 0.6,
    hsv_v        = 0.4,
    close_mosaic = 30,
    patience     = 60,
    save_period  = 10,
    plots        = True,
)

# ── W&B 自定义 callback ──────────────────────────────────────────────────────
class WandbCallback:
    """每个 epoch 结束后把 metrics/lr/val loss 全量推送到 W&B。"""

    # 从 ultralytics results.csv 的列名到 W&B 显示名的映射
    METRIC_KEYS = {
        'metrics/precision(B)' : 'val/precision',
        'metrics/recall(B)'    : 'val/recall',
        'metrics/mAP50(B)'     : 'val/mAP50',
        'metrics/mAP50-95(B)'  : 'val/mAP50-95',
        'train/box_loss'       : 'train/box_loss',
        'train/cls_loss'       : 'train/cls_loss',
        'train/dfl_loss'       : 'train/dfl_loss',
        'val/box_loss'         : 'val/box_loss',
        'val/cls_loss'         : 'val/cls_loss',
        'val/dfl_loss'         : 'val/dfl_loss',
    }

    def on_train_epoch_end(self, trainer):
        """每 epoch 训练完后记录 train loss + lr。"""
        log = {'epoch': trainer.epoch}

        # 学习率（支持多组 param_group）
        if hasattr(trainer, 'optimizer') and trainer.optimizer:
            for i, pg in enumerate(trainer.optimizer.param_groups):
                log[f'lr/pg{i}'] = pg['lr']

        # train loss
        if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
            names = getattr(trainer, 'loss_names',
                            ['box', 'cls', 'dfl'])
            for name, val in zip(names, trainer.loss_items):
                log[f'train/{name}_loss'] = float(val)

        wandb.log(log, step=trainer.epoch)

    def on_val_end(self, validator):
        """每次验证完后记录 val metrics + val loss。"""
        log = {}

        # metrics dict（precision / recall / mAP）
        if hasattr(validator, 'metrics') and validator.metrics:
            rd = validator.metrics.results_dict
            for src_key, dst_key in self.METRIC_KEYS.items():
                if src_key in rd:
                    log[dst_key] = rd[src_key]

        # per-class AP（如果有）
        if hasattr(validator, 'metrics') and \
           hasattr(validator.metrics, 'ap_class_index'):
            names = validator.names  # {0: 'D00', 1: 'D10', ...}
            for i, cls_idx in enumerate(validator.metrics.ap_class_index):
                cls_name = names.get(int(cls_idx), str(cls_idx))
                ap50 = validator.metrics.ap50()[i]
                log[f'val/AP50_{cls_name}'] = float(ap50)

        # val loss
        if hasattr(validator, 'loss') and validator.loss is not None:
            losses = validator.loss.cpu().numpy()
            loss_names = getattr(validator, 'loss_names',
                                 ['box', 'cls', 'dfl'])
            for name, val in zip(loss_names, losses):
                log[f'val/{name}_loss_raw'] = float(val)

        if log:
            epoch = getattr(validator, 'epoch',
                            getattr(validator, 'trainer', None) and
                            validator.trainer.epoch or 0)
            wandb.log(log, step=epoch)

    def on_train_end(self, trainer):
        """训练完毕：上传 best.pt + confusion matrix。"""
        best = os.path.join(trainer.save_dir, 'weights', 'best.pt')
        if os.path.exists(best):
            artifact = wandb.Artifact(
                name=f'{NAME}-best',
                type='model',
                description='Best checkpoint by val mAP50-95',
            )
            artifact.add_file(best)
            wandb.log_artifact(artifact)

        # confusion matrix（ultralytics 保存在 save_dir/confusion_matrix.png）
        cm_path = os.path.join(trainer.save_dir, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            wandb.log({'confusion_matrix': wandb.Image(cm_path)})

        # PR / F1 curves（如果有）
        for fname in ['PR_curve.png', 'F1_curve.png',
                      'P_curve.png', 'R_curve.png']:
            fpath = os.path.join(trainer.save_dir, fname)
            if os.path.exists(fpath):
                wandb.log({fname.replace('.png', ''): wandb.Image(fpath)})


# ─────────────────────────────────────────────────────────────────────────────
def init_wandb(resume: bool = False):
    wandb.init(
        project = WANDB_PROJECT,
        name    = WANDB_RUN,
        resume  = 'allow',
        config  = {
            **TRAIN_KWARGS,
            'model'  : 'yolo26l',
            'freeze' : 0,          # 明确标注不冻结
        }
    )


def train():
    last = os.path.join(SAVE_DIR, NAME, 'weights', 'last.pt')
    cb   = WandbCallback()

    if os.path.exists(last):
        print(f"[Resume] 从断点恢复 → {last}")
        init_wandb(resume=True)
        model = YOLO(last)
        model.add_callback('on_train_epoch_end', cb.on_train_epoch_end)
        model.add_callback('on_val_end',         cb.on_val_end)
        model.add_callback('on_train_end',        cb.on_train_end)
        results = model.train(resume=True)
    else:
        print("[New] 未发现 checkpoint，从头开始训练")
        init_wandb(resume=False)
        model = YOLO('yolo26l.pt')
        model.add_callback('on_train_epoch_end', cb.on_train_epoch_end)
        model.add_callback('on_val_end',         cb.on_val_end)
        model.add_callback('on_train_end',        cb.on_train_end)
        results = model.train(**TRAIN_KWARGS)

    # ── 最终 summary ─────────────────────────────────────────────────────────
    if results and hasattr(results, 'results_dict'):
        rd = results.results_dict
        map50   = rd.get('metrics/mAP50(B)',    'N/A')
        map5095 = rd.get('metrics/mAP50-95(B)', 'N/A')
        print(f"\nBest mAP50    : {map50}")
        print(f"Best mAP50-95 : {map5095}")
        wandb.summary['best_mAP50']    = map50
        wandb.summary['best_mAP50-95'] = map5095

    wandb.finish()
    return results


if __name__ == '__main__':
    train()