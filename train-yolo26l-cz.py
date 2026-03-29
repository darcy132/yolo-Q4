"""
train_cz_yolo26l.py
路面车辙 (cz) 二分类检测 —— YOLO26l 训练脚本

YOLO26 关键特性:
  - NMS-free 端到端推理 (无需 NMS 后处理)
  - 移除 DFL，简化 bounding box 回归
  - MuSGD 优化器 (SGD + Muon 混合，更稳定的收敛)
  - ProgLoss + STAL (小目标感知标签分配)

用法:
  python train_cz_yolo26l.py                    # 默认配置
  python train_cz_yolo26l.py --epochs 200       # 自定义 epoch
  python train_cz_yolo26l.py --resume            # 断点续训
  python train_cz_yolo26l.py --device 0,1        # 多卡
"""

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CFG = dict(
    # ── 数据 ──
    data        = "./dataset_cz/dataset.yaml",

    # ── 模型 ──
    model       = "yolo26l.pt",

    # ── 基础训练 ──
    epochs      = 200,          # 车辙特征细微，适当延长收敛时间
    imgsz       = 1280,         # ★ 提高分辨率：车辙细长，640 下细节丢失严重
    batch       = 8,            # imgsz 提高后显存压力上升，对应降 batch
    workers     = 8,

    # ── 优化器 ──
    optimizer        = "AdamW",
    lr0              = 0.001,   # ★ 降低：AdamW 配合大 imgsz，lr 不宜过高
    lrf              = 0.01,
    momentum         = 0.937,
    weight_decay     = 0.0005,
    warmup_epochs    = 5.0,     # 适当延长 warmup，分辨率大时更稳定
    warmup_momentum  = 0.8,
    warmup_bias_lr   = 0.1,

    # ── 损失权重 ──
    box         = 9.0,          # ★ 提高：车辙位置回归精度要求高
    cls         = 0.3,          # ★ 降低：只有一类（或少类），分类损失权重不宜过大

    # ── 数据增强 ──
    hsv_h       = 0.01,         # ★ 降低：路面色相变化很小，不需要大范围偏移
    hsv_s       = 0.5,          # 适当降低，路面饱和度本身就低
    hsv_v       = 0.5,          # ★ 略微提高：模拟光照变化（阴影、逆光）
    degrees     = 3.0,          # ★ 降低：车辙方向性强，大旋转会破坏特征
    translate   = 0.1,
    scale       = 0.3,          # ★ 降低：避免缩放后车辙细节过小或被裁断
    shear       = 1.0,          # ★ 降低：剪切会破坏车辙的平行结构
    perspective = 0.00005,      # ★ 降低：透视畸变对细长目标影响大
    flipud      = 0.2,          # ★ 略微提高：路面图像上下翻转依然合理
    fliplr      = 0.5,
    mosaic      = 0.8,          # ★ 适当降低：mosaic 拼接可能截断长车辙
    mixup       = 0.05,         # ★ 大幅降低：mixup 对细长低对比度目标干扰大
    copy_paste  = 0.0,          # ★ 关闭：车辙粘贴到随机位置会引入错误上下文

    # ── 验证 & 保存 ──
    val         = True,
    save        = True,
    save_period = 10,
    patience    = 40,           # ★ 延长：车辙收敛曲线往往有平台期，不要过早停止
    close_mosaic= 20,           # ★ 新增：最后 20 epoch 关闭 mosaic，稳定收敛

    # ── 日志 ──
    project     = "runs/cz_detection",
    name        = "yolo26l_rut_v1",
    exist_ok    = False,
    plots       = True,
    verbose     = True,
)


def check_ultralytics_version():
    """检查 ultralytics 是否安装，版本是否支持 YOLO26"""
    try:
        import ultralytics
        ver = tuple(int(x) for x in ultralytics.__version__.split(".")[:2])
        print(f"[INFO] ultralytics {ultralytics.__version__} 已安装")
        # YOLO26 在 8.3.x+ 引入
        if ver < (8, 3):
            print(f"[WARN] YOLO26 需要 ultralytics >= 8.3，当前 {ultralytics.__version__}")
            print("       请运行: pip install -U ultralytics")
    except ImportError:
        print("[ERROR] 未找到 ultralytics，请先安装:")
        print("        pip install ultralytics")
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="YOLO26l cz 路面车辙训练脚本")
    p.add_argument("--data",     default=DEFAULT_CFG["data"],    help="数据集 yaml 路径")
    p.add_argument("--model",    default=DEFAULT_CFG["model"],   help="模型权重 (yolo26n/s/m/l/x.pt)")
    p.add_argument("--epochs",   type=int, default=DEFAULT_CFG["epochs"])
    p.add_argument("--imgsz",    type=int, default=DEFAULT_CFG["imgsz"])
    p.add_argument("--batch",    default=DEFAULT_CFG["batch"],   help="batch size，'auto' 自动推断")
    p.add_argument("--device",   default=None,                   help="'cpu', '0', '0,1' 等")
    p.add_argument("--workers",  type=int, default=DEFAULT_CFG["workers"])
    p.add_argument("--project",  default=DEFAULT_CFG["project"])
    p.add_argument("--name",     default=DEFAULT_CFG["name"])
    p.add_argument("--resume",   action="store_true",            help="从 last.pt 断点续训")
    p.add_argument("--exist-ok", action="store_true",            help="允许覆盖已有实验目录")
    p.add_argument("--amp",      action="store_true", default=True, help="开启混合精度 (默认开)")
    p.add_argument("--freeze",   type=int, default=0,            help="冻结前 N 层 backbone (0=不冻结)")
    return p.parse_args()


def build_train_kwargs(args) -> dict:
    """合并默认配置与命令行参数"""
    cfg = DEFAULT_CFG.copy()
    cfg.update({
        "data":     args.data,
        "model":    args.model,
        "epochs":   args.epochs,
        "imgsz":    args.imgsz,
        "batch":    args.batch,
        "workers":  args.workers,
        "project":  args.project,
        "name":     args.name,
        "exist_ok": args.exist_ok,
        "amp":      args.amp,
        "freeze":   args.freeze,
    })
    if args.device is not None:
        cfg["device"] = args.device
    if args.resume:
        # 断点续训：找最近的 last.pt
        last_pt = Path(args.project) / args.name / "weights" / "last.pt"
        if last_pt.exists():
            cfg["model"]  = str(last_pt)
            cfg["resume"] = True
            print(f"[INFO] 断点续训: {last_pt}")
        else:
            print(f"[WARN] 找不到 {last_pt}，从头开始训练")
    return cfg


def train(cfg: dict):
    from ultralytics import YOLO

    print("\n" + "="*60)
    print(f"  模型 : {cfg['model']}")
    print(f"  数据 : {cfg['data']}")
    print(f"  轮次 : {cfg['epochs']}  |  批次 : {cfg['batch']}  |  分辨率 : {cfg['imgsz']}")
    print(f"  保存 : {cfg['project']}/{cfg['name']}")
    print("="*60 + "\n")

    model = YOLO(cfg.pop("model"))

    results = model.train(**cfg)
    return results, model


def post_train_eval(model, data_yaml: str):
    """训练完成后在验证集上跑一次完整评估"""
    print("\n[INFO] 训练完成，执行最终验证 ...")
    metrics = model.val(data=data_yaml, split="val")
    print(f"\n[RESULT] mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"[RESULT] mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"[RESULT] Precision   : {metrics.box.mp:.4f}")
    print(f"[RESULT] Recall      : {metrics.box.mr:.4f}")
    return metrics


def export_onnx(model):
    """导出 ONNX（可选）"""
    print("\n[INFO] 导出 ONNX ...")
    path = model.export(format="onnx", simplify=True, dynamic=False, imgsz=640)
    print(f"[INFO] ONNX 已保存到: {path}")


def main():
    check_ultralytics_version()
    args = parse_args()
    cfg  = build_train_kwargs(args)

    data_yaml = cfg["data"]

    # 验证数据集路径
    if not Path(data_yaml).exists():
        print(f"[ERROR] 找不到数据集配置文件: {data_yaml}")
        print("        请先运行 extract_cz_dataset.py 生成 dataset_cz/")
        sys.exit(1)

    results, model = train(cfg)
    post_train_eval(model, data_yaml)

    print("\n[DONE] 所有文件保存在:", Path(args.project) / args.name)


if __name__ == "__main__":
    main()