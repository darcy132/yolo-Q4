以下是为您的 YOLO 道路损伤检测训练脚本编写的 README 文档：

---

# YOLO Road Damage Detection Training

基于 YOLOv8 的道路损伤检测模型训练脚本，针对道路裂缝、坑洼等损伤类型进行目标检测训练。

## 📋 目录

- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [安装配置](#安装配置)
- [数据集准备](#数据集准备)
- [使用方法](#使用方法)
- [训练参数说明](#训练参数说明)
- [训练策略](#训练策略)
- [输出结果](#输出结果)
- [常见问题](#常见问题)

## ✨ 功能特性

- 支持断点续训（自动检测 checkpoint 并恢复）
- 针对道路损伤检测优化的数据增强策略
- 类别不平衡处理（提高分类损失权重）
- 小样本类别专项增强（copy_paste、mixup）
- 自动保存最佳模型和定期检查点

## 💻 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+（GPU 训练）
- Ultralytics YOLOv8

### 依赖安装

```bash
pip install torch torchvision ultralytics
```

## 📦 安装配置

1. 克隆仓库（如适用）：
```bash
git clone <your-repo-url>
cd <your-project-directory>
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 确认 GPU 可用：
```python
import torch
print(torch.cuda.is_available())  # 应返回 True
print(torch.cuda.device_count())  # GPU 数量
```

## 📁 数据集准备

### 数据集结构

```
yolo_dataset/
├── dataset.yaml          # 数据集配置文件
├── images/
│   ├── train/           # 训练集图片
│   └── val/             # 验证集图片
└── labels/
    ├── train/           # 训练集标注
    └── val/             # 验证集标注
```

### dataset.yaml 配置示例

```yaml
# dataset.yaml
path: /home/forge/workspace/yolo-Q4/yolo_dataset  # 数据集根目录
train: images/train
val: images/val

names:
  0: crack
  1: pothole
  2: rutting
  3: hbgdf  # 小样本类别
  # 根据实际类别名称修改
```

## 🚀 使用方法

### 1. 从头开始训练

```bash
python train.py
```

### 2. 从 checkpoint 恢复训练

脚本会自动检测 `runs/detect/road_damage_v1/weights/last.pt` 是否存在，如存在则自动恢复训练。

### 3. 修改训练配置

编辑 `train.py` 中的以下变量：

```python
# 修改数据集路径
data='/your/path/to/dataset.yaml'

# 修改 checkpoint 路径（如需）
checkpoint = 'runs/detect/runs/detect/road_damage_v1/weights/last.pt'

# 修改模型大小（n/s/m/l/x）
model = YOLO('yolov8s.pt')  # 可选: yolov8n/s/m/l/x
```

### 4. 自定义训练参数

在 `model.train()` 中修改参数：

```python
results = model.train(
    epochs=200,        # 训练轮数
    batch=32,          # 批次大小（根据显存调整）
    imgsz=640,         # 输入图像尺寸
    device=0,          # GPU 设备 ID
    # ... 其他参数
)
```

## ⚙️ 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 200 | 训练轮数 |
| `imgsz` | 640 | 输入图像尺寸（道路损伤需要较高分辨率） |
| `batch` | 32 | 批次大小（根据 GPU 显存调整） |
| `device` | 0 | GPU 设备编号 |
| `optimizer` | AdamW | 优化器 |
| `lr0` | 1e-3 | 初始学习率 |
| `lrf` | 0.01 | 最终学习率因子 |
| `weight_decay` | 5e-4 | 权重衰减 |
| `cls` | 1.5 | 分类损失权重（提高以处理类别不平衡） |
| `mosaic` | 1.0 | Mosaic 数据增强概率 |
| `mixup` | 0.2 | Mixup 增强概率 |
| `copy_paste` | 0.3 | Copy-Paste 增强（对小样本有效） |

## 🎯 训练策略

### 针对道路损伤检测的优化

1. **高分辨率输入**（640x640）：捕捉细小裂缝特征
2. **类别不平衡处理**：
   - 提高分类损失权重（`cls=1.5`）
   - 针对性数据增强
3. **小样本类别增强**：
   - Copy-Paste（0.3）：复制粘贴小目标
   - Mixup（0.2）：图像混合
   - Mosaic（1.0）：四图拼接
4. **训练策略**：
   - 最后 20 个 epoch 关闭 Mosaic（`close_mosaic=20`）
   - 早停机制（`patience=30`）
   - 每 10 个 epoch 保存检查点

## 📊 输出结果

训练完成后，结果保存在：

```
runs/detect/road_damage_v1/
├── weights/
│   ├── best.pt      # 最佳模型（基于 mAP）
│   └── last.pt      # 最后一个 epoch 的模型
├── args.yaml        # 训练参数配置
├── results.csv      # 训练指标（可用于绘图）
├── results.png      # 训练曲线图
└── confusion_matrix.png  # 混淆矩阵
```

### 查看训练结果

脚本会在训练结束后打印最佳指标：

```python
Best mAP50: 0.XXX
Best mAP50-95: 0.XXX
```

## ❓ 常见问题

### 1. GPU 显存不足（OOM）

**解决方案**：
- 减小 `batch` 大小（如 16、8）
- 减小 `imgsz`（如 512）
- 使用梯度累积：添加 `accumulate=2`

### 2. WandB 连接失败（国内环境）

**解决方案**：
```python
# 方法一：离线模式
wandb.init(mode="offline")

# 方法二：禁用 wandb
model.train(project='runs/detect', name='road_damage_v1', exist_ok=True)
```

### 3. 数据集路径错误

**检查**：
- 确认 `dataset.yaml` 中的 `path` 为绝对路径
- 确认训练/验证图片和标签目录存在

### 4. Checkpoint 找不到

脚本会自动检测并提示：
- 如需从头训练，删除或移动旧的 checkpoint 文件
- 如需恢复训练，确保路径正确

## 📝 版本历史

- v1.0 - 初始版本，支持基础训练和断点续训

## 📄 许可证

[根据您的项目许可证填写]

## 👥 联系方式

[您的联系方式或项目链接]

---

**注意**：训练前请根据实际 GPU 显存调整 `batch` 参数，建议至少 8GB 显存进行 640x640 分辨率训练。