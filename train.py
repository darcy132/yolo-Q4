from ultralytics import YOLO
import torch
import os

def train():
    checkpoint = 'runs/detect/runs/detect/road_damage_v2/weights/last.pt' # 请根据实际路径调整
    
    # 自动判断是新训练还是恢复训练
    if os.path.exists(checkpoint):
        print(f"[Resume] 发现 checkpoint，从 {checkpoint} 恢复训练")
        model = YOLO(checkpoint)
        results = model.train(resume=True)
    else:
        print("[New] 未发现 checkpoint，从头开始训练")
        model = YOLO('yolov8s.pt')  # 可选: yolov8n/s/m/l/x，精度要求高用 l 或 x
        results = model.train(
            data='/home/forge/workspace/yolo-Q4/yolo_dataset/dataset.yaml', # 请根据实际路径调整
            epochs=200,
            imgsz=640,         # 道路裂缝细节多，建议高分辨率
            batch=32,            # 根据显存调整
            device=0,           # GPU
            
            # 优化器
            optimizer='AdamW',
            lr0=1e-3,
            lrf=0.01,
            weight_decay=5e-4,
            warmup_epochs=5,
            
            # 针对类别不平衡的关键设置
            cls=1.5,            # 提高分类损失权重
            
            # 数据增强（应对小样本类别）
            mosaic=1.0,
            mixup=0.2,
            copy_paste=0.3,     # 对小样本类别（hbgdf:47个）非常有效
            degrees=10.0,
            flipud=0.3,
            fliplr=0.5,
            scale=0.7,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            
            # 训练策略
            close_mosaic=20,    # 最后20个epoch关闭mosaic，稳定收敛
            patience=30,        # 早停
            
            # 保存
            project='runs/detect',
            name='road_damage_v2',
            save_period=10,
        )
    
    if results:
        print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
            
    return results

if __name__ == '__main__':
    train()