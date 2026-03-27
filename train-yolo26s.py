import os
from ultralytics import YOLO

def train():
    checkpoint = 'runs/detect/runs/detect/road_damage_v1_yolo26s3/weights/last.pt'
    
    if os.path.exists(checkpoint):
        print(f"[Resume] 发现 checkpoint，从 {checkpoint} 恢复训练")
        model = YOLO(checkpoint)
        results = model.train(resume=True)

    else:
        print("[New] 未发现 checkpoint，从头开始训练")
        model = YOLO('yolo26s.pt')  # 或 yolov8s.pt
        
        results = model.train(
            data='/home/forge/workspace/yolo-Q4/yolo_dataset/dataset.yaml',
            epochs=200,
            imgsz=640,
            batch=16,
            device=0,
            
            # 优化器
            optimizer='AdamW',
            lr0=1e-3,
            lrf=0.01,
            weight_decay=5e-4,
            warmup_epochs=5,
            
            # 针对类别不平衡
            cls=1.5,
            
            # 数据增强（调小参数版本）
            mosaic=0.8,           # 降低mosaic使用率
            mixup=0.1,            # 降低mixup概率
            copy_paste=0.2,       # 降低copy_paste概率
            degrees=5.0,          # 减小旋转角度
            flipud=0.2,           # 降低垂直翻转
            fliplr=0.5,           # 水平翻转保持不变
            scale=0.5,            # 降低缩放范围
            hsv_h=0.01,           # 降低色调变化
            hsv_s=0.5,            # 降低饱和度变化
            hsv_v=0.3,            # 降低明度变化
            
            # 训练策略
            close_mosaic=20,
            patience=30,
            
            # 保存
            project='runs/detect',
            name='road_damage_v1_yolo26s',
            save_period=10,
        )
    
    if results:
        print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
    return results

if __name__ == '__main__':
    train()
