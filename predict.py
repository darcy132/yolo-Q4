import json
import os
from pathlib import Path
from ultralytics import YOLO

# 类别ID → 编码映射
ID2CODE = {
    0: 'lmlj', 1: 'hbgdf', 2: 'hxlf', 3: 'zxlf',
    4: 'jl',   5: 'kc',    6: 'ssf',  7: 'cz'
}

def predict(
    model_path: str,
    test_images_dir: str,
    output_json: str = 'result.json',
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
):
    model = YOLO(model_path)
    
    image_files = sorted(Path(test_images_dir).glob('*.jpg'))
    print(f"共找到 {len(image_files)} 张测试图片")
    
    results_dict = {}
    
    for img_path in image_files:
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=1280,
            verbose=False,
            augment=True,   # TTA（测试时增强），提升mAP
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls.item())
                score = float(box.conf.item())
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                
                detections.append({
                    "category": ID2CODE[cls_id],
                    "bbox": [round(xmin, 1), round(ymin, 1), 
                             round(xmax, 1), round(ymax, 1)],
                    "score": round(score, 4)
                })
        
        results_dict[img_path.name] = detections
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存至: {output_json}")
    total_det = sum(len(v) for v in results_dict.values())
    print(f"总检测数: {total_det}")

if __name__ == '__main__':
    predict(
        model_path='runs/detect/runs/detect/road_damage_v1/weights/last.pt', # 请根据实际路径调整
        test_images_dir='/home/forge/yolo-Q4/Q4-Dataset/test_set', # 请根据实际路径调整
        output_json='result.json',
        conf_threshold=0.25,
    )