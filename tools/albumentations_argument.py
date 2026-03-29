import albumentations as A
import cv2
import os
from tqdm import tqdm

# ==================== 离线增强专用 Transform ====================
transform = A.Compose([
    A.CLAHE(clip_limit=3.0, p=0.8),                    # 重点：增强对比度，对车辙很关键
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.GaussNoise(var_limit=(10, 50), p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.4),
    A.Rotate(limit=5, p=0.5),                          # 小角度旋转
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, p=0.6),
    A.RandomShadow(p=0.4),                             # 模拟路面阴影
    # 注意：这里**不要**加 Normalize 和 ToTensorV2
], bbox_params=A.BboxParams(
    format='yolo', 
    label_fields=['class_labels'],
    min_visibility=0.2,      # 只保留至少20%可见的框（防止变换后框跑出图像）
    min_area=0.0005          # 过滤极小框
))

# ====================== 配置区 ======================
IMAGE_DIR = "dataset_cz/images/train"          # 原图文件夹
LABEL_DIR = "dataset_cz/labels/train"          # 原标注文件夹（YOLO txt）
OUTPUT_IMAGE_DIR = "augmented_cz/images/train" # 输出增强图片文件夹
OUTPUT_LABEL_DIR = "augmented_cz/labels/train" # 输出增强标注文件夹

NUM_AUG_PER_IMAGE = 3                 # 每张原图生成几张增强图（可调）
TARGET_CLASSES = [0]               # ←←← 这里控制“指定类”！
                                      # 例如只增强包含 class 0（车辙）和 class 1 的图片
                                      # 如果想增强所有图片，就写 [] 或 None

# 创建输出文件夹
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# ====================== 核心处理函数 ======================
def augment_image(image_path: str, label_path: str, aug_idx: int):
    # 1. 读取图片（BGR）
    image = cv2.imread(image_path)
    if image is None:
        return False

    # 2. 读取 YOLO 标注
    bboxes = []
    class_labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cls, x, y, w, h = map(float, line.strip().split())
                bboxes.append([x, y, w, h])
                class_labels.append(int(cls))

    # 3. 【控制指定类】—— 如果设置了 TARGET_CLASSES，则只处理包含指定类的图片
    if TARGET_CLASSES and not any(c in TARGET_CLASSES for c in class_labels):
        return False   # 不包含目标类，跳过

    # 4. 应用增强（每次调用都会产生不同随机效果）
    transformed = transform(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels
    )

    aug_image = transformed['image']
    aug_bboxes = transformed['bboxes']
    aug_labels = transformed['class_labels']

    # 5. 保存增强后的图片和标注
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    new_image_name = f"{base_name}_aug{aug_idx}.jpg"
    new_label_name = f"{base_name}_aug{aug_idx}.txt"

    cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, new_image_name), aug_image)

    with open(os.path.join(OUTPUT_LABEL_DIR, new_label_name), 'w', encoding='utf-8') as f:
        for bbox, cls in zip(aug_bboxes, aug_labels):
            f.write(f"{int(cls)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    return True


# ====================== 批量执行 ======================
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"开始离线增强，共 {len(image_files)} 张原图，每图生成 {NUM_AUG_PER_IMAGE} 张增强图...")
success_count = 0

for img_file in tqdm(image_files):
    image_path = os.path.join(IMAGE_DIR, img_file)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(img_file)[0] + ".txt")
    
    if not os.path.exists(label_path):
        continue

    for i in range(NUM_AUG_PER_IMAGE):
        if augment_image(image_path, label_path, i):
            success_count += 1

print(f"✅ 增强完成！共生成 {success_count} 张新图片 + 标注")