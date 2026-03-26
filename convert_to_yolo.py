import os
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import shutil
from collections import defaultdict

# ==================== 类别定义 ====================
# 根据你提供的类别表定义
CLASSES = {
    'lmlj': 0,    # 路面垃圾 Roadside Litter
    'hbgdf': 1,   # 红白杆倒伏 Fallen Red-White Pole
    'hxlf': 2,    # 横向裂缝 Transverse Crack
    'zxlf': 3,    # 纵向裂缝 Longitudinal Crack
    'jl': 4,      # 龟裂 Alligator Crack
    'kc': 5,      # 坑槽 Pothole
    'ssf': 6,     # 伸缩缝破损 Expansion Joint Damage
    'cz': 7,      # 路面车辙 Rutting
}

# 创建反向映射，用于显示
CLASS_NAMES = {v: k for k, v in CLASSES.items()}
# 创建中文名称映射
CLASS_NAMES_CN = {
    'lmlj': '路面垃圾',
    'hbgdf': '红白杆倒伏',
    'hxlf': '横向裂缝',
    'zxlf': '纵向裂缝',
    'jl': '龟裂',
    'kc': '坑槽',
    'ssf': '伸缩缝破损',
    'cz': '路面车辙',
}

def get_class_id(class_name):
    """获取类别ID，如果类别不在预定义中则返回None"""
    return CLASSES.get(class_name, None)

def extract_classes_from_xml(xml_dir):
    """从所有XML文件中提取所有唯一的类别名称（用于验证）"""
    classes_found = set()
    xml_dir = Path(xml_dir)
    
    for xml_file in xml_dir.glob('*.xml'):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                classes_found.add(class_name)
        except Exception as e:
            print(f"读取文件 {xml_file} 时出错: {e}")
    
    return sorted(list(classes_found))

def count_classes_in_dataset(xml_dir):
    """统计每个类别在数据集中的出现次数"""
    class_count = defaultdict(int)
    file_count = 0
    xml_dir = Path(xml_dir)
    
    print("\n" + "="*60)
    print("开始统计类别分布...")
    print("="*60)
    
    for xml_file in xml_dir.glob('*.xml'):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            file_has_object = False
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_count[class_name] += 1
                file_has_object = True
            
            if file_has_object:
                file_count += 1
                
        except Exception as e:
            print(f"统计文件 {xml_file} 时出错: {e}")
    
    return class_count, file_count

def print_class_statistics(class_count, file_count):
    """打印类别统计信息"""
    print(f"\n总文件数（包含标注）: {file_count}")
    print(f"总标注数: {sum(class_count.values())}")
    print("\n各类别统计详情:")
    print("-" * 50)
    print(f"{'类别编码':<10} {'中文名称':<12} {'英文名称':<20} {'数量':<8} {'占比':<8}")
    print("-" * 50)
    
    total_annotations = sum(class_count.values())
    
    for class_code, count in sorted(class_count.items(), key=lambda x: x[1], reverse=True):
        cn_name = CLASS_NAMES_CN.get(class_code, '未知')
        en_name = CLASSES.get(class_code, '未知')
        percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
        
        # 处理英文名称显示
        if isinstance(en_name, int):
            en_name = '未定义'
        
        print(f"{class_code:<10} {cn_name:<12} {en_name:<20} {count:<8} {percentage:.1f}%")
    
    print("-" * 50)
    
    # 检查是否有未定义的类别
    unknown_classes = [cls for cls in class_count.keys() if cls not in CLASSES]
    if unknown_classes:
        print(f"\n⚠️ 警告: 发现未定义的类别: {unknown_classes}")
        print("请在CLASSES字典中添加这些类别的定义")
    
    return total_annotations

def convert_voc_to_yolo(xml_file, output_dir, stats=None):
    """将单个XML文件转换为YOLO TXT格式"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 获取图片尺寸
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # 生成对应的txt文件名
        txt_filename = Path(xml_file).stem + '.txt'
        txt_path = Path(output_dir) / txt_filename
        
        # 存储所有标注
        annotations = []
        
        for obj in root.findall('object'):
            # 获取类别
            class_name = obj.find('name').text
            class_id = get_class_id(class_name)
            
            if class_id is None:
                print(f"警告: 未知类别 '{class_name}' 在文件 {xml_file}，已跳过")
                continue
            
            # 获取边界框坐标
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # 确保坐标在图片范围内
            xmin = max(0, min(xmin, img_width))
            xmax = max(0, min(xmax, img_width))
            ymin = max(0, min(ymin, img_height))
            ymax = max(0, min(ymax, img_height))
            
            # 计算边界框宽度和高度
            box_width = xmax - xmin
            box_height = ymax - ymin
            
            # 转换为YOLO格式（归一化）
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = box_width / img_width
            height = box_height / img_height
            
            # 确保数值在0-1范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # 更新统计信息
            if stats is not None:
                stats['converted'][class_name] += 1
        
        # 写入文件
        if annotations:
            with open(txt_path, 'w') as f:
                f.write('\n'.join(annotations))
            return True, len(annotations)
        else:
            # 如果没有有效标注，创建空文件
            with open(txt_path, 'w') as f:
                pass
            return False, 0
            
    except Exception as e:
        print(f"转换文件 {xml_file} 时出错: {e}")
        return False, 0

def organize_dataset(xml_dir, images_dir, output_base_dir, train_ratio=0.8, val_ratio=0.2):
    """组织数据集为YOLO格式"""
    
    # 创建目录结构
    train_img_dir = Path(output_base_dir) / 'images' / 'train'
    val_img_dir = Path(output_base_dir) / 'images' / 'val'
    train_label_dir = Path(output_base_dir) / 'labels' / 'train'
    val_label_dir = Path(output_base_dir) / 'labels' / 'val'
    
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有XML文件
    xml_files = list(Path(xml_dir).glob('*.xml'))
    
    if not xml_files:
        print(f"在 {xml_dir} 中没有找到XML文件")
        return None
    
    # 首先统计原始数据集的类别分布
    print("\n📊 原始数据集统计:")
    class_count, file_count = count_classes_in_dataset(xml_dir)
    total_annotations = print_class_statistics(class_count, file_count)
    
    # 提取所有类别（用于验证）
    classes_found = extract_classes_from_xml(xml_dir)
    print(f"\n实际找到的类别: {classes_found}")
    
    # 检查是否有未定义的类别
    unknown_classes = [cls for cls in classes_found if cls not in CLASSES]
    if unknown_classes:
        print(f"\n❌ 错误: 发现未在CLASSES中定义的类别: {unknown_classes}")
        print("请在CLASSES字典中添加这些类别的定义后再运行")
        return None
    
    # 创建类别映射文件
    with open(Path(output_base_dir) / 'classes.txt', 'w', encoding='utf-8') as f:
        for class_code, class_id in sorted(CLASSES.items(), key=lambda x: x[1]):
            cn_name = CLASS_NAMES_CN.get(class_code, '')
            f.write(f"{class_code} {class_id} {cn_name}\n")
    
    # 转换所有XML文件到YOLO格式
    print("\n🔄 正在转换XML文件到YOLO格式...")
    temp_label_dir = Path(output_base_dir) / 'temp_labels'
    temp_label_dir.mkdir(exist_ok=True)
    
    # 统计转换后的数据
    stats = {
        'converted': defaultdict(int),
        'files_with_annotations': 0,
        'total_annotations': 0
    }
    
    converted_count = 0
    for xml_file in xml_files:
        success, num_annots = convert_voc_to_yolo(xml_file, temp_label_dir, stats)
        if success:
            converted_count += 1
            if num_annots > 0:
                stats['files_with_annotations'] += 1
                stats['total_annotations'] += num_annots
    
    print(f"成功转换 {converted_count}/{len(xml_files)} 个文件")
    
    # 创建图片和标注的对应关系
    valid_pairs = []
    for xml_file in xml_files:
        base_name = xml_file.stem
        label_file = temp_label_dir / f"{base_name}.txt"
        
        # 查找对应的图片文件
        img_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            potential_img = Path(images_dir) / f"{base_name}{ext}"
            if potential_img.exists():
                img_file = potential_img
                break
        
        if img_file and label_file.exists():
            valid_pairs.append((img_file, label_file))
        else:
            if not img_file:
                print(f"警告: 找不到图片文件 {base_name}.*")
            if not label_file.exists():
                print(f"警告: 找不到标注文件 {base_name}.txt")
    
    print(f"找到 {len(valid_pairs)} 对有效的图片-标注对")
    
    # 随机划分数据集
    random.shuffle(valid_pairs)
    train_count = int(len(valid_pairs) * train_ratio)
    
    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:]
    
    # 复制文件到最终目录
    print("📁 正在复制文件到最终目录...")
    
    for img_file, label_file in train_pairs:
        shutil.copy2(img_file, train_img_dir / img_file.name)
        shutil.copy2(label_file, train_label_dir / label_file.name)
    
    for img_file, label_file in val_pairs:
        shutil.copy2(img_file, val_img_dir / img_file.name)
        shutil.copy2(label_file, val_label_dir / label_file.name)
    
    # 清理临时文件
    shutil.rmtree(temp_label_dir)
    
    # 打印转换后的统计信息
    print("\n📊 转换后数据统计:")
    print(f"训练集: {len(train_pairs)} 张图片")
    print(f"验证集: {len(val_pairs)} 张图片")
    print(f"总图片数: {len(valid_pairs)} 张")
    print(f"包含标注的图片数: {stats['files_with_annotations']} 张")
    print(f"总标注数: {stats['total_annotations']} 个")
    
    print("\n📈 转换后各类别统计:")
    print("-" * 50)
    print(f"{'类别编码':<10} {'中文名称':<12} {'数量':<8} {'占比':<8}")
    print("-" * 50)
    
    for class_code, count in sorted(stats['converted'].items(), key=lambda x: x[1], reverse=True):
        cn_name = CLASS_NAMES_CN.get(class_code, '未知')
        percentage = (count / stats['total_annotations'] * 100) if stats['total_annotations'] > 0 else 0
        print(f"{class_code:<10} {cn_name:<12} {count:<8} {percentage:.1f}%")
    
    print("-" * 50)

     # 打开文件准备写入统计信息
        # 打印转换后的统计信息
    print("\n📊 转换后数据统计:")
    print(f"训练集: {len(train_pairs)} 张图片")
    print(f"验证集: {len(val_pairs)} 张图片")
    print(f"总图片数: {len(valid_pairs)} 张")
    print(f"包含标注的图片数: {stats['files_with_annotations']} 张")
    print(f"总标注数: {stats['total_annotations']} 个")
    
    print("\n📈 转换后各类别统计:")
    print("-" * 50)
    print(f"{'ID':<4} {'类别编码':<10} {'中文名称':<12} {'数量':<8} {'占比':<8}")
    print("-" * 50)
    
    # 打开文件准备写入统计信息
    stats_file = Path(output_base_dir) / 'class_statistics.txt'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("类别统计信息\n")
        f.write("="*60 + "\n")
        f.write(f"{'ID':<4} {'类别编码':<10} {'中文名称':<12} {'数量':<8}\n")
        f.write("-"*60 + "\n")
        
        # 按ID排序（CLASSES中的原始ID）
        sorted_classes = sorted(CLASSES.items(), key=lambda x: x[1])
        
        for class_code, class_id in sorted_classes:
            count = stats['converted'].get(class_code, 0)
            cn_name = CLASS_NAMES_CN.get(class_code, '未知')
            # ID+1显示（1-8）
            display_id = class_id + 1
            percentage = (count / stats['total_annotations'] * 100) if stats['total_annotations'] > 0 else 0
            
            print(f"{display_id:<4} {class_code:<10} {cn_name:<12} {count:<8} {percentage:.1f}%")
            
            # 写入文件
            f.write(f"{display_id:<4} {class_code:<10} {cn_name:<12} {count:<8}\n")
        
        f.write("-"*60 + "\n")
        f.write(f"总计: {stats['total_annotations']} 个标注\n")
    
    print("-" * 50)
    print(f"\n✅ 统计信息已保存到: {stats_file}")
    
    return CLASSES  # 返回预定义的类别字典

def create_yaml_config(output_base_dir, classes_dict):
    """创建YOLO训练配置文件"""
    # 创建类别名称列表（按ID排序）
    class_names = []
    for class_id in range(len(classes_dict)):
        for code, cid in classes_dict.items():
            if cid == class_id:
                class_names.append(code)
                break
    
    yaml_content = f"""# YOLO数据集配置文件
# 道路病害检测数据集
# 生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

path: {Path(output_base_dir).absolute()}  # 数据集根目录
train: images/train  # 训练集图片路径
val: images/val      # 验证集图片路径

# 类别配置
nc: {len(classes_dict)}  # 类别数量
names: {class_names}  # 类别名称列表

# 类别详细说明
class_info:
"""
    
    for class_code, class_id in sorted(classes_dict.items(), key=lambda x: x[1]):
        cn_name = CLASS_NAMES_CN.get(class_code, '')
        yaml_content += f"  {class_id}: {{code: {class_code}, name_cn: {cn_name}}}\n"
    
    yaml_path = Path(output_base_dir) / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n✅ 配置文件已保存到: {yaml_path}")
    return yaml_path

def verify_conversion(dataset_dir):
    """验证转换结果"""
    print("\n" + "="*60)
    print("🔍 验证转换结果:")
    print("="*60)
    
    # 检查目录结构
    train_img_dir = Path(dataset_dir) / 'images/train'
    train_label_dir = Path(dataset_dir) / 'labels/train'
    
    if train_img_dir.exists() and train_label_dir.exists():
        train_images = list(train_img_dir.glob('*'))
        train_labels = list(train_label_dir.glob('*.txt'))
        
        print(f"训练集图片数量: {len(train_images)}")
        print(f"训练集标注数量: {len(train_labels)}")
        
        # 显示一个样本
        if train_labels:
            sample_label = train_labels[0]
            print(f"\n📄 样本标注文件: {sample_label.name}")
            with open(sample_label, 'r') as f:
                content = f.read().strip()
                if content:
                    lines = content.split('\n')
                    print(f"包含 {len(lines)} 个目标:")
                    for i, line in enumerate(lines[:5]):  # 只显示前5个
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            xc, yc, w, h = parts[1:]
                            # 查找类别名称
                            class_code = None
                            for code, cid in CLASSES.items():
                                if cid == class_id:
                                    class_code = code
                                    break
                            class_name = CLASS_NAMES_CN.get(class_code, f'未知({class_id})')
                            print(f"  目标{i+1}: {class_name}({class_code}) 中心=({xc}, {yc}) 尺寸=({w}, {h})")
                else:
                    print("文件为空")
    else:
        print("❌ 目录结构不完整")

def main():
    # ==================== 配置参数 ====================
    # 请根据实际情况修改这些路径
    xml_dir = "Q4-Dataset/train_set/annotation"      # XML文件所在目录
    images_dir = "Q4-Dataset/train_set/images"    # 图片文件所在目录
    output_dir = "yolo_dataset"          # 输出目录
    
    # 数据集划分比例
    train_ratio = 0.8
    val_ratio = 0.2
    
    print("🚀 开始转换道路病害数据集...")
    print(f"XML目录: {xml_dir}")
    print(f"图片目录: {images_dir}")
    print(f"输出目录: {output_dir}")
    print(f"类别总数: {len(CLASSES)}")
    
    # 显示预定义的类别
    print("\n📋 预定义类别列表:")
    for class_code, class_id in sorted(CLASSES.items(), key=lambda x: x[1]):
        cn_name = CLASS_NAMES_CN.get(class_code, '')
        print(f"  {class_id}: {class_code} ({cn_name})")
    
    # 组织数据集
    classes_dict = organize_dataset(xml_dir, images_dir, output_dir, train_ratio, val_ratio)
    
    if classes_dict:
        # 创建配置文件
        create_yaml_config(output_dir, classes_dict)
        
        # 验证转换结果
        verify_conversion(output_dir)
        
        print("\n" + "="*60)
        print("✅ 转换完成！")
        print(f"📁 数据集已保存到: {output_dir}")
        print(f"⚙️  YOLO配置文件: {output_dir}/dataset.yaml")
        print(f"📊 类别映射文件: {output_dir}/classes.txt")
        print("="*60)
        
        # 显示训练命令示例
        print("\n💡 训练命令示例:")
        print(f"python train.py --data {output_dir}/dataset.yaml --weights yolov5s.pt --epochs 100")
    else:
        print("❌ 转换失败，请检查输入路径和文件格式")

if __name__ == "__main__":
    main()