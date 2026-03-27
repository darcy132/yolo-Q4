#!/usr/bin/env python3
"""
下载并解压 Q4 数据集
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

# 文件 URL 列表
URLS = [
    "https://nactrans.myvessel.cn/data/Q4-TestDataset.zip",
    "https://nactrans.myvessel.cn/data/Q4-TrainingDataset.zip"
]

# 下载目录（可选，默认当前目录）
DOWNLOAD_DIR = Path("downloads")


def download_file(url, dest_path):
    """下载文件并显示进度"""
    print(f"正在下载: {url}")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, int(downloaded * 100 / total_size))
            sys.stdout.write(f"\r  进度: {percent}% ({downloaded}/{total_size} bytes)")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, dest_path, report_progress)
        print()  # 换行
        print(f"  保存到: {dest_path}")
        return True
    except Exception as e:
        print(f"\n  下载失败: {e}")
        return False


def unzip_file(zip_path, extract_to):
    """解压 zip 文件"""
    print(f"正在解压: {zip_path.name}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"  解压到: {extract_to}")
        return True
    except Exception as e:
        print(f"  解压失败: {e}")
        return False


def main():
    # 创建下载目录
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    
    for url in URLS:
        # 从 URL 获取文件名
        filename = url.split('/')[-1]
        zip_path = DOWNLOAD_DIR / filename
        extract_dir = DOWNLOAD_DIR / filename.replace('.zip', '')
        
        # 检查文件是否已存在
        if zip_path.exists():
            print(f"文件已存在，跳过下载: {zip_path}")
        else:
            # 下载文件
            if not download_file(url, zip_path):
                continue
        
        # 检查解压目录是否已存在
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"解压目录已存在且非空，跳过解压: {extract_dir}")
        else:
            # 解压文件
            unzip_file(zip_path, DOWNLOAD_DIR)
        
        print()  # 空行分隔
    
    print("所有操作完成！")
'''
downloads/
├── Q4-TestDataset.zip
├── Q4-TestDataset/          # 解压后的测试数据
├── Q4-TrainingDataset.zip
└── Q4-TrainingDataset/      # 解压后的训练数据
'''
if __name__ == "__main__":
    main()