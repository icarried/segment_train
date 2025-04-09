"""
FID分数计算工具
=============

此脚本用于计算生成细胞图像与原始数据集之间的Fréchet Inception Distance (FID)。
FID是评估生成图像质量的标准指标，越低表示生成图像与真实图像越相似。

使用方法:
python calculate_fid.py --real_path /path/to/real/images --generated_path /path/to/generated/images
"""

import os
import argparse
import torch
from pytorch_fid import fid_score
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


def calculate_fid(real_path, generated_path, batch_size=50, device='cuda'):
    """
    计算两个图像文件夹之间的FID分数
    
    参数:
        real_path: 原始数据集的图像路径
        generated_path: 生成图像的路径
        batch_size: 批次大小
        device: 计算设备 ('cuda' 或 'cpu')
    
    返回:
        fid_value: FID分数
    """
    print(f"计算FID分数...")
    print(f"真实图像路径: {real_path}")
    print(f"生成图像路径: {generated_path}")
    
    fid_value = fid_score.calculate_fid_given_paths(
        [real_path, generated_path],
        batch_size,
        device,
        dims=2048,
        num_workers=8
    )
    
    return fid_value


def main():
    parser = argparse.ArgumentParser(description='计算生成细胞图像与原始数据集的FID分数')
    parser.add_argument('--real_path', type=str, required=True, help='原始数据集图像路径')
    parser.add_argument('--generated_path', type=str, required=True, help='生成图像路径')
    parser.add_argument('--batch_size', type=int, default=50, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='计算设备')
    args = parser.parse_args()
    
    fid_value = calculate_fid(args.real_path, args.generated_path, args.batch_size, args.device)
    
    print(f"\nFID分数: {fid_value:.4f}")
    print(f"FID分数越低表示生成图像与原始图像越相似")


if __name__ == "__main__":
    main()