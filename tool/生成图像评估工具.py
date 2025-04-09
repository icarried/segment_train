"""
生成图像评估工具
=============

此脚本用于评估生成的细胞图像质量，包括:
1. 计算与原始数据集的FID分数
2. 可视化对比生成图像与原始图像

使用方法:
python 生成图像评估工具.py --dataset ISBI2012 --generated_path ./xxxx/img/train
"""

import os
import argparse
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np

# 尝试导入优化版本，如果失败则导入原版
try:
    from img_generator.calculate_fid_optimized import calculate_fid
    print("使用内存优化版FID计算")
except ImportError:
    try:
        from img_generator.calculate_fid import calculate_fid
        print("使用标准FID计算")
    except ImportError:
        # 内联定义简单版本的FID计算函数
        def calculate_fid(real_path, generated_path, batch_size=8, device='cpu'):
            try:
                import torch
                from pytorch_fid import fid_score
                return fid_score.calculate_fid_given_paths(
                    [real_path, generated_path],
                    batch_size, device, dims=2048, num_workers=0  # 设置num_workers=0减少内存使用
                )
            except ImportError:
                print("警告: pytorch-fid 未安装，无法计算FID分数")
                print("请运行: pip install pytorch-fid")
                return float('nan')
            except Exception as e:
                print(f"计算FID时发生错误: {e}")
                return float('nan')


# 数据集路径配置
DATASET_PATHS = {
    'ISBI2012': '../dataset_ISBI2012/img/train',
    'CryoNuSeg': '../xxxx/img/train',
    # 添加其他数据集路径
}


def visualize_comparison(real_images_path, generated_images_path, num_samples=5):
    """
    可视化对比原始图像和生成图像
    
    参数:
        real_images_path: 原始图像路径
        generated_images_path: 生成图像路径
        num_samples: 显示的图像对数量
    """
    real_files = [f for f in os.listdir(real_images_path) if f.endswith(('.png', '.jpg', '.tif'))]
    gen_files = [f for f in os.listdir(generated_images_path) if f.endswith(('.png', '.jpg', '.tif'))]
    
    if len(real_files) < num_samples or len(gen_files) < num_samples:
        num_samples = min(len(real_files), len(gen_files))
        print(f"注意: 由于图像数量限制，仅显示 {num_samples} 个样本")
    
    real_samples = random.sample(real_files, num_samples)
    gen_samples = random.sample(gen_files, num_samples)
    
    plt.figure(figsize=(12, 8))
    
    for i in range(num_samples):
        # 原始图像
        real_img = Image.open(os.path.join(real_images_path, real_samples[i]))
        plt.subplot(2, num_samples, i+1)
        plt.imshow(np.array(real_img), cmap='gray' if real_img.mode == 'L' else None)
        plt.title(f"原始 {i+1}")
        plt.axis('off')
        
        # 生成图像
        gen_img = Image.open(os.path.join(generated_images_path, gen_samples[i]))
        plt.subplot(2, num_samples, i+num_samples+1)
        plt.imshow(np.array(gen_img), cmap='gray' if gen_img.mode == 'L' else None)
        plt.title(f"生成 {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('image_comparison.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='评估生成的细胞图像')
    parser.add_argument('--dataset', type=str, default='ISBI2012', choices=DATASET_PATHS.keys(), 
                        help='数据集名称') # ISBI2012
    parser.add_argument('--generated_path', type=str, default="./xxxx/img/train", 
                        help='生成图像路径')
    parser.add_argument('--batch_size', type=int, default=8, help='FID计算的批次大小')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                        help='计算设备(建议使用CPU以减少内存占用)')
    parser.add_argument('--visualize', action='store_true', default=True, help='可视化对比生成图像和原始图像')
    parser.add_argument('--num_visualize', type=int, default=5, help='可视化的图像对数量')
    args = parser.parse_args()
    
    real_path = DATASET_PATHS[args.dataset]
    generated_path = args.generated_path
    
    # 修正路径，确保是图像目录
    if os.path.isdir(os.path.join(generated_path, "img")) and not os.path.isdir(os.path.join(generated_path, "train")):
        generated_path = os.path.join(generated_path, "img", "train")
    elif os.path.isdir(os.path.join(generated_path, "train")):
        generated_path = os.path.join(generated_path, "train")
    
    print(f"开始评估生成图像: {generated_path}")
    print(f"参考原始数据集: {args.dataset} ({real_path})")
    
    # 计算FID分数
    fid_value = calculate_fid(real_path, generated_path, args.batch_size, args.device)
    
    # 可视化对比
    if args.visualize:
        print("\n生成图像与原始图像对比可视化...")
        visualize_comparison(real_path, generated_path, args.num_visualize)
    
    # 保存结果
    with open(f"evaluation_results_{args.dataset}.txt", "w") as f:
        f.write(f"数据集: {args.dataset}\n")
        f.write(f"生成图像路径: {generated_path}\n")
        f.write(f"FID分数: {fid_value:.4f}\n")
    
    print(f"\nFID分数: {fid_value:.4f}")
    print(f"\n评估结果已保存至 evaluation_results_{args.dataset}.txt")


if __name__ == "__main__":
    main()