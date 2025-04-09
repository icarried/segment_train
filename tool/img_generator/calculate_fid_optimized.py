"""
优化版 FID 计算工具
================

针对内存受限系统优化的 FID 计算工具，通过批量处理和限制并行进程减少内存使用。
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from scipy import linalg
import gc  # 垃圾回收


class InceptionV3Features:
    def __init__(self, device='cpu'):
        # 使用CPU可以减少GPU内存使用
        self.device = device
        # 只加载到Mixed_7c层用于特征提取
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.dropout = torch.nn.Identity()  # 移除dropout
        self.model.fc = torch.nn.Identity()  # 移除全连接层
        self.model.eval()
        self.model.to(device)
        
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        if img.mode == 'L':  # 灰度图像转RGB
            img = img.convert('RGB')
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img)
        return features.squeeze().cpu().numpy()


def calculate_activation_statistics(files, model, batch_size=8):
    """计算图像集的激活统计数据"""
    act_list = []
    
    for i in tqdm(range(0, len(files), batch_size)):
        batch_files = files[i:i+batch_size]
        batch_act = []
        
        for img_path in batch_files:
            img = Image.open(img_path)
            features = model(img)
            batch_act.append(features)
            
        act_list.extend(batch_act)
        
        # 每个批次后清理内存
        if i % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    act = np.stack(act_list)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算FID"""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # 确保协方差矩阵是正定的
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # 处理复数结果
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    trace_term = np.trace(sigma1 + sigma2 - 2 * covmean)
    return diff.dot(diff) + trace_term


def calculate_fid(real_path, generated_path, batch_size=8, device='cpu'):
    """计算两个图像文件夹之间的FID分数"""
    print(f"计算FID分数...")
    print(f"真实图像路径: {real_path}")
    print(f"生成图像路径: {generated_path}")
    
    # 获取图像路径列表
    real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) 
                  if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    gen_files = [os.path.join(generated_path, f) for f in os.listdir(generated_path)
                 if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    print(f"找到 {len(real_files)} 个真实图像和 {len(gen_files)} 个生成图像")
    
    # 初始化模型
    model = InceptionV3Features(device)
    
    # 计算统计信息
    print("计算真实图像统计信息...")
    real_mu, real_sigma = calculate_activation_statistics(real_files, model, batch_size)
    
    print("计算生成图像统计信息...")
    gen_mu, gen_sigma = calculate_activation_statistics(gen_files, model, batch_size)
    
    # 释放内存
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 计算FID
    print("计算FID...")
    fid_value = calculate_frechet_distance(real_mu, real_sigma, gen_mu, gen_sigma)
    
    return fid_value


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='计算FID分数(优化版)')
    parser.add_argument('--real_path', type=str, required=True, help='真实图像路径')
    parser.add_argument('--generated_path', type=str, required=True, help='生成图像路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='计算设备')
    args = parser.parse_args()
    
    fid_value = calculate_fid(args.real_path, args.generated_path, args.batch_size, args.device)
    print(f"\nFID分数: {fid_value:.4f}")