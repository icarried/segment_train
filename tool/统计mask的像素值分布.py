import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

# Configure matplotlib to use a font that supports Chinese characters
try:
    # For Windows
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    # Fallback options
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
except:
    print("Could not set Chinese font, using default font instead")

def analyze_mask_distribution(mask_path):
    """
    分析单通道mask的像素分布，并以累进和非累进两种方式显示
    
    参数:
        mask_path: mask图像的路径
    """
    # 读取mask图像（以灰度模式读取确保单通道）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"无法读取图像: {mask_path}")
        return
    
    # 计算直方图（非累进）
    hist = cv2.calcHist([mask], [0], None, [256], [0, 256])
    
    # 计算累进直方图
    cumulative_hist = np.cumsum(hist)
    
    # 归一化以便更好地显示
    normalized_hist = hist / mask.size * 100  # 转换为百分比
    normalized_cumulative = cumulative_hist / mask.size * 100  # 转换为百分比
    
    # 创建图形
    plt.figure(figsize=(15, 6))
    
    # 绘制非累进直方图
    plt.subplot(121)
    plt.bar(range(256), normalized_hist.flatten(), width=1.0)
    plt.title('非累进像素分布')
    plt.xlabel('像素值 (0-255)')
    plt.ylabel('百分比 (%)')
    plt.xlim([0, 255])
    plt.grid(True, alpha=0.3)
    
    # 绘制累进直方图
    plt.subplot(122)
    plt.plot(range(256), normalized_cumulative.flatten(), 'r', linewidth=2)
    plt.title('累进像素分布')
    plt.xlabel('像素值 (0-255)')
    plt.ylabel('累计百分比 (%)')
    plt.xlim([0, 255])
    plt.ylim([0, 100])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印一些统计信息
    print(f"图像尺寸: {mask.shape}")
    print(f"像素值范围: [{np.min(mask)}, {np.max(mask)}]")
    print(f"平均像素值: {np.mean(mask):.2f}")
    
    # 查找主要的像素值（直方图中的峰值）
    peak_values = np.where(hist > np.max(hist) * 0.1)[0]
    print(f"主要像素值: {peak_values}")

# 使用示例
if __name__ == "__main__":
    # 替换为您的mask文件路径
    mask_path = "./xxxxx.png"
    analyze_mask_distribution(mask_path)