import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.font_manager as fm
import os
import glob
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy import stats
import os

# Create directory for saving plots
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

# Configure matplotlib to use a font that supports Chinese characters
try:
    # For Windows
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    # Fallback options
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
except:
    print("Could not set Chinese font, using default font instead")

def analyze_mask_distribution(mask_path, show_plot=True):
    """
    分析单通道mask的像素分布，并以累进和非累进两种方式显示
    
    参数:
        mask_path: mask图像的路径
        show_plot: 是否显示图表
    
    返回:
        包含统计信息的字典
    """
    # 读取mask图像（以灰度模式读取确保单通道）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"无法读取图像: {mask_path}")
        return None
    
    # 计算直方图（非累进）
    hist = cv2.calcHist([mask], [0], None, [256], [0, 256])
    
    # 计算累进直方图
    cumulative_hist = np.cumsum(hist)
    
    # 归一化以便更好地显示
    normalized_hist = hist / mask.size * 100  # 转换为百分比
    normalized_cumulative = cumulative_hist / mask.size * 100  # 转换为百分比
    
    if show_plot:
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
    
    # 返回统计信息
    stats = {
        'filename': os.path.basename(mask_path),
        'shape': mask.shape,
        'min': np.min(mask),
        'max': np.max(mask),
        'mean': np.mean(mask),
        'std': np.std(mask),
        'median': np.median(mask),
        'histogram': hist.flatten(),
        'normalized_histogram': normalized_hist.flatten(),
        'nonzero_percentage': np.count_nonzero(mask) / mask.size * 100,
        'zero_percentage': 100 - (np.count_nonzero(mask) / mask.size * 100),
        'peak_values': np.where(hist > np.max(hist) * 0.1)[0]
    }
    
    return stats

def analyze_folder_masks(folder_path, extensions=('.png', '.jpg', '.tif', '.bmp')):
    """
    分析文件夹中所有mask图像的像素分布并聚合统计信息
    
    参数:
        folder_path: 包含mask图像的文件夹路径
        extensions: 要处理的文件扩展名元组
    
    返回:
        包含所有图像统计信息的列表和聚合统计信息
    """
    all_stats = []
    combined_histogram = np.zeros(256)
    total_pixels = 0
    all_mins = []
    all_maxs = []
    all_means = []
    all_medians = []
    all_stds = []
    all_nonzero_percentages = []
    all_zero_percentages = []
    all_peak_values = set()
    all_histograms = []  # 存储所有图像的直方图用于3D展示
    
    # 获取所有符合扩展名的文件
    mask_files = []
    for ext in extensions:
        mask_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
    
    if not mask_files:
        print(f"在文件夹 {folder_path} 中未找到mask图像文件")
        return None, None
    
    print(f"在文件夹 {folder_path} 中找到 {len(mask_files)} 个mask文件")
    
    # 处理每个mask文件
    for mask_file in tqdm(mask_files, desc="处理图像"):
        stats = analyze_mask_distribution(mask_file, show_plot=False)
        
        if stats is None:
            continue
            
        all_stats.append(stats)
        
        # 更新聚合统计数据
        image_pixels = stats['shape'][0] * stats['shape'][1]
        total_pixels += image_pixels
        combined_histogram += stats['histogram']
        
        all_mins.append(stats['min'])
        all_maxs.append(stats['max'])
        all_means.append(stats['mean'])
        all_medians.append(stats['median'])
        all_stds.append(stats['std'])
        all_nonzero_percentages.append(stats['nonzero_percentage'])
        all_zero_percentages.append(stats['zero_percentage'])
        all_peak_values.update(stats['peak_values'])
        all_histograms.append(stats['normalized_histogram'])  # 添加归一化的直方图
    
    if not all_stats:
        print("未能成功处理任何图像")
        return None, None
    
    # 计算聚合统计信息
    normalized_combined_histogram = combined_histogram / total_pixels * 100
    cumulative_combined_histogram = np.cumsum(combined_histogram)
    normalized_cumulative_combined = cumulative_combined_histogram / total_pixels * 100
    
    aggregate_stats = {
        'total_images': len(all_stats),
        'total_pixels': total_pixels,
        'min_values': {
            'min': np.min(all_mins),
            'max': np.max(all_mins),
            'mean': np.mean(all_mins),
            'std': np.std(all_mins)
        },
        'max_values': {
            'min': np.min(all_maxs),
            'max': np.max(all_maxs),
            'mean': np.mean(all_maxs),
            'std': np.std(all_maxs)
        },
        'mean_values': {
            'min': np.min(all_means),
            'max': np.max(all_means),
            'mean': np.mean(all_means),
            'std': np.std(all_means)
        },
        'median_values': {
            'min': np.min(all_medians),
            'max': np.max(all_medians),
            'mean': np.mean(all_medians),
            'std': np.std(all_medians)
        },
        'std_values': {
            'min': np.min(all_stds),
            'max': np.max(all_stds),
            'mean': np.mean(all_stds),
            'std': np.std(all_stds)
        },
        'nonzero_percentages': {
            'min': np.min(all_nonzero_percentages),
            'max': np.max(all_nonzero_percentages),
            'mean': np.mean(all_nonzero_percentages),
            'std': np.std(all_nonzero_percentages)
        },
        'zero_percentages': {
            'min': np.min(all_zero_percentages),
            'max': np.max(all_zero_percentages),
            'mean': np.mean(all_zero_percentages),
            'std': np.std(all_zero_percentages)
        },
        'combined_histogram': combined_histogram,
        'normalized_combined_histogram': normalized_combined_histogram,
        'normalized_cumulative_combined': normalized_cumulative_combined,
        'common_peak_values': sorted(list(all_peak_values)),
        'all_histograms': all_histograms,  # 添加所有图像的直方图
        'filenames': [stats['filename'] for stats in all_stats]  # 添加文件名列表
    }
    
    return all_stats, aggregate_stats

def fit_bimodal_gaussian(histogram, pixel_values=np.arange(256)):
    """
    对直方图数据拟合双峰高斯分布
    
    参数:
        histogram: 像素值直方图数据
        pixel_values: 对应的像素值，默认为0-255
        
    返回:
        拟合参数和拟合后的曲线数据
    """
    # 定义双高斯模型函数
    def bimodal_gaussian(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
        gaussian1 = amp1 * np.exp(-(x - mean1)**2 / (2 * sigma1**2))
        gaussian2 = amp2 * np.exp(-(x - mean2)**2 / (2 * sigma2**2))
        return gaussian1 + gaussian2
    
    # 估计初始参数
    # 查找最大的两个峰值
    hist_smooth = np.convolve(histogram, np.ones(5)/5, mode='same')  # 平滑处理
    peaks = []
    for i in range(1, len(hist_smooth)-1):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1] and hist_smooth[i] > np.max(hist_smooth) * 0.1:
            peaks.append(i)
    
    # 如果找到少于2个峰值，使用默认值
    if len(peaks) < 2:
        print("警告：未能找到两个明显的峰值，使用默认峰值位置")
        mean1, mean2 = 20, 240  # 默认峰值位置
    else:
        # 取最大的两个峰值
        peaks.sort(key=lambda i: hist_smooth[i], reverse=True)
        mean1, mean2 = peaks[0], peaks[1]
        if mean1 > mean2:
            mean1, mean2 = mean2, mean1  # 确保mean1 < mean2
    
    # 初始参数估计：[amp1, mean1, sigma1, amp2, mean2, sigma2]
    p0 = [histogram[mean1], mean1, 10, histogram[mean2], mean2, 10]
    
    try:
        # 拟合双高斯模型
        params, _ = curve_fit(bimodal_gaussian, pixel_values, histogram, p0=p0, maxfev=10000)
        
        # 生成拟合曲线
        fitted_curve = bimodal_gaussian(pixel_values, *params)
        
        # 拆分为两个单独的高斯分量
        gaussian1 = params[0] * np.exp(-(pixel_values - params[1])**2 / (2 * params[2]**2))
        gaussian2 = params[3] * np.exp(-(pixel_values - params[4])**2 / (2 * params[5]**2))
        
        return {
            'params': params,
            'fitted_curve': fitted_curve,
            'gaussian1': gaussian1,
            'gaussian2': gaussian2,
            'mean1': params[1],
            'std1': abs(params[2]),
            'mean2': params[4],
            'std2': abs(params[5])
        }
    except Exception as e:
        print(f"高斯拟合失败: {e}")
        return None

def visualize_aggregate_stats(aggregate_stats):
    """
    可视化聚合统计信息，并将每个子图保存到mask_像素分布文件夹
    
    参数:
        aggregate_stats: 由 analyze_folder_masks 函数返回的聚合统计信息
    """
    # 创建输出目录
    output_dir = "mask_像素分布"
    ensure_directory_exists(output_dir)
    
    # 拟合双高斯分布
    bimodal_fit = fit_bimodal_gaussian(aggregate_stats['normalized_combined_histogram'])
    
    # 创建并保存每个子图为单独的图像，但不显示
    
    # 1. 聚合非累进直方图及双高斯拟合
    plt.figure(figsize=(10, 8))
    plt.bar(range(256), aggregate_stats['normalized_combined_histogram'], width=1.0, alpha=0.6, label='像素分布')
    
    if bimodal_fit:
        plt.plot(range(256), bimodal_fit['fitted_curve'], 'r-', linewidth=2, label='双高斯拟合')
        plt.plot(range(256), bimodal_fit['gaussian1'], 'g--', linewidth=1.5, 
                 label=f'高斯1 (μ={bimodal_fit["mean1"]:.1f}, σ={bimodal_fit["std1"]:.1f})')
        plt.plot(range(256), bimodal_fit['gaussian2'], 'b--', linewidth=1.5, 
                 label=f'高斯2 (μ={bimodal_fit["mean2"]:.1f}, σ={bimodal_fit["std2"]:.1f})')
        plt.legend(loc='upper right', fontsize=10)
    
    plt.title(f'所有图像聚合像素分布 ({aggregate_stats["total_images"]} 个图像)')
    plt.xlabel('像素值 (0-255)')
    plt.ylabel('百分比 (%)')
    plt.xlim([0, 255])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '聚合直方图.png'), dpi=300)
    plt.close()  # 关闭图形而不显示
    
    # 2. 聚合累进直方图
    plt.figure(figsize=(10, 8))
    plt.plot(range(256), aggregate_stats['normalized_cumulative_combined'], 'r', linewidth=2)
    plt.title('所有图像聚合累进像素分布')
    plt.xlabel('像素值 (0-255)')
    plt.ylabel('累计百分比 (%)')
    plt.xlim([0, 255])
    plt.ylim([0, 100])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '累进直方图.png'), dpi=300)
    plt.close()  # 关闭图形而不显示
    
    # 3. 显示常见峰值
    plt.figure(figsize=(10, 8))
    peak_values = aggregate_stats['common_peak_values']
    plt.bar(peak_values, [aggregate_stats['normalized_combined_histogram'][p] for p in peak_values])
    plt.title('常见峰值像素值')
    plt.xlabel('像素值')
    plt.ylabel('百分比 (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '常见峰值.png'), dpi=300)
    plt.close()  # 关闭图形而不显示
    
    # 4. 绘制双高斯分布及其混合效果的比较图
    plt.figure(figsize=(10, 8))
    if bimodal_fit:
        x = np.linspace(0, 255, 500)
        params = bimodal_fit['params']
        
        # 两个高斯分布
        g1 = params[0] * np.exp(-(x - params[1])**2 / (2 * params[2]**2))
        g2 = params[3] * np.exp(-(x - params[4])**2 / (2 * params[5]**2))
        combined = g1 + g2
        
        # 绘制双高斯叠加效果
        plt.plot(x, g1, 'g-', linewidth=1.5, alpha=0.7, label=f'高斯1: μ={params[1]:.1f}')
        plt.plot(x, g2, 'b-', linewidth=1.5, alpha=0.7, label=f'高斯2: μ={params[4]:.1f}')
        plt.plot(x, combined, 'r-', linewidth=2, label='双高斯混合')
        plt.bar(range(256), aggregate_stats['normalized_combined_histogram'], width=1.0, alpha=0.3, color='gray', label='原始数据')
        
        plt.title('双高斯分布拟合与原始数据比较')
        plt.xlabel('像素值')
        plt.ylabel('百分比 (%)')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    else:
        plt.hist(aggregate_stats['nonzero_percentages'], bins=20)
        plt.title('非零像素百分比分布')
        plt.xlabel('非零像素百分比')
        plt.ylabel('图像数量')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '双高斯拟合.png'), dpi=300)
    plt.close()  # 关闭图形而不显示
    
    # 保存组合图像（四合一图）
    plt.figure(figsize=(20, 15))
    
    # 子图1
    plt.subplot(221)
    plt.bar(range(256), aggregate_stats['normalized_combined_histogram'], width=1.0, alpha=0.6, label='像素分布')
    if bimodal_fit:
        plt.plot(range(256), bimodal_fit['fitted_curve'], 'r-', linewidth=2, label='双高斯拟合')
        plt.plot(range(256), bimodal_fit['gaussian1'], 'g--', linewidth=1.5, 
                 label=f'高斯1 (μ={bimodal_fit["mean1"]:.1f}, σ={bimodal_fit["std1"]:.1f})')
        plt.plot(range(256), bimodal_fit['gaussian2'], 'b--', linewidth=1.5, 
                 label=f'高斯2 (μ={bimodal_fit["mean2"]:.1f}, σ={bimodal_fit["std2"]:.1f})')
        plt.legend(loc='upper right', fontsize=8)
    plt.title(f'所有图像聚合像素分布 ({aggregate_stats["total_images"]} 个图像)')
    plt.xlabel('像素值 (0-255)')
    plt.ylabel('百分比 (%)')
    plt.xlim([0, 255])
    plt.grid(True, alpha=0.3)
    
    # 子图2
    plt.subplot(222)
    plt.plot(range(256), aggregate_stats['normalized_cumulative_combined'], 'r', linewidth=2)
    plt.title('所有图像聚合累进像素分布')
    plt.xlabel('像素值 (0-255)')
    plt.ylabel('累计百分比 (%)')
    plt.xlim([0, 255])
    plt.ylim([0, 100])
    plt.grid(True, alpha=0.3)
    
    # 子图3
    plt.subplot(223)
    peak_values = aggregate_stats['common_peak_values']
    plt.bar(peak_values, [aggregate_stats['normalized_combined_histogram'][p] for p in peak_values])
    plt.title('常见峰值像素值')
    plt.xlabel('像素值')
    plt.ylabel('百分比 (%)')
    plt.grid(True, alpha=0.3)
    
    # 子图4
    plt.subplot(224)
    if bimodal_fit:
        x = np.linspace(0, 255, 500)
        params = bimodal_fit['params']
        g1 = params[0] * np.exp(-(x - params[1])**2 / (2 * params[2]**2))
        g2 = params[3] * np.exp(-(x - params[4])**2 / (2 * params[5]**2))
        combined = g1 + g2
        plt.plot(x, g1, 'g-', linewidth=1.5, alpha=0.7, label=f'高斯1: μ={params[1]:.1f}')
        plt.plot(x, g2, 'b-', linewidth=1.5, alpha=0.7, label=f'高斯2: μ={params[4]:.1f}')
        plt.plot(x, combined, 'r-', linewidth=2, label='双高斯混合')
        plt.bar(range(256), aggregate_stats['normalized_combined_histogram'], width=1.0, alpha=0.3, color='gray', label='原始数据')
        plt.title('双高斯分布拟合与原始数据比较')
        plt.xlabel('像素值')
        plt.ylabel('百分比 (%)')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    else:
        plt.hist(aggregate_stats['nonzero_percentages'], bins=20)
        plt.title('非零像素百分比分布')
        plt.xlabel('非零像素百分比')
        plt.ylabel('图像数量')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '像素分布统计.png'), dpi=300)
    plt.close()  # 关闭图形而不显示
    
    # 创建3D展示各样本的像素分布
    # 为了可视化效果，选择最多40个样本展示
    max_samples = min(40, len(aggregate_stats['all_histograms']))
    selected_indices = np.linspace(0, len(aggregate_stats['all_histograms'])-1, max_samples, dtype=int)
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 准备数据
    x = np.arange(0, 256)
    for i, idx in enumerate(selected_indices):
        y = [i] * 256
        z = aggregate_stats['all_histograms'][idx]
        
        # 绘制3D曲线
        ax.plot(x, y, z, linewidth=1.5, alpha=0.7, label=f"样本 {i+1}")
        
        # 添加峰值点标记
        peaks = np.where((z > 0.5) & (z > np.roll(z, 1)) & (z > np.roll(z, -1)))[0]
        if len(peaks) > 0:
            for peak in peaks:
                ax.scatter([peak], [i], [z[peak]], color='red', s=30)
    
    ax.set_xlabel('像素值')
    ax.set_ylabel('样本索引')
    ax.set_zlabel('百分比 (%)')
    ax.set_title('多样本像素分布3D视图')
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '像素分布3D视图.png'), dpi=300)
    plt.close()  # 关闭图形而不显示
    
    # 打印统计信息
    print("\n========== 图像统计信息汇总 ==========")
    print(f"总图像数: {aggregate_stats['total_images']}")
    print(f"总像素数: {aggregate_stats['total_pixels']:,}")
    print("\n--- 像素值分布 ---")
    print(f"零值像素百分比: {aggregate_stats['zero_percentages']['mean']:.2f}%")
    print(f"非零像素百分比: {aggregate_stats['nonzero_percentages']['mean']:.2f}%")
    
    if bimodal_fit:
        print("\n--- 双高斯分布拟合结果 ---")
        print(f"高斯1: 均值 = {bimodal_fit['mean1']:.2f}, 标准差 = {bimodal_fit['std1']:.2f}, 振幅 = {bimodal_fit['params'][0]:.2f}")
        print(f"高斯2: 均值 = {bimodal_fit['mean2']:.2f}, 标准差 = {bimodal_fit['std2']:.2f}, 振幅 = {bimodal_fit['params'][3]:.2f}")
    
    # 导出统计数据到CSV
    import csv
    with open(os.path.join(output_dir, '统计分析数据.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['总图像数', '总像素数', '零值像素百分比', '非零像素百分比'])
        writer.writerow([
            aggregate_stats['total_images'], 
            aggregate_stats['total_pixels'],
            aggregate_stats['zero_percentages']['mean'],
            aggregate_stats['nonzero_percentages']['mean']
        ])
        
        # 如果双高斯拟合成功，添加拟合结果
        if bimodal_fit:
            writer.writerow([])
            writer.writerow(['双高斯分布拟合结果', '', '', ''])
            writer.writerow(['分布', '均值', '标准差', '振幅'])
            writer.writerow(['高斯1', bimodal_fit['mean1'], bimodal_fit['std1'], bimodal_fit['params'][0]])
            writer.writerow(['高斯2', bimodal_fit['mean2'], bimodal_fit['std2'], bimodal_fit['params'][3]])
    
    print(f"\n统计数据和图表已保存到 '{output_dir}' 目录")

# 使用示例
if __name__ == "__main__":
    # 替换为您的mask文件夹路径
    mask_folder = "./xxxxx/mask/train/"
    
    # 分析文件夹中的所有mask
    all_stats, aggregate_stats = analyze_folder_masks(mask_folder)
    
    if aggregate_stats:
        visualize_aggregate_stats(aggregate_stats)
    
    # 可选：查看单个图像的分布
    # single_mask_path = "xxxxx.png"