# 遍历指定目录下的所有Python文件（排除计算参数量.py本身）
# 动态加载每个Python文件作为模块
# 检查模块中是否包含列表中的任一模型类名
# 如果找到，实例化该类并计算其FLOPs和参数量
# 打印出每个文件中模型的计算结果
# 在处理完每个模型后释放内存
import torch
import torchvision
from thop import profile
import sys
import os
import importlib.util
import inspect

# 指定要搜索的目录
target_directory = r'../models'

# 要查找的模型类名列表
model_names = ["U_Net", "unet", "UNet"]

# 检测是否有可用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 准备输入数据
dummy_input = torch.randn(1, 3, 256, 256).to(device)

# 遍历目标目录下的所有.py文件
for filename in os.listdir(target_directory):
    if filename.endswith('.py') and filename != '计算参数量.py':
        file_path = os.path.join(target_directory, filename)
        module_name = filename[:-3]  # 去掉.py后缀
        
        try:
            # 动态加载模块
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 检查模块中是否有列表中的任一模型类
            found_model = False
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name in model_names:
                    found_model = True
                    print(f"\n{'='*50}")
                    print(f"文件: {filename}")
                    print(f"模型: {name}")
                    
                    # 实例化模型
                    model = obj()
                    model.to(device)
                    
                    # 计算FLOPs和参数量
                    flops, params = profile(model, (dummy_input, ))
                    print(f'FLOPs: {flops/1000000.0:.2f} M, 参数量: {params/1000000.0:.2f} M')
                    print(f"{'='*50}")
                    
                    # 清理内存
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # 每个文件最多处理一个列表中的模型类
                    break
            
            if not found_model:
                print(f"\n{'='*50}")
                print(f"文件 {filename} 中没有找到任何目标模型类")
                print(f"{'='*50}")
        
        except Exception as e:
            print(f"\n{'='*50}")
            print(f"处理文件 {filename} 时出错: {str(e)}")
            print(f"{'='*50}")

print("\n计算完成！")