from scipy.io import loadmat
import numpy as np

# 加载 .mat 文件
mat_data = loadmat(r'D:\桌面\项目\Stratton\data\sacndon_d50.mat')  # 替换为实际文件名

# 排除掉 MATLAB 自动加的字段（如 __header__、__version__、__globals__）
variables = {k: v for k, v in mat_data.items() if not k.startswith('__')}

# 保存每个变量为单独的 .npy 文件
for var_name, value in variables.items():
    np.save(f"{var_name}.npy", value)
    print(f"已保存变量 {var_name} -> {var_name}.npy")
