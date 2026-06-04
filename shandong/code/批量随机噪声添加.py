import os
import numpy as np
import scipy.signal as signal
# 加噪函数（不再进行滤波）
def add_noise(data, std_dev, k):
    noise = np.random.randn(*data.shape) * (std_dev * k)
    # noise = filter_data(noise, lowcut, highcut)
    return data + noise, noise

def filter_data(data, lowcut, highcut):
    dt = 0.004  # 采样时间间隔（秒），请根据实际情况修改
    # 计算 Nyquist 频率（奈奎斯特频率 = 采样率的一半）
    nyquist = 0.5 / dt  
    # 归一化截止频率（相对于 Nyquist 频率）
    low = lowcut / nyquist
    high = highcut / nyquist
    # 设计 4 阶 Butterworth 带通滤波器
    b, a = signal.butter(N=4, Wn=[low, high], btype='band', analog=False)
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    return filtered_data

# 批量加噪主函数
def batch_add_noise(input_dir, output_dir, k=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            path = os.path.join(input_dir, filename)
            try:
                clean = np.load(path)
                std = np.std(clean)
                noised, _ = add_noise(clean, std, k)  # 只保留加噪后的数据，去除噪声部分

                # 原文件名加上_noised后缀
                base_name = os.path.splitext(filename)[0]
                noised_path = os.path.join(output_dir, f"{base_name}_noised.npy")

                np.save(noised_path, noised)

                print(f"Processed: {filename} -> {base_name}_noised.npy")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 设置路径并执行
input_dir = r"D:\桌面\项目\实际数据去噪\data\clean3"  # 输入文件夹路径
output_dir = r"D:\桌面\项目\实际数据去噪\data\noised"  # 输出文件夹路径

batch_add_noise(input_dir, output_dir, k = 4)


# import os
# import shutil
# import re

# def duplicate_npy_files(folder: str, groups: int = 168):
#     """
#     将 folder 下所有符合 ###.npy 或 ###_*.npy 的文件，
#     按序号复制两次，偏移 groups 和 2*groups。

#     例如：
#       001.npy  -> 169.npy, 337.npy
#       001_1.npy -> 169_1.npy, 337_1.npy
#     """
#     # 匹配形如 001.npy 或 001_1.npy 等
#     pattern = re.compile(r"^(\d{3})(.*)\.npy$")
#     files = os.listdir(folder)

#     for fname in files:
#         m = pattern.match(fname)
#         if not m:
#             continue

#         # 原序号与后缀
#         orig_idx = int(m.group(1))
#         suffix   = m.group(2)  # 比如 ""、"_1"、"_2" 等

#         src_path = os.path.join(folder, fname)

#         # 两次复制，偏移 groups 和 2*groups
#         for k in (1, 2):
#             new_idx = orig_idx + k * groups
#             new_name = f"{new_idx:03d}{suffix}.npy"
#             dst_path = os.path.join(folder, new_name)

#             # 如果目标文件已存在，可选择覆盖或跳过
#             if os.path.exists(dst_path):
#                 print(f"跳过已存在: {new_name}")
#             else:
#                 shutil.copy(src_path, dst_path)
#                 print(f"已复制: {fname} -> {new_name}")

# if __name__ == "__main__":
#     data_folder = r"D:\桌面\项目\实际数据去噪\迁移学习训练集"  # 请修改为你的路径
#     duplicate_npy_files(data_folder, groups=168)
