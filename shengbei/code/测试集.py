import os
import numpy as np
import segyio
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
import scipy.signal as signal
 
# 读取SEGY数据
def read_segy(data_dir, shotnum=0):
    with segyio.open(data_dir, 'r', ignore_geometry=True) as f:
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX)  # 所有道的数量
        if shotnum:
            shot_num = shotnum
        else:
            shot_num = len(set(sourceX))  # 记录数
        len_shot = trace_num // shot_num  # 每个shot的数据长度
        time = f.trace[0].shape[0]
        print('start read segy data')
        data = np.zeros((shot_num, time, len_shot))
        for j in range(0, shot_num):
            data[j, :, :] = np.asarray([np.copy(x) for x in f.trace[j * len_shot:(j + 1) * len_shot]]).T
        return data

# 归一化处理
def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

# 为数据添加噪声
def add_noise(data, std_dev, k, lowcut, highcut):
    noise = np.random.randn(*data.shape) * (std_dev * k)  # 生成噪声，标准差为 std_dev * k
    noise = filter_data(noise, lowcut, highcut)
    return data + noise

# 从 SEGY 数据中提取多个 128x128 的切片
def process_data(data, slice_size=128, stride=64):
    time, trace_len = data.shape
    shape = (time - slice_size) // stride + 1, (trace_len - slice_size) // stride + 1, slice_size, slice_size
    strides = data.strides[0] * stride, data.strides[1] * stride, data.strides[0], data.strides[1]
    # 使用 as_strided 生成切片
    slices = as_strided(data, shape=shape, strides=strides)
    # 返回切片的二维数组，形状为 (num_slices, slice_size, slice_size)
    return slices.reshape(-1, slice_size, slice_size)

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

# 主程序
segyfile = r"D:\桌面\U_Net3\DATA\SYNTHETIC_time.segy"
output_dir = "测试集"
# 创建文件夹存储结果
os.makedirs(output_dir, exist_ok=True)
# 创建六个子文件夹分别保存噪声、归一化前后数据、和图片
npy_dir_norm = os.path.join(output_dir, "normalized_npy")
npy_dir_noisy = os.path.join(output_dir, "noisy_npy")
npy_dir_noise = os.path.join(output_dir, "noise_npy")
# 创建所有文件夹
os.makedirs(npy_dir_norm, exist_ok=True)
os.makedirs(npy_dir_noisy, exist_ok=True)
os.makedirs(npy_dir_noise, exist_ok=True)

# 读取SEGY数据
marmousi = read_segy(segyfile, shotnum=1)
# 去除第一个维度，得到 (701, 2721)
marmousi = marmousi[0][100:750, 2000:]
# 计算整个数据的标准差
std_dev = marmousi.std()
# 裁剪成多个 128x128 的切片
slices = process_data(marmousi, slice_size=128, stride=32)
low_cut = 15
high_cut = 55
start = 0

# 对每个切片进行处理：归一化并加噪声
for idx, slice_data in enumerate(slices):
    # 生成四位数的文件序号（从 0001 开始）
    file_idx = str(idx+start).zfill(4)
    # 0-1 归一化
    normalized_slice = normalize_data(slice_data)
    # 生成一个 0.1 到 0.3 之间的随机噪声倍数 k
    k = 3
    # 加噪声
    noisy_slice = add_noise(normalized_slice, std_dev, k, low_cut, high_cut)
    # 噪声数据（归一化后数据的噪声部分）
    noise_data = noisy_slice - normalized_slice
    # 保存原始切片的归一化、加噪声及噪声切片
    np.save(os.path.join(npy_dir_norm, f"normalized_{file_idx}.npy"), normalized_slice)
    np.save(os.path.join(npy_dir_noisy, f"noisy_{file_idx}.npy"), noisy_slice)
    np.save(os.path.join(npy_dir_noise, f"noise_{file_idx}.npy"), noise_data)
print("所有文件已保存至:", output_dir)
