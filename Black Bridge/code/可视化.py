import numpy as np
import matplotlib.pyplot as plt
import torch
from U_Net import UNet 
from matplotlib import rcParams
from UNet_3Plus import UNet_3Plus_DeepSup  # 请确保你的模型类是正确导入的
import scipy.signal as signal

DPI = 600
FIG_WIDTH = 6  # 自定义宽度（英寸）

# data1 = np.load(r'D:\桌面\项目\raw_data\data\U_Net_denoised(k=1).npy')
# data1 = data1[80:360, 312:512]  # 截取数据
# data2 = data2[80:360, 312:512]  # 截取数据
# data2 = np.load(r'D:\桌面\项目\raw_data\data\U_Net3+_denoised(k=2).npy')
# np.save(r'D:\桌面\项目\Stratton\数据汇总\Black Bridge\Black_UNet3+_noise', noise)  # 保存噪声数据
# plt.figure()
# plt.title('加噪数据')
# plt.imshow(noise, aspect='auto', cmap='seismic', extent=(0, 512, 512, 0))
# plt.clim(-1, 1)
# plt.colorbar(label='Amplitude')
# plt.ylabel('Samples')
# plt.savefig(r'D:\桌面\项目\raw_data\可视化\加噪数据(k=1).png', 
# plt.savefig(r'D:\桌面\项目\raw_data\可视化\U-Net3+去噪结果(local).png', 

def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

def global_gain(data):
    return data / np.max(np.abs(data))

def trace_gain(data):
    normalized = data.copy()
    for i in range(data.shape[1]):
        trace_max = np.max(np.abs(data[:, i]))
        if trace_max != 0:
            normalized[:, i] /= trace_max
    return normalized

def agc(data, window_len=30):
    agc_data = np.zeros_like(data)
    half_win = window_len // 2
    for i in range(data.shape[1]):
        trace = data[:, i]
        agc_trace = np.zeros_like(trace)
        for j in range(len(trace)):
            start = max(0, j - half_win)
            end = min(len(trace), j + half_win)
            window = trace[start:end]
            rms = np.sqrt(np.mean(window ** 2)) + 1e-8
            agc_trace[j] = trace[j] / rms
        agc_data[:, i] = agc_trace

    # 归一化到 [0, 1]
    min_val = agc_data.min()
    max_val = agc_data.max()
    agc_data = (agc_data - min_val) / (max_val - min_val + 1e-8)

    return agc_data

def filter_data(data, lowcut, highcut):
    dt = 0.000025 
    # 计算 Nyquist 频率（奈奎斯特频率 = 采样率的一半）
    nyquist = 0.5 / dt  
    # 归一化截止频率（相对于 Nyquist 频率）
    low = lowcut / nyquist
    high = highcut / nyquist
    # 设计 4 阶 Butterworth 带通滤波器
    b, a = signal.butter(N=4, Wn=[low, high], btype='band', analog=False)
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    return filtered_data

def main():
    rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
    rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块
    data2 = np.load(r'D:\桌面\随机噪声压制\raw_data\data\svd_denoised_(k=1).npy')
    height, width = data2.shape
    aspect_ratio = width / height
    fig_height = FIG_WIDTH / aspect_ratio  # 根据比例计算高度
    figsize=(FIG_WIDTH, fig_height)
    plt.figure(figsize=figsize)
    plt.title('U-Net 3Plus去噪结果')
    plt.subplot(1, 2, 2)
    plt.imshow(data2, aspect='auto', cmap='seismic', extent=(312, 512, 280, 80))
    plt.clim(0, 1)
    plt.colorbar(label='Amplitude')
    plt.ylabel('Samples')
    plt.show()  


if __name__ == "__main__":
    main()
