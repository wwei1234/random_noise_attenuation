import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.signal as signal
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

# 增益调节方式一：全局归一化
def global_gain(data):
    return data / np.max(np.abs(data))

# 增益调节方式二：每道归一化（不推荐用于分析振幅对比）
def trace_gain(data):
    normalized = data.copy()
    for i in range(data.shape[1]):
        trace_max = np.max(np.abs(data[:, i]))
        if trace_max != 0:
            normalized[:, i] /= trace_max
    return normalized

# 增益调节方式三：自动增益控制（滑动窗口）
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

# 读取 .npy 文件
data = np.load(r'D:\桌面\随机噪声压制\shandong\数据\fx_denoised_yxsn5cdpn.npy')
data = data[80:600, :]
print(data.shape)  # 输出数据形状

plt.figure(figsize=(10, 6))
# plt.title('Noisy Data', fontsize=24, family='Times New Roman')
# plt.title('Noise', fontsize=24, family='Times New Roman')
plt.title('Denoised Result', fontsize=24, family='Times New Roman')
plt.imshow(data, aspect='auto', cmap='seismic', extent=(0, 541.5, 150, 20))
plt.clim(0,1)
# plt.colorbar()
plt.ylabel('Time/ms', fontsize=20, family='Times New Roman')
plt.xlabel('X/m',  fontsize=20, family='Times New Roman')
plt.xticks(fontsize=14, fontname='Times New Roman')
plt.yticks(fontsize=14, fontname='Times New Roman')
plt.tight_layout()
plt.savefig(r'D:\桌面\随机噪声压制\shandong\可视化\fx_denoised_yxsn5cdpn', dpi=600, bbox_inches='tight')
plt.show()  

