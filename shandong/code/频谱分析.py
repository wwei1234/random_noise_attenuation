import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.signal as signal
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

def filter_data(data, lowcut, highcut):
    dt = 0.000025 #采样时间间隔（秒），请根据实际情况修改
    # 计算 Nyquist 频率（奈奎斯特频率 = 采样率的一半）
    nyquist = 0.5 / dt  
    # 归一化截止频率（相对于 Nyquist 频率）
    low = lowcut / nyquist
    high = highcut / nyquist
    # 设计 4 阶 Butterworth 带通滤波器
    b, a = signal.butter(N=4, Wn=[low, high], btype='band', analog=False)
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    return filtered_data

def plot_spectrum(data, dt=0.025):
    # 加载数据
    nt, nx = data.shape  # nt: 采样点数（时间），nx: 道数（空间）
    # 对数据进行FFT变换
    fft_result = np.fft.fft(data, axis=0)  # 只对时间轴做FFT
    magnitude_spectrum = np.abs(fft_result).mean(axis=1)  # 对所有地震道取平均

    # 计算频率刻度
    freqs = np.fft.fftfreq(nt, d=dt)
    freqs = freqs[:nt // 2]  # 只保留正频率部分
    magnitude_spectrum = magnitude_spectrum[:nt // 2]  # 只保留正频率部分

    # 找到主频
    nonzero_idx = freqs > 0  # 找到大于0的频率索引
    freqs = freqs[nonzero_idx]  
    magnitude_spectrum = magnitude_spectrum[nonzero_idx]  

    dominant_freq = freqs[np.argmax(magnitude_spectrum)]
    print(f"主频: {dominant_freq:.2f} Hz")

    # 绘制频谱
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, magnitude_spectrum, color='b', lw=1.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("地震记录频谱")
    plt.grid()
    plt.show()

data = np.load(r"D:\桌面\项目\shandong\data\shandongnoised.npy")
# data = np.load(r"D:\桌面\实际数据去噪\test\noise.npy")
# data = filter_data(data, 5, 60)
# data = np.load(r"D:\桌面\实际数据去噪\test\001_0_clean.npy")
plot_spectrum(data, dt=0.00025)



