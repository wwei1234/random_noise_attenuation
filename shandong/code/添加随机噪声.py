import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.signal as signal
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

def calculate_snr(predictions, targets):
    noise = predictions - targets
    signal_power = np.mean(targets**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)

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

def add_noise(data, std_dev, k):
    noise = np.random.randn(*data.shape) * (std_dev * k)  # 生成噪声，标准差为 std_dev * k
    # noise = filter_data(noise)
    return data + noise, noise

clean = np.load(r"D:\桌面\项目\shandong\数据\sta_f0_D.npy")
# clean = np.load(r"D:\桌面\实际地震数据\实测数据\modified_SN03.npy")

std = np.std(clean)
print(std)

noised, noise = add_noise(clean, std_dev = std, k = 0.6)
snr = calculate_snr(noised, clean)
print(snr)

np.save(r"D:\桌面\项目\shandong\数据\sta_f0_D_noised.npy", noised)
plt.figure()
plt.subplot(1, 3, 1)
plt.title('干净数据')
plt.imshow(clean, aspect='equal', cmap='seismic', extent=(0, 512, 512, 0))
plt.colorbar(label='Amplitude (VDD)')
plt.ylabel('Samples')

plt.subplot(1, 3, 2)
plt.title('含噪数据')
plt.imshow(noised, aspect='equal', cmap='seismic', extent=(0, 512, 512, 0))
plt.colorbar(label='Amplitude (VDD)')
plt.ylabel('Samples')

plt.subplot(1, 3, 3)
plt.title('含噪数据')
plt.imshow(noise, aspect='equal', cmap='seismic', extent=(0, 512, 512, 0))
plt.colorbar(label='Amplitude (VDD)')
plt.ylabel('Samples')
plt.show()