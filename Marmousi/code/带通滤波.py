import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

noise = np.load(r"D:\桌面\项目\开源数据集训练CODE\data\MAMI_noised(k=3).npy")
nt, nx =noise.shape  # nt: 时间采样点数, nx: 道数
dt = 0.004  # 采样时间间隔（秒），请根据实际情况修改

lowcut = 15   # 低截止频率（Hz）
highcut = 55  # 高截止频率（Hz）

# 计算 Nyquist 频率（奈奎斯特频率 = 采样率的一半）
nyquist = 0.5 / dt  

# 归一化截止频率（相对于 Nyquist 频率）
low = lowcut / nyquist
high = highcut / nyquist

# 设计 4 阶 Butterworth 带通滤波器
b, a = signal.butter(N=4, Wn=[low, high], btype='band', analog=False)

filtered_data = signal.filtfilt(b, a, noise, axis=0)
# np.save(r"D:\桌面\U_Net3\开源数据集训练CODE\data\带通滤波去噪结果.npy", filtered_data)
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# 设置时间轴
time = np.arange(nt) * dt  # 计算时间轴（秒）

# 绘制原始地震剖面
ax = axs[0]
im1 = ax.imshow(noise, aspect='auto', cmap='seismic', extent=[0, nx, time[-1], time[0]])
ax.set_title("Original Seismic Section")
ax.set_xlabel("Trace Number")
ax.set_ylabel("Time (s)")
fig.colorbar(im1, ax=ax, orientation="vertical")

# 绘制滤波后的地震剖面
ax = axs[1]
im2 = ax.imshow(filtered_data, aspect='auto', cmap='seismic', extent=[0, nx, time[-1], time[0]])
ax.set_title("Filtered Seismic Section (15-55Hz)")
ax.set_xlabel("Trace Number")
fig.colorbar(im2, ax=ax, orientation="vertical")
# np.save("noise1555", filtered_data)
plt.tight_layout()
plt.show()
