# import numpy as np
# import matplotlib.pyplot as plt

# # 读取npy文件
# data = np.load(r"D:\桌面\f-x域去噪\data\marmousi_clean.npy")  # 替换为你的文件路径
# noisy = np.load(r"D:\桌面\f-x域去噪\data\marmousi_noisy_0.6.npy")

# data = noisy - data
# # 计算二维FFT
# fft_result = np.fft.fft2(data)
# fft_shifted = np.fft.fftshift(fft_result)  # 将零频率移到中心

# # 计算频谱（振幅谱）
# magnitude_spectrum = np.abs(fft_shifted)

# dx = 25  # 每道间距（米）
# dt = 0.004  # 采样间隔（秒）
# nx, nt = data.shape  # 获取数据的空间和时间维度

# # 计算空间和时间频率轴
# fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))  # 空间频率（波数）
# ft = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))  # 时间频率（Hz）

# plt.figure(figsize=(10, 6))
# plt.imshow(np.log1p(magnitude_spectrum), aspect='auto', cmap='jet', extent=[fx.min(), fx.max(), ft.min(), ft.max()])
# plt.colorbar(label="Log Amplitude")
# plt.xlabel("Spatial Frequency (1/m)")
# plt.ylabel("Temporal Frequency (Hz)")
# plt.title("Seismic Spectrum")
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块
# 读取npy文件
# data = np.load(r"D:\桌面\f-x域去噪\data\marmousi_clean.npy")  # 替换为你的文件路径

# noisy = np.load(r"D:\桌面\f-x域去噪\data\marmousi_noisy_0.6.npy")

# clean = np.load(r"D:\桌面\U_Net3\开源数据集训练CODE\data\clean.npy")
# noised = np.load(r"D:\桌面\U_Net3\开源数据集训练CODE\data\noised(k=3).npy")
# data = noised - clean

data = np.load(r"D:\桌面\项目\shengbei\data\shengbei_pstm_cb.npy")
nt, nx = data.shape  # nt: 采样点数（时间），nx: 道数（空间）

dt = 0.001  # 时间采样间隔（秒），请根据实际情况修改

# 对每一条地震道（每列）进行FFT，并取平均
fft_result = np.fft.fft(data, axis=0)  # 只对时间轴做FFT
magnitude_spectrum = np.abs(fft_result).mean(axis=1)  # 对所有地震道取平均

# 计算频率轴（Hz）
freqs = np.fft.fftfreq(nt, d=dt)  # 计算频率刻度
freqs = freqs[:nt // 2]  # 只保留正频率部分
magnitude_spectrum = magnitude_spectrum[:nt // 2]  # 只保留正频率部分

# 过滤掉零频率（直流分量）
nonzero_idx = freqs > 0  # 找到大于0的频率索引
freqs = freqs[nonzero_idx]  
magnitude_spectrum = magnitude_spectrum[nonzero_idx]  

# 重新绘制去除零频率后的频谱
plt.figure(figsize=(8, 5))
plt.plot(freqs, magnitude_spectrum, color='b', lw=1.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("地震记录频谱")
plt.grid()
# plt.savefig(r"D:\桌面\U_Net3\开源数据集训练CODE\可视化\地震记录频谱", dpi = 2400)
plt.show()

# 计算去掉零频率后的主频
dominant_freq = freqs[np.argmax(magnitude_spectrum)]
print(f"主频: {dominant_freq:.2f} Hz")

