import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

DT = 0.004  # 采样时间间隔（秒），请根据实际情况修改
LOWCUT = 15   # 低截止频率（Hz）
HIGHCUT = 55  # 高截止频率（Hz）

# np.save("noise1555", filtered_data)

def main():
    noise = np.load(r"D:\桌面\U_Net3\开源数据集训练CODE\data\noised(k=3).npy")
    nt, nx =noise.shape  # nt: 时间采样点数, nx: 道数
    nyquist = 0.5 / DT  
    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist
    b, a = signal.butter(N=4, Wn=[low, high], btype='band', analog=False)
    filtered_data = signal.filtfilt(b, a, noise, axis=0)
    np.save(r"D:\桌面\U_Net3\开源数据集训练CODE\data\带通滤波去噪结果.npy", filtered_data)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    time = np.arange(nt) * DT  # 计算时间轴（秒）
    ax = axs[0]
    im1 = ax.imshow(noise, aspect='auto', cmap='seismic', extent=[0, nx, time[-1], time[0]])
    ax.set_title("Original Seismic Section")
    ax.set_xlabel("Trace Number")
    ax.set_ylabel("Time (s)")
    fig.colorbar(im1, ax=ax, orientation="vertical")
    ax = axs[1]
    im2 = ax.imshow(filtered_data, aspect='auto', cmap='seismic', extent=[0, nx, time[-1], time[0]])
    ax.set_title("Filtered Seismic Section (15-55Hz)")
    ax.set_xlabel("Trace Number")
    fig.colorbar(im2, ax=ax, orientation="vertical")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
