import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


NOISE_PATH = r"D:\桌面\项目\开源数据集训练CODE\data\MAMI_noised(k=3).npy"
DT = 0.004
LOWCUT = 15
HIGHCUT = 55
FILTER_ORDER = 4
FIGURE_SIZE = (12, 6)
CMAP = "seismic"

# np.save(r"D:\桌面\U_Net3\开源数据集训练CODE\data\带通滤波去噪结果.npy", filtered_data)
# np.save("noise1555", filtered_data)


def bandpass_filter(data, dt, lowcut, highcut, order):
    nyquist = 0.5 / dt
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(N=order, Wn=[low, high], btype='band', analog=False)
    return signal.filtfilt(b, a, data, axis=0)


def plot_sections(noise, filtered_data, dt):
    nt, nx = noise.shape
    time = np.arange(nt) * dt
    fig, axs = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=True)

    ax = axs[0]
    im1 = ax.imshow(noise, aspect='auto', cmap=CMAP, extent=[0, nx, time[-1], time[0]])
    ax.set_title("Original Seismic Section")
    ax.set_xlabel("Trace Number")
    ax.set_ylabel("Time (s)")
    fig.colorbar(im1, ax=ax, orientation="vertical")

    ax = axs[1]
    im2 = ax.imshow(filtered_data, aspect='auto', cmap=CMAP, extent=[0, nx, time[-1], time[0]])
    ax.set_title(f"Filtered Seismic Section ({LOWCUT}-{HIGHCUT}Hz)")
    ax.set_xlabel("Trace Number")
    fig.colorbar(im2, ax=ax, orientation="vertical")

    plt.tight_layout()
    plt.show()


def main():
    noise = np.load(NOISE_PATH)
    filtered_data = bandpass_filter(noise, DT, LOWCUT, HIGHCUT, FILTER_ORDER)
    plot_sections(noise, filtered_data, DT)


if __name__ == "__main__":
    main()
