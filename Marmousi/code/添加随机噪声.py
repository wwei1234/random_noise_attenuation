import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from matplotlib import rcParams


CLEAN_PATH = r"D:\桌面\U_Net3\开源数据集训练CODE\data\clean.npy"
NOISED_SAVE_PATH = r"D:\桌面\U_Net3\开源数据集训练CODE\data\noised(k=1).npy"
NOISE_SAVE_PATH = r"D:\桌面\U_Net3\开源数据集训练CODE\data\noise(k=1).npy"

DT = 0.004
LOWCUT = 15
HIGHCUT = 55
FILTER_ORDER = 4
NOISE_SCALE_K = 1
IMAGE_EXTENT = (0, 512, 512, 0)
CMAP = "seismic"


def setup_font():
    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False


def calculate_snr(predictions, targets):
    noise = predictions - targets
    signal_power = np.mean(targets**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)


def filter_data(data, lowcut, highcut, dt=DT, order=FILTER_ORDER):
    nyquist = 0.5 / dt
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(N=order, Wn=[low, high], btype='band', analog=False)
    return signal.filtfilt(b, a, data, axis=0)


def add_noise(data, std_dev, k, lowcut, highcut):
    noise = np.random.randn(*data.shape) * (std_dev * k)
    noise = filter_data(noise, lowcut, highcut)
    return data + noise, noise


def plot_noise_result(clean, noised):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Clean Data')
    plt.imshow(clean, aspect='equal', cmap=CMAP, extent=IMAGE_EXTENT)
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')

    plt.subplot(1, 2, 2)
    plt.title('Noisy Data')
    plt.imshow(noised, aspect='equal', cmap=CMAP, extent=IMAGE_EXTENT)
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')
    plt.show()


def main():
    setup_font()
    clean = np.load(CLEAN_PATH)
    std = np.std(clean)
    print(std)

    noised, noise = add_noise(clean, std_dev=std, k=NOISE_SCALE_K, lowcut=LOWCUT, highcut=HIGHCUT)
    np.save(NOISED_SAVE_PATH, noised)
    np.save(NOISE_SAVE_PATH, noise)

    snr = calculate_snr(noised, clean)
    print(snr)
    plot_noise_result(clean, noised)


if __name__ == "__main__":
    main()
