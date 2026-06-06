import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


DATA_PATH = r"D:\桌面\随机噪声压制\Stratton\数据汇总\MAMI\MAMI_CONVENTIONAL_NOISE.npy"
DT = 0.004
FIGURE_SIZE = (8, 5)
LINE_WIDTH = 1.5
LINE_COLOR = 'b'

# plt.savefig(r"D:\桌面\U_Net3\开源数据集训练CODE\可视化\地震记录频谱", dpi = 2400)


def setup_font():
    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False


def calculate_average_spectrum(data, dt):
    nt = data.shape[0]
    fft_result = np.fft.fft(data, axis=0)
    magnitude_spectrum = np.abs(fft_result).mean(axis=1)
    freqs = np.fft.fftfreq(nt, d=dt)

    freqs = freqs[:nt // 2]
    magnitude_spectrum = magnitude_spectrum[:nt // 2]

    nonzero_idx = freqs > 0
    return freqs[nonzero_idx], magnitude_spectrum[nonzero_idx]


def plot_spectrum(freqs, magnitude_spectrum):
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(freqs, magnitude_spectrum, color=LINE_COLOR, lw=LINE_WIDTH)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Seismic Record Spectrum")
    plt.grid()
    plt.show()


def main():
    setup_font()
    data = np.load(DATA_PATH)
    freqs, magnitude_spectrum = calculate_average_spectrum(data, DT)
    plot_spectrum(freqs, magnitude_spectrum)

    dominant_freq = freqs[np.argmax(magnitude_spectrum)]
    print(f"Dominant frequency: {dominant_freq:.2f} Hz")


if __name__ == "__main__":
    main()
