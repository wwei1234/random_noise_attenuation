import matplotlib.pyplot as plt
import numpy as np
import segyio


SEGY_FILE = r"D:\桌面\项目\开源数据集训练CODE\data\SYNTHETIC_time.segy"
SAVE_NPY_PATH = r"D:\桌面\项目\开源数据集训练CODE\data\合成地震剖面.npy"
SHOT_NUM = 1
DATA_SLICE = (slice(None), slice(None))
FIGURE_SIZE = (15, 3)
CMAP = "seismic"

# plt.savefig(r"D:\桌面\Denoise\可视化\合成地震记录", dpi =2400)


def read_segy(data_dir, shotnum=0):
    with segyio.open(data_dir, 'r', ignore_geometry=True) as f:
        source_x = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(source_x)
        shot_num = shotnum if shotnum else len(set(source_x))
        len_shot = trace_num // shot_num
        time = f.trace[0].shape[0]
        print('start read segy data')

        data = np.zeros((shot_num, time, len_shot))
        for j in range(shot_num):
            data[j, :, :] = np.asarray([np.copy(x) for x in f.trace[j * len_shot:(j + 1) * len_shot]]).T
        return data


def seismic_to_3d_array(segyfile):
    with segyio.open(segyfile, "r", 9, 21) as sgydata:
        print('start read segy data')
        ilines = sgydata.ilines
        xlines = sgydata.xlines
        samples = sgydata.samples
        seismic_data = np.zeros((len(ilines), len(xlines), len(samples)))

        for i, iline in enumerate(ilines):
            for j, xline in enumerate(xlines):
                seismic_data[i, j, :] = sgydata.iline[iline][j]

    return seismic_data


def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)


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

    min_val = agc_data.min()
    max_val = agc_data.max()
    return (agc_data - min_val) / (max_val - min_val + 1e-8)


def plot_section(data):
    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(data, CMAP)
    plt.colorbar()
    plt.show()


def main():
    data = read_segy(SEGY_FILE, shotnum=SHOT_NUM)
    data = data[0][DATA_SLICE]
    data = normalize_data(data)
    np.save(SAVE_NPY_PATH, data)
    print(data.shape)
    plot_section(data)


if __name__ == "__main__":
    main()
