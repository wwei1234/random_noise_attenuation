import os

import numpy as np
import scipy.signal as signal
import segyio
from numpy.lib.stride_tricks import as_strided


SEGY_FILE = r"D:\桌面\U_Net3\DATA\SYNTHETIC_time.segy"
OUTPUT_DIR = "数据集(整个剖面)"

SHOT_NUM = 1
MARMOUISI_TIME_SLICE = slice(100, 750)
SLICE_SIZE = 128
STRIDE = 32
START_INDEX = 2788
LOW_CUT = 15
HIGH_CUT = 55
DT = 0.004
FILTER_ORDER = 4
NOISE_SCALE_K = 1

NORMALIZED_DIR_NAME = "normalized_npy"
NOISY_DIR_NAME = "noisy_npy"
NOISE_DIR_NAME = "noise_npy"


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


def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)


def filter_data(data, lowcut, highcut, dt=DT, order=FILTER_ORDER):
    nyquist = 0.5 / dt
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(N=order, Wn=[low, high], btype='band', analog=False)
    return signal.filtfilt(b, a, data, axis=0)


def add_noise(data, std_dev, k, lowcut, highcut):
    noise = np.random.randn(*data.shape) * (std_dev * k)
    noise = filter_data(noise, lowcut, highcut)
    return data + noise


def process_data(data, slice_size=SLICE_SIZE, stride=STRIDE):
    time, trace_len = data.shape
    shape = (
        (time - slice_size) // stride + 1,
        (trace_len - slice_size) // stride + 1,
        slice_size,
        slice_size,
    )
    strides = (
        data.strides[0] * stride,
        data.strides[1] * stride,
        data.strides[0],
        data.strides[1],
    )
    slices = as_strided(data, shape=shape, strides=strides)
    return slices.reshape(-1, slice_size, slice_size)


def augment_image(data):
    return [
        np.fliplr(data),
        np.flipud(data),
        np.rot90(data, k=1),
        np.rot90(data, k=2),
        np.rot90(data, k=3),
    ]


def create_output_dirs(output_dir):
    npy_dir_norm = os.path.join(output_dir, NORMALIZED_DIR_NAME)
    npy_dir_noisy = os.path.join(output_dir, NOISY_DIR_NAME)
    npy_dir_noise = os.path.join(output_dir, NOISE_DIR_NAME)

    os.makedirs(npy_dir_norm, exist_ok=True)
    os.makedirs(npy_dir_noisy, exist_ok=True)
    os.makedirs(npy_dir_noise, exist_ok=True)
    return npy_dir_norm, npy_dir_noisy, npy_dir_noise


def save_slice(slice_data, file_idx, output_dirs):
    npy_dir_norm, npy_dir_noisy, npy_dir_noise = output_dirs
    normalized_slice = normalize_data(slice_data)
    local_std = slice_data.std()
    noisy_slice = add_noise(normalized_slice, local_std, NOISE_SCALE_K, LOW_CUT, HIGH_CUT)
    noise_data = noisy_slice - normalized_slice

    np.save(os.path.join(npy_dir_norm, f"normalized_{file_idx}.npy"), normalized_slice)
    np.save(os.path.join(npy_dir_noisy, f"noisy_{file_idx}.npy"), noisy_slice)
    np.save(os.path.join(npy_dir_noise, f"noise_{file_idx}.npy"), noise_data)


def process_and_save_slices(slices, output_dirs):
    for idx, slice_data in enumerate(slices):
        file_idx = str(idx + START_INDEX).zfill(4)
        save_slice(slice_data, file_idx, output_dirs)

        for i, aug_slice in enumerate(augment_image(slice_data)):
            aug_file_idx = f"{file_idx}_{i + 1}"
            save_slice(aug_slice, aug_file_idx, output_dirs)

        print(f"slice group {idx} saved")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_dirs = create_output_dirs(OUTPUT_DIR)

    marmousi = read_segy(SEGY_FILE, shotnum=SHOT_NUM)
    marmousi = marmousi[0][MARMOUISI_TIME_SLICE]
    slices = process_data(marmousi, slice_size=SLICE_SIZE, stride=STRIDE)

    process_and_save_slices(slices, output_dirs)
    print("all files saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
