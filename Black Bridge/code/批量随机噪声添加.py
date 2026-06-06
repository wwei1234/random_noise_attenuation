import os
import numpy as np
import scipy.signal as signal

INPUT_DIR = r"D:\桌面\实际数据去噪\data2\clean3"  # 输入文件夹路径
OUTPUT_DIR = r"D:\桌面\实际数据去噪\data2\noised"  # 输出文件夹路径

def add_noise(data, std_dev, k):
    noise = np.random.randn(*data.shape) * (std_dev * k)
    # noise = filter_data(noise, lowcut, highcut)
    return data + noise, noise

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

def batch_add_noise(input_dir, output_dir, k=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            path = os.path.join(input_dir, filename)
            try:
                clean = np.load(path)
                std = np.std(clean)
                noised, _ = add_noise(clean, std, k)  # 只保留加噪后的数据，去除噪声部分

                # 原文件名加上_noised后缀
                base_name = os.path.splitext(filename)[0]
                noised_path = os.path.join(output_dir, f"{base_name}_noised.npy")

                np.save(noised_path, noised)

                print(f"Processed: {filename} -> {base_name}_noised.npy")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

def main():
    batch_add_noise(INPUT_DIR, OUTPUT_DIR, k=2)


if __name__ == "__main__":
    main()
