import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import rcParams
from U_Net_CBAM import UNet
import segyio
import os
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

def read_segy(data_dir: str, shotnum: int = 0) -> np.ndarray:
    """读取segy文件"""
    with segyio.open(data_dir, 'r', ignore_geometry=True) as f:
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX)
        if shotnum:
            shot_num = shotnum 
        else:
            shot_num = len(set(sourceX))
        len_shot = trace_num // shot_num
        time = f.trace[0].shape[0]
        print(f'开始读取 {os.path.basename(data_dir)}, shots: {shot_num}, traces: {len_shot}, samples: {time}')
        
        data = np.zeros((shot_num, time, len_shot))
        for j in range(shot_num):
            data[j, :, :] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
    return data

def normalize_data(data: np.ndarray) -> np.ndarray:
    """数据归一化"""
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val + 1e-8)

model = UNet()
model_path = r"model.pth"  # 请将路径替换为实际模型文件路径
model.load_state_dict(torch.load(model_path, weights_only=True))

path = r"data.sgy" # 请将路径替换为实际数据文件路径
noisy = normalize_data(read_segy(path, shotnum=1)[0])  # 读取第一个shot
noisy = torch.tensor(noisy, dtype=torch.float32)
input = noisy.unsqueeze(0).unsqueeze(0)

def calculate_snr(predictions, targets):
    noise = predictions - targets
    signal_power = np.mean(targets**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)

# 输入网络进行预测
model.eval()  # 切换到评估模式
with torch.no_grad():
    prediction = model(input)
    prediction, *_ = prediction  # 解包元组，取第一个元素
    initial = input.squeeze(0).squeeze(0).numpy()
    result = prediction.squeeze(0).squeeze(0).numpy()
    noise = initial - result
    res = initial - result

    plt.figure()
    # plt.subplot(3, 1, 1)
    plt.title('含噪数据')
    plt.imshow(noisy, aspect='auto', cmap='seismic', extent=(0, 359, 400, 0))
    plt.colorbar(label='Amplitude (VDD)')
    # plt.ylabel('Samples')

    plt.figure()
    # plt.subplot(3, 1, 2)
    plt.title('去噪结果')
    plt.imshow(result, aspect='auto', cmap='seismic', extent=(0, 359, 400, 0))
    plt.clim(0, 1)
    plt.colorbar(label='Amplitude (VDD)')
    # plt.ylabel('Samples')

    plt.figure()
    # plt.subplot(3, 1, 3)
    plt.title('去除的噪声')
    plt.imshow(res, aspect='auto', cmap='seismic', extent=(0, 359, 400, 0))
    plt.clim(-0.2, 0.2)
    plt.colorbar(label='Amplitude (VDD)')
    # plt.ylabel('Samples')
    plt.show()

