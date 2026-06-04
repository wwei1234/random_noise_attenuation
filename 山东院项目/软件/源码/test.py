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
model_path = r"D:\桌面\山东院项目\软件\测试模型\model_epoch_30.pth"
model.load_state_dict(torch.load(model_path, weights_only=True))

path = r"D:\桌面\山东院项目\软件\测试地震数据\dzsn5cdpn.sgy"
noisy = normalize_data(read_segy(path, shotnum=0)[0][80:600])  # 读取第一个shot
print(f'输入数据形状: {noisy.shape}')
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

    np.save(r'去噪结果2\dzsn5\denoised_result.npy', result)
    np.save(r'去噪结果2\dzsn5\noise.npy', noise)
    np.save(r'去噪结果2\dzsn5\noisy_data.npy', initial)
    np.save(r'去噪结果2\dzsn5\res_data.npy', res)

    plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 1)
    plt.title("Noisy Data", fontsize=24, fontname='Times New Roman')
    plt.imshow(noisy, aspect='auto', cmap='seismic', extent=(0, 541.5, 150, 20))
    plt.xlabel('X/m', fontsize=20, fontname='Times New Roman') 
    plt.ylabel(f"Time/ms", fontsize=20, fontname='Times New Roman') 
    plt.xticks(fontsize=14, fontname='Times New Roman')
    plt.yticks(fontsize=14, fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig('noisy_data.png', dpi=600)


    plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 2)
    plt.title("Denoised Result", fontsize=24, fontname='Times New Roman')
    plt.imshow(result, aspect='auto', cmap='seismic', extent=(0, 541.5, 150, 20))
    plt.xlabel('X/m', fontsize=20, fontname='Times New Roman') 
    plt.ylabel(f"Time/ms", fontsize=20, fontname='Times New Roman') 
    plt.xticks(fontsize=14, fontname='Times New Roman')
    plt.yticks(fontsize=14, fontname='Times New Roman')
    plt.clim(0, 1)
    plt.tight_layout()
    plt.savefig('denoised_result.png', dpi=600)


    plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 3)
    plt.title("Noise", fontsize=24, fontname='Times New Roman')
    plt.imshow(res, aspect='auto', cmap='seismic', extent=(0, 541.5, 150, 20))
    plt.xlabel('X/m', fontsize=20, fontname='Times New Roman') 
    plt.ylabel(f"Time/ms", fontsize=20, fontname='Times New Roman') 
    plt.xticks(fontsize=14, fontname='Times New Roman')
    plt.yticks(fontsize=14, fontname='Times New Roman')
    plt.clim(-1, 1)
    plt.tight_layout()
    plt.savefig('noise.png', dpi=600)
    plt.show()

