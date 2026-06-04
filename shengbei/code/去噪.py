import numpy as np
import matplotlib.pyplot as plt
import torch
from U_Net import UNet 
from UNet_3Plus import UNet_3Plus_DeepSup  # 请确保你的模型类是正确导入的
import segyio
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

# model = UNet()
# model_path = r'D:\桌面\U_Net3\开源数据集训练CODE\conventional_model\model_epoch_600.pth'
# model.load_state_dict(torch.load(model_path, weights_only=True))

model = UNet_3Plus_DeepSup(in_channels=1, n_classes=1)
# model_path = r'D:\桌面\U_Net3\开源数据集训练CODE\U_Net3+_model_4_10\model_epoch_431.pth'
model_path = r'D:\桌面\U_Net3\开源数据集训练CODE\U_Net3+_model_4_10\model_epoch_600.pth'
model.load_state_dict(torch.load(model_path, weights_only=True))

# marmousi_clean = np.load(r"D:\桌面\U_Net3\开源数据集训练CODE\实际地震数据\clean.npy")
# marmousi_noisy = np.load(r"D:\桌面\U_Net3\开源数据集训练CODE\实际地震数据\noised(k=3)(15-55).npy")
# marmousi_noisy = np.load(r"D:\桌面\U_Net3\开源数据集训练CODE\data\noised(k=1).npy")

marmousi_clean = np.load(r"D:\桌面\项目\raw_data\data\shandongnoised.npy")
marmousi_noisy = np.load(r"D:\桌面\项目\raw_data\data\shandongnoised.npy")

marmousi_clean = torch.tensor(marmousi_clean, dtype=torch.float32)
marmousi_noisy = torch.tensor(marmousi_noisy, dtype=torch.float32)
input = marmousi_noisy.unsqueeze(0).unsqueeze(0)

def calculate_snr(predictions, targets, use_variance=False): 
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    noise = predictions - targets
    
    if use_variance:
        signal_power = np.var(targets)
        noise_power = np.var(noise)
    else:
        signal_power = np.mean(targets**2)
        noise_power = np.mean(noise**2)
    
    # 避免除以零
    epsilon = 1e-10
    if noise_power == 0:
        if signal_power == 0:
            return 0.0  # 信号和噪声均为零，定义SNR为0
        else:
            return np.inf  # 无噪声，SNR无穷大
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# 输入网络进行预测
model.eval()  # 切换到评估模式
with torch.no_grad():
    prediction = model(input)
    prediction, *_ = prediction  # 解包元组，取第一个元素
    initial = input.squeeze(0).squeeze(0).numpy()
    pure = marmousi_clean.squeeze(0).numpy()
    result = prediction.squeeze(0).squeeze(0).numpy()
    noise = initial - result
    res = pure - result
    # np.save(r"D:\桌面\U_Net3\开源数据集训练CODE\data\conventional_U-Net_denoised(k=3)",result)

    snr_noised = calculate_snr(initial, pure)
    snr_denoised = calculate_snr(result, pure)
    print("before:",snr_noised,"\n","after:",snr_denoised)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('含噪数据')
    plt.imshow(initial, aspect='equal', cmap='seismic', extent=(0, 512, 512, 0))
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')

    plt.subplot(2, 2, 3)
    plt.title('不含噪数据标签')
    plt.imshow(pure, aspect='equal', cmap='seismic', extent=(0, 512, 512, 0))
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')

    plt.subplot(2, 2, 2)
    plt.title('去噪结果')
    plt.imshow(result, aspect='equal', cmap='seismic', extent=(0, 512, 512, 0))
    plt.clim(0, 1)
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')

    plt.subplot(2, 2, 4)
    plt.title('残差')
    plt.imshow(res, aspect='equal', cmap='seismic', extent=(0, 512, 512, 0))
    plt.clim(-1, 1)
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')

    plt.tight_layout()
    # plt.savefig(r"D:\桌面\U_Net3\开源数据集训练CODE\训练日志(15-55hz)\model_725.png", dpi = 2400)
    plt.show()

