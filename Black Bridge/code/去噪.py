import numpy as np
import matplotlib.pyplot as plt
import torch
from U_Net_CBAM import UNet 
from UNet_3Plus import UNet_3Plus_DeepSup  # 请确保你的模型类是正确导入的
from matplotlib import rcParams
# from U_Net_CBAM import UNet
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

# model = UNet()
# model_path = r'D:\桌面\项目\raw_data\conventional_model\model_epoch_27.pth'
# model.load_state_dict(torch.load(model_path, weights_only=True))

# model = UNet()
# model_path = r'D:\桌面\随机噪声压制\transfer_model_11_5\model_epoch_390.pth'
# model.load_state_dict(torch.load(model_path, weights_only=True))

model = UNet_3Plus_DeepSup(in_channels=1, n_classes=1, dropout_prob = 0)
# model_path = r'D:\桌面\raw_data\迁移学习训练日志\迁移学习训练日志_final_best.pth'
# model_path = r'D:\桌面\随机噪声压制\Black Bridge\U_Net3+_model_6_9\model_epoch_196.pth'
model_path = r'D:\桌面\随机噪声压制\Black Bridge\U_Net3+_model_6_9\model_epoch_100.pth'
model.load_state_dict(torch.load(model_path, weights_only=True))

marmousi_clean = np.load(r"D:\桌面\随机噪声压制\Black Bridge\data\clean.npy")
marmousi_noisy = np.load(r"D:\桌面\随机噪声压制\Black Bridge\data\noised(k=2)(15-55).npy")
marmousi_clean = torch.tensor(marmousi_clean, dtype=torch.float32)
marmousi_noisy = torch.tensor(marmousi_noisy, dtype=torch.float32)
input = marmousi_noisy.unsqueeze(0).unsqueeze(0)

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
    pure = marmousi_clean.squeeze(0).numpy()
    result = prediction.squeeze(0).squeeze(0).numpy()
    noise = initial - result
    res = pure - result

    snr_noised = calculate_snr(initial, pure)
    snr_denoised = calculate_snr(result, pure)
    print("before:",snr_noised,"\n","after:",snr_denoised)

    # np.save(r"D:\桌面\项目\raw_data\data\U_Net3+_denoised(k=2)", result)
    # np.save(r"D:\桌面\项目\raw_data\data\U_Net3+_noise(k=2)", noise)
    
    plt.figure()
    # plt.subplot(1, 3, 1)
    plt.title('含噪数据')
    plt.imshow(marmousi_noisy, aspect='auto', cmap='seismic', extent=(0, 48, 4000, 0))
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')

    plt.figure()
    # plt.subplot(1, 3, 2)
    plt.title('去噪结果')
    plt.imshow(result, aspect='auto', cmap='seismic', extent=(0, 48, 4000, 0))
    plt.clim(0, 1)
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')

    plt.figure()
    # plt.subplot(1, 3, 3)
    plt.title('去除的噪声')
    plt.imshow(noise, aspect='auto', cmap='seismic', extent=(0, 48, 4000, 0))
    plt.clim(-1, 1)
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')
    plt.show()

