import numpy as np
import matplotlib.pyplot as plt
import torch
from UNet_3Plus import UNet_3Plus_DeepSup
from U_Net import UNet
from matplotlib import rcParams
import segyio
from skimage.metrics import structural_similarity as ssim

rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

# model = UNet_3Plus_DeepSup(in_channels=1, n_classes=1)
# model = UNet()
# model_path = r'D:\桌面\U_Net3\开源数据集训练CODE\model\model_epoch_200.pth'
# model_path = r"D:\桌面\U_Net3\开源数据集训练CODE\model2\model_epoch_200.pth"
# model_path = r'D:\桌面\U_Net3\开源数据集训练CODE\conventional_model\model_epoch_100.pth'
# model.load_state_dict(torch.load(model_path, weights_only=True))


# 计算信噪比（SNR）
def calculate_snr(predictions, targets):
    noise = predictions - targets
    signal_power = np.mean(targets**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)

# def calculate_snr(predictions, targets, use_variance=True):
#     predictions = np.asarray(predictions)
#     targets = np.asarray(targets)
#     noise = predictions - targets
    
#     if use_variance:
#         signal_power = np.var(targets)
#         noise_power = np.var(noise)
#     else:
#         signal_power = np.mean(targets**2)
#         noise_power = np.mean(noise**2)
    
#     # 避免除以零
#     epsilon = 1e-10
#     if noise_power == 0:
#         if signal_power == 0:
#             return 0.0  # 信号和噪声均为零，定义SNR为0
#         else:
#             return np.inf  # 无噪声，SNR无穷大
    
#     snr = 10 * np.log10(signal_power / noise_power)
#     return snr

def mse(x, y):
    return np.mean((x - y) ** 2)

def psnr(x, y, max_value=1.0):
    """计算峰值信噪比（PSNR）"""
    mse_value = mse(x, y)
    if mse_value == 0:
        return float('inf')  # 完全相同的数据，PSNR 为无穷大
    return 10 * np.log10((max_value ** 2) / mse_value)

# def read_segy(data_dir,shotnum=0):
#     with segyio.open(data_dir,'r',ignore_geometry=True) as f:
#         sourceX = f.attributes(segyio.TraceField.SourceX)[:]
#         trace_num = len(sourceX) #number of all trace
#         if shotnum:
#             shot_num = shotnum 
#         else:
#             shot_num = len(set(sourceX)) #shot number 
#         len_shot = trace_num//shot_num   #The length of the data in each shot data
#         time = f.trace[0].shape[0]
#         print('start read segy data')
#         data = np.zeros((shot_num,time,len_shot))
#         for j in range(0,shot_num):
#             data[j,:,:] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
#         return data
    
# segyfile = r"D:\桌面\U_Net3\DATA\elastic-marmousi-model\elastic-marmousi-model\processed_data\SEGY-Time\SYNTHETIC_time.segy\SYNTHETIC_time.segy"
# marmousi = read_segy(segyfile,shotnum=1)

# marmousi = torch.tensor(marmousi, dtype=torch.float32)
# marmousi = marmousi[:, 200:712, 2000:2512]
# min_val = marmousi.min()
# max_val = marmousi.max()
# marmousi = (marmousi - min_val) / (max_val - min_val)
# std_dev = marmousi.std()
# # 2. 生成噪声，标准差为原始数据标准差的0.2倍
# noise = torch.randn_like(marmousi) * (std_dev * 0.6)
# # 3. 将噪声加到原始张量上
# marmousi_noisy = marmousi + noise
# # 2. 进行0-1归一化
# input = marmousi_noisy.unsqueeze(0)

# 输入网络进行预测
# model.eval()  # 切换到评估模式
# with torch.no_grad():
#     prediction = model(input)
#     prediction, *_ = prediction  # 解包元组，取第一个元素
#     initial = input.squeeze(0).squeeze(0).numpy()
#     pure = marmousi.squeeze(0).numpy()
#     result = prediction.squeeze(0).squeeze(0).numpy()
#     noise = initial - result

    # fk(initial)
    
    # result = result.astype(float)
    # pure = pure.astype(float)
    # data_range = result.max() - result.min()

    # Mse = mse(result, pure)
    # Ssim = ssim(result, pure, full = False, data_range=data_range)
    # Snr = calculate_snr(result, pure)
    # Psnr = psnr(result, pure)
    # print("MSE:", Mse, "\n","SSIM:", Ssim, "\n", "SNR:", Snr, "\n","PSNR:", Psnr)

    # re = result[256]
    # tag = pure[256]

    # plt.plot(tag, label = "clean", color = "blue", linestyle = "-", linewidth = 2)
    # plt.plot(re, label = "prediction", color = "red", linestyle = "--", linewidth = 2)
    # plt.xlabel("Samples")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # title_text = f"Single Trace Prediction (noise = 0.4)"
    # plt.title(title_text, fontsize=14)
    # plt.show()

result = np.load(r"D:\桌面\项目\U_Net3+ 合成记录测试\开源数据集训练CODE\data\noised(k=3).npy")
pure = np.load(r"D:\桌面\项目\U_Net3+ 合成记录测试\开源数据集训练CODE\data\clean.npy")

data_range = result.max() - result.min()
Mse = mse(result, pure)
Ssim = ssim(result, pure, full = False, data_range=data_range)
Snr = calculate_snr(result, pure)
Psnr = psnr(result, pure)
print("MSE:", Mse, "\n","SSIM:", Ssim, "\n", "SNR:", Snr, "\n","PSNR:", Psnr)
