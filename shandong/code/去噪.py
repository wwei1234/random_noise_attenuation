import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import rcParams
from U_Net_CBAM import UNet
import segyio
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

def read_segy(data_dir,shotnum=0):
    with segyio.open(data_dir,'r',ignore_geometry=True) as f:
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX) #number of all trace
        if shotnum:
            shot_num = shotnum 
        else:
            shot_num = len(set(sourceX)) #shot number 
        len_shot = trace_num//shot_num   #The length of the data in each shot data
        time = f.trace[0].shape[0]
        print('start read segy data')
        data = np.zeros((shot_num,time,len_shot))
        for j in range(0,shot_num):
            data[j,:,:] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
        return data

def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

model = UNet()
model_path = r"D:\桌面\随机噪声压制\shandong\CU_Net_828\model_epoch_400.pth"
model.load_state_dict(torch.load(model_path, weights_only=True))

path = r"D:\桌面\随机噪声压制\shandong\雷达\ldsn06.sgy"
# clean = read_segy(path)
noisy = read_segy(path)[0]
noisy = normalize_data(noisy)
# clean = torch.tensor(clean, dtype=torch.float32)
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
    # pure = clean.squeeze(0).numpy()
    result = prediction.squeeze(0).squeeze(0).numpy()
    noise = initial - result
    res = initial - result

    # snr_noised = calculate_snr(initial, pure)
    # snr_denoised = calculate_snr(result, pure)
    # print("before:",snr_noised,"\n","after:",snr_denoised)

    # np.save(r"D:\桌面\项目\shandong\数据\yxsn5cdpn_denoised.npy", result)
    # np.save(r"D:\桌面\项目\shandong\数据\yxsn5cdpn_noise.npy", noise)

    # marmousi_noisy = marmousi_noisy[0:1500, 30:48]
    # result = result[0:1500, 30:48]
    # res = res[0:1500, 30:48]

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
    plt.clim(-1, 1)
    plt.colorbar(label='Amplitude (VDD)')
    # plt.ylabel('Samples')
    plt.show()

