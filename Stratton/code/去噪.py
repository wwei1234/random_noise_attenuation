import numpy as np
import matplotlib.pyplot as plt
import torch
# from U_Net import UNet 
from UNet_3Plus import UNet_3Plus_DeepSup  # 请确保你的模型类是正确导入的
from matplotlib.colors import LinearSegmentedColormap
from U_Net_CBAM import UNet
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

model1 = UNet()
model_path = r'D:\桌面\项目\Stratton\CU_Net_723\model_epoch_150.pth'
model1.load_state_dict(torch.load(model_path, weights_only=True))

model2 = UNet_3Plus_DeepSup(in_channels=1, n_classes=1)
model_path2 = r'D:\桌面\项目\Stratton\U_Net3+_718\model_epoch_150.pth'
model2.load_state_dict(torch.load(model_path2, weights_only=True))

# data = np.load(r"D:\桌面\项目\Stratton\data\dn50.npy")
# data = data[15:, 6:]  # 选择特定的切片

data = np.load(r"D:\桌面\项目\Stratton\data\Stratton3D_32bit.npy")
# data = data[50, 6:230, 126:1486]  
# data = data[50, 6:230, 1126:1494] 
data = data[200, 72:200, 1050:1418] #论文里的
# data = data[150, 50:178, 750:1118] 
data = data.T

data = normalize_data(data)
marmousi_clean = data
marmousi_noisy = data
marmousi_clean = torch.tensor(marmousi_clean, dtype=torch.float32)
marmousi_noisy = torch.tensor(marmousi_noisy, dtype=torch.float32)
input = marmousi_noisy.unsqueeze(0).unsqueeze(0)

model1.eval() 
with torch.no_grad():
    prediction = model1(input)
    prediction, *_ = prediction  
    initial = input.squeeze(0).squeeze(0).numpy()
    pure = marmousi_clean.squeeze(0).numpy()
    result = prediction.squeeze(0).squeeze(0).numpy()
    noise = initial - result

    # initial = initial[1000:, 150:]
    # result = result[1000:, 150:]
    # res = res[1000:, 150:]

    # initial = initial[1000:1368, :]
    # result = result[1000:1368, :]
    # res = res[1000:1368, :]

    # initial = initial[900:1200, 0:75]
    # result = result[900:1200, 0:75]
    # res = res[900:1200, 0:75]

    # np.save(r"D:\桌面\项目\Stratton\data\U_Net_data",initial)
    # np.save(r"D:\桌面\项目\Stratton\data\U_Net_denoised.npy",result)
    # np.save(r"D:\桌面\项目\Stratton\data\U_Net_noise.npy",noise)

    colors = [(0, 0, 0), (1, 1, 1), (1, 0, 0)]  # 黑 → 白 → 红
    black_white_red = LinearSegmentedColormap.from_list('black_white_red', colors, N=256)

    plt.figure()
    plt.title('含噪数据')
    # plt.imshow(initial, aspect='auto', cmap='seismic')
    plt.imshow(initial, cmap=black_white_red, aspect='auto', interpolation='nearest')
    plt.clim(0, 1)
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')

    plt.figure()
    plt.title('去噪结果')
    plt.imshow(result, cmap=black_white_red, aspect='auto', interpolation='nearest')
    # plt.imshow(result, aspect='auto', cmap='seismic')
    plt.clim(0, 1)
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')

    plt.figure()
    plt.title('去除的噪声')
    # plt.imshow(res, aspect='auto', cmap='seismic')
    plt.imshow(noise, cmap=black_white_red, aspect='auto', interpolation='nearest')
    plt.clim(-1, 1)
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')

    # plt.tight_layout()
    # plt.savefig(r"D:\桌面\U_Net3\开源数据集训练CODE\训练日志(15-55hz)\model_725.png", dpi = 2400)
    plt.show()

