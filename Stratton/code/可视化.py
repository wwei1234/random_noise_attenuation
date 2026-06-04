import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

data = np.load(r"D:\桌面\项目\Stratton\data\dn50.npy")
denoised = np.load(r"D:\桌面\项目\Stratton\data\d50.npy")
noise = np.load(r"D:\桌面\项目\Stratton\data\d50_noise.npy")

data = normalize_data(data)
denoised = normalize_data(denoised)

colors = [(0, 0, 0), (1, 1, 1), (1, 0, 0)]  # 黑 → 白 → 红
black_white_red = LinearSegmentedColormap.from_list('black_white_red', colors, N=256)

plt.figure()
plt.title('原始含噪数据')
plt.imshow(data, cmap=black_white_red, aspect='auto', interpolation='nearest')
# plt.clim(0, 1)
plt.colorbar(label='Amplitude')
plt.ylabel('Samples')

plt.figure()
plt.title('去噪结果')
plt.imshow(denoised, cmap=black_white_red, aspect='auto', interpolation='nearest')
# plt.clim(0, 1)
plt.colorbar(label='Amplitude')
plt.ylabel('Samples')

plt.figure()
plt.title('去除的噪声')
plt.imshow(noise, cmap=black_white_red, aspect='auto', interpolation='nearest')
# plt.clim(-1, 1)
plt.colorbar(label='Amplitude')
plt.ylabel('Samples')
plt.show()  