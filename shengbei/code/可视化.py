import numpy as np
import matplotlib.pyplot as plt
import torch
from U_Net import UNet 
from matplotlib import rcParams
from UNet_3Plus import UNet_3Plus_DeepSup  # 请确保你的模型类是正确导入的
from skimage.metrics import structural_similarity as ssim
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

data = np.load(r"D:\桌面\项目\shengbei\data\shengbei_pstm_cb_local.npy")
denoised = np.load(r"D:\桌面\项目\shengbei\data\U-Net-711-200-denoised.npy")
noise = np.load(r"D:\桌面\项目\shengbei\data\U-Net-711-200-noise.npy")
data = data[50: 450, 0: 400]
denoised = denoised[50: 450, 0: 400]
noise = noise[50: 450, 0: 400]

plt.figure()
plt.title('原始含噪数据')
plt.imshow(data, aspect='equal', cmap='seismic', extent=(0, 400, 400, 0))
# plt.clim(0, 1)
plt.colorbar(label='Amplitude')
plt.ylabel('Samples')

plt.figure()
plt.title('去噪结果')
plt.imshow(denoised, aspect='equal', cmap='seismic', extent=(0, 400, 400, 0))
# plt.clim(0, 1)
plt.colorbar(label='Amplitude')
plt.ylabel('Samples')

plt.figure()
plt.title('去除的噪声')
plt.imshow(noise, aspect='equal', cmap='seismic', extent=(0, 400, 400, 0))
plt.clim(-1, 1)
plt.colorbar(label='Amplitude')
plt.ylabel('Samples')
plt.show()  