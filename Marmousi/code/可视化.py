# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from U_Net import UNet 
# from matplotlib import rcParams
# from UNet_3Plus import UNet_3Plus_DeepSup  # 请确保你的模型类是正确导入的
# from skimage.metrics import structural_similarity as ssim
# rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
# rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

# snr1 = np.load(r"D:\桌面\随机噪声压制\U_Net_snr.npy")
# snr2 = np.load(r"D:\桌面\随机噪声压制\U_Net3+_snr.npy")
# loss1 = np.load(r"D:\桌面\随机噪声压制\Marmousi\训练日志(UNet3+)(4_10)\train_losses.npy")

# plt.figure(figsize=(10, 5))
# plt.plot(snr1, color='g', linewidth=1.5)
# plt.plot(snr2, color='r', linewidth=1.5)
# plt.plot(loss1, color='r', linewidth=1.5)
# plt.xlabel("Epoch", fontsize=18, fontname='Times New Roman')
# plt.ylabel("SNR(dB)", fontsize=18, fontname='Times New Roman')
# # 设置坐标刻度字体
# plt.xticks(fontsize=16, fontname='Times New Roman')
# plt.yticks(fontsize=16, fontname='Times New Roman')
# plt.tight_layout()
# plt.legend(["SNR"] ,loc='upper left', fontsize=16, prop={'family': 'Times New Roman'})

# # plt.savefig(r"D:\桌面\项目\Stratton\矢量图(Times New Roman)\MAMI\snr.eps", format='eps', dpi=600, bbox_inches='tight')
# # plt.savefig(r"D:\桌面\项目\Stratton\矢量图(Times New Roman)\MAMI\snr.png", format='png', dpi=600, bbox_inches='tight')
# plt.show()  

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 载入数据
snr1 = np.load(r"D:\桌面\随机噪声压制\U_Net3+_snr.npy")
snr2 = np.load(r"D:\桌面\随机噪声压制\U_Net_snr.npy")
loss1 = np.load(r"D:\桌面\随机噪声压制\U_Net_loss.npy")
loss2 = np.load(r"D:\桌面\随机噪声压制\U_Net3+_loss.npy")

plt.figure(figsize=(15,10))

# 左轴
ax1 = plt.gca()
l1, = ax1.plot(loss1, 'b--', linewidth=1.5, label='U-Net3+ Loss')
l2, = ax1.plot(loss2, 'm--', linewidth=1.5, label='U-Net Loss')
ax1.set_xlabel("Epoch", fontsize=24, fontname='Times New Roman')
ax1.set_ylabel("Loss", fontsize=24, fontname='Times New Roman')
ax1.tick_params(axis='y', labelsize=20)
ax1.tick_params(axis='x', labelsize=20)

# ✅ 设置左轴刻度字体
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontname('Times New Roman')

# 右轴
ax2 = ax1.twinx()
l3, = ax2.plot(snr1, 'g-', linewidth=1.5, label='U-Net3+ SNR')
l4, = ax2.plot(snr2, 'r-', linewidth=1.5, label='U-Net SNR')
ax2.set_ylabel("SNR (dB)", fontsize=24, fontname='Times New Roman')
ax2.tick_params(axis='y', labelsize=20)

# ✅ 设置右轴刻度字体
for label in ax2.get_yticklabels():
    label.set_fontname('Times New Roman')

# 合并图例
lines = [l1, l2, l3, l4]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels,
            loc='upper right',
            bbox_to_anchor=(1, 0.5),
            frameon=True,
            fancybox=True,
            shadow=False,
            prop={'family':'Times New Roman', 'size':20},
            ncol=1)
 
# 保存图片
plt.savefig(r"D:\桌面\随机噪声压制\Loss&SNR.png", dpi=600, bbox_inches='tight', format='png')
plt.savefig(r"D:\桌面\随机噪声压制\Loss&SNR.pdf", dpi=600, bbox_inches='tight', format='pdf')

plt.tight_layout()
plt.show()
