# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from U_Net import UNet
# from matplotlib import rcParams
# from UNet_3Plus import UNet_3Plus_DeepSup
# from skimage.metrics import structural_similarity as ssim
# rcParams['font.sans-serif'] = ['SimHei']
# rcParams['axes.unicode_minus'] = False
#
# snr1 = np.load(r"D:\桌面\随机噪声压制\U_Net_snr.npy")
# snr2 = np.load(r"D:\桌面\随机噪声压制\U_Net3+_snr.npy")
# loss1 = np.load(r"D:\桌面\随机噪声压制\Marmousi\训练日志(UNet3+)(4_10)\train_losses.npy")
#
# plt.figure(figsize=(10, 5))
# plt.plot(snr1, color='g', linewidth=1.5)
# plt.plot(snr2, color='r', linewidth=1.5)
# plt.plot(loss1, color='r', linewidth=1.5)
# plt.xlabel("Epoch", fontsize=18, fontname='Times New Roman')
# plt.ylabel("SNR(dB)", fontsize=18, fontname='Times New Roman')
# plt.xticks(fontsize=16, fontname='Times New Roman')
# plt.yticks(fontsize=16, fontname='Times New Roman')
# plt.tight_layout()
# plt.legend(["SNR"] ,loc='upper left', fontsize=16, prop={'family': 'Times New Roman'})
#
# # plt.savefig(r"D:\桌面\项目\Stratton\矢量图(Times New Roman)\MAMI\snr.eps", format='eps', dpi=600, bbox_inches='tight')
# # plt.savefig(r"D:\桌面\项目\Stratton\矢量图(Times New Roman)\MAMI\snr.png", format='png', dpi=600, bbox_inches='tight')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


SNR_3PLUS_PATH = r"D:\桌面\随机噪声压制\U_Net3+_snr.npy"
SNR_UNET_PATH = r"D:\桌面\随机噪声压制\U_Net_snr.npy"
LOSS_UNET_PATH = r"D:\桌面\随机噪声压制\U_Net_loss.npy"
LOSS_3PLUS_PATH = r"D:\桌面\随机噪声压制\U_Net3+_loss.npy"
OUTPUT_PNG_PATH = r"D:\桌面\随机噪声压制\Loss&SNR.png"
OUTPUT_PDF_PATH = r"D:\桌面\随机噪声压制\Loss&SNR.pdf"

FIGURE_SIZE = (15, 10)
LINE_WIDTH = 1.5
AXIS_FONT_SIZE = 24
TICK_FONT_SIZE = 20
LEGEND_FONT_SIZE = 20
FONT_NAME = 'Times New Roman'
DPI = 600


def setup_font():
    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False


def set_tick_font(axis):
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontname(FONT_NAME)


def plot_loss_snr(loss1, loss2, snr1, snr2):
    plt.figure(figsize=FIGURE_SIZE)
    ax1 = plt.gca()
    l1, = ax1.plot(loss1, 'b--', linewidth=LINE_WIDTH, label='U-Net3+ Loss')
    l2, = ax1.plot(loss2, 'm--', linewidth=LINE_WIDTH, label='U-Net Loss')
    ax1.set_xlabel("Epoch", fontsize=AXIS_FONT_SIZE, fontname=FONT_NAME)
    ax1.set_ylabel("Loss", fontsize=AXIS_FONT_SIZE, fontname=FONT_NAME)
    ax1.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
    ax1.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
    set_tick_font(ax1)

    ax2 = ax1.twinx()
    l3, = ax2.plot(snr1, 'g-', linewidth=LINE_WIDTH, label='U-Net3+ SNR')
    l4, = ax2.plot(snr2, 'r-', linewidth=LINE_WIDTH, label='U-Net SNR')
    ax2.set_ylabel("SNR (dB)", fontsize=AXIS_FONT_SIZE, fontname=FONT_NAME)
    ax2.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
    for label in ax2.get_yticklabels():
        label.set_fontname(FONT_NAME)

    lines = [l1, l2, l3, l4]
    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines,
        labels,
        loc='upper right',
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        fancybox=True,
        shadow=False,
        prop={'family': FONT_NAME, 'size': LEGEND_FONT_SIZE},
        ncol=1,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_PATH, dpi=DPI, bbox_inches='tight', format='png')
    plt.savefig(OUTPUT_PDF_PATH, dpi=DPI, bbox_inches='tight', format='pdf')
    plt.show()


def main():
    setup_font()
    snr1 = np.load(SNR_3PLUS_PATH)
    snr2 = np.load(SNR_UNET_PATH)
    loss1 = np.load(LOSS_UNET_PATH)
    loss2 = np.load(LOSS_3PLUS_PATH)
    plot_loss_snr(loss1, loss2, snr1, snr2)


if __name__ == "__main__":
    main()
