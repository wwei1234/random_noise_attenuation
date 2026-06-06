# import numpy as np
#
# data = np.load(r'D:\桌面\随机噪声压制\U_Net_train_loss.npy')
# np.savetxt('U_Net3_loss_400.txt', data, fmt='%.6f')

import numpy as np


TXT_INPUT_PATH = r'D:\桌面\随机噪声压制\U_Net3_loss_400.txt'
NPY_OUTPUT_PATH = 'U_Net_loss.npy'

# import numpy as np
# import matplotlib.pyplot as plt
#
# data = np.load(r"D:\桌面\随机噪声压制\U_Net3+_train_losses.npy")
#
# winow_size = 200
# loss_smoothed = np.convolve(data, np.ones(winow_size)/winow_size, mode='valid')
#
# plt.figure(figsize=(10,5))
# plt.plot(loss_smoothed, color='r', linewidth=1.5)
# plt.xlabel("Epoch", fontsize=18, fontname='Times New Roman')
# plt.ylabel("Loss", fontsize=18, fontname='Times New Roman')
# plt.tick_params(axis='y', labelsize=16)
# plt.tick_params(axis='x', labelsize=16)
# plt.tight_layout()
# plt.show()
#
# np.save(r"D:\桌面\随机噪声压制\U_Net_train_loss.npy", loss_smoothed)


def convert_txt_to_npy(txt_path, npy_path):
    data = np.loadtxt(txt_path)
    np.save(npy_path, data)


def main():
    convert_txt_to_npy(TXT_INPUT_PATH, NPY_OUTPUT_PATH)


if __name__ == "__main__":
    main()
