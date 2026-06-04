# import numpy as np

# # 读取 .npy 文件
# data = np.load(r'D:\桌面\随机噪声压制\U_Net_train_loss.npy')

# # 保存为 .txt 文件（保留浮点数精度）
# np.savetxt('U_Net3_loss_400.txt', data, fmt='%.6f')


import numpy as np

# 从 .txt 文件读取数据
data = np.loadtxt(r'D:\桌面\随机噪声压制\U_Net3_loss_400.txt')

# 保存为 .npy 文件
np.save('U_Net_loss.npy', data)


# import numpy as np
# import matplotlib.pyplot as plt

# data = np.load(r"D:\桌面\随机噪声压制\U_Net3+_train_losses.npy")

# winow_size = 200
# loss_smoothed = np.convolve(data, np.ones(winow_size)/winow_size, mode='valid')
    
# plt.figure(figsize=(10,5))
# plt.plot(loss_smoothed, color='r', linewidth=1.5)   
# plt.xlabel("Epoch", fontsize=18, fontname='Times New Roman')
# plt.ylabel("Loss", fontsize=18, fontname='Times New Roman') 
# plt.tick_params(axis='y', labelsize=16)
# plt.tick_params(axis='x', labelsize=16)
# plt.tight_layout()
# plt.show()

# np.save(r"D:\桌面\随机噪声压制\U_Net_train_loss.npy", loss_smoothed)