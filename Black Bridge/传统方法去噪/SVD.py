import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# SVD 去噪函数
def svd_denoise(data, k):
    U, S, Vh = np.linalg.svd(data, full_matrices=False)
    S_k = np.diag(S[:k]) @ Vh[:k, :]
    denoised_data = U[:, :k] @ S_k
    return denoised_data

# SNR 计算函数
def calculate_snr(predictions, targets):
    noise = predictions - targets
    signal_power = np.mean(targets**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)

# 加载数据
marmousi_clean = np.load(r"D:\桌面\随机噪声压制\Black Bridge\data\clean.npy")
marmousi_noisy = np.load(r"D:\桌面\随机噪声压制\Black Bridge\data\noised(k=2)(15-55).npy")

# 搜索最优秩
max_rank = min(marmousi_noisy.shape)
snr_list = []
k_list = list(range(1, max_rank + 1, 1))  # 逐步测试每一个秩

for k in k_list:
    denoised = svd_denoise(marmousi_noisy, k)
    snr = calculate_snr(denoised, marmousi_clean)
    snr_list.append(snr)

# 找到最优的k
best_k = k_list[np.argmax(snr_list)]
best_snr = snr_list[np.argmax(snr_list)]

print(f"最佳保留秩 k = {best_k}, 最佳 SNR = {best_snr:.2f} dB")

# 使用最优k去噪
best_denoised = svd_denoise(marmousi_noisy, best_k)

# np.save(r"D:\桌面\项目\raw_data\data\svd_denoised_(k=1)", best_denoised)

# 绘制 SNR 曲线
plt.figure(figsize=(6, 4))
plt.plot(k_list, snr_list, label='SNR vs k')
plt.axvline(best_k, color='r', linestyle='--', label=f'Best k = {best_k}')
plt.xlabel('保留奇异值个数 k')
plt.ylabel('SNR (dB)')
plt.title('SVD 去噪 - SNR 曲线')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 可视化对比图
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(marmousi_clean, cmap='seismic', aspect='equal')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(marmousi_noisy, cmap='seismic', aspect='equal')
plt.title('Noisy Image')

plt.subplot(1, 3, 3)
plt.imshow(best_denoised, cmap='seismic', aspect='equal')
plt.title(f'Denoised (k={best_k})')
plt.tight_layout()
plt.show()
