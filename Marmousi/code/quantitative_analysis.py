import numpy as np
from matplotlib import rcParams
from skimage.metrics import structural_similarity as ssim

# 计算信噪比（SNR）
def calculate_snr(predictions, targets):
    noise = predictions - targets
    signal_power = np.mean(targets**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)

def mse(x, y):
    return np.mean((x - y) ** 2)

def psnr(x, y, max_value=1.0):
    """计算峰值信噪比（PSNR）"""
    mse_value = mse(x, y)
    if mse_value == 0:
        return float('inf')  # 完全相同的数据，PSNR 为无穷大
    return 10 * np.log10((max_value ** 2) / mse_value)

result = np.load(r"D:\桌面\项目\U_Net3+ 合成记录测试\开源数据集训练CODE\data\noised(k=3).npy")
pure = np.load(r"D:\桌面\项目\U_Net3+ 合成记录测试\开源数据集训练CODE\data\clean.npy")

data_range = result.max() - result.min()
Mse = mse(result, pure)
Ssim = ssim(result, pure, full = False, data_range=data_range)
Snr = calculate_snr(result, pure)
Psnr = psnr(result, pure)
print("MSE:", Mse, "\n","SSIM:", Ssim, "\n", "SNR:", Snr, "\n","PSNR:", Psnr)
