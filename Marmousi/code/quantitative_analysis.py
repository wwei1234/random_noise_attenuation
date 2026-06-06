import numpy as np
from skimage.metrics import structural_similarity as ssim


RESULT_PATH = r"D:\桌面\项目\U_Net3+ 合成记录测试\开源数据集训练CODE\data\noised(k=3).npy"
PURE_PATH = r"D:\桌面\项目\U_Net3+ 合成记录测试\开源数据集训练CODE\data\clean.npy"
PSNR_MAX_VALUE = 1.0


def calculate_snr(predictions, targets):
    noise = predictions - targets
    signal_power = np.mean(targets**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)


def mse(x, y):
    return np.mean((x - y) ** 2)


def psnr(x, y, max_value=PSNR_MAX_VALUE):
    mse_value = mse(x, y)
    if mse_value == 0:
        return float('inf')
    return 10 * np.log10((max_value ** 2) / mse_value)


def main():
    result = np.load(RESULT_PATH)
    pure = np.load(PURE_PATH)

    data_range = result.max() - result.min()
    mse_value = mse(result, pure)
    ssim_value = ssim(result, pure, full=False, data_range=data_range)
    snr_value = calculate_snr(result, pure)
    psnr_value = psnr(result, pure)
    print("MSE:", mse_value, "\n", "SSIM:", ssim_value, "\n", "SNR:", snr_value, "\n", "PSNR:", psnr_value)


if __name__ == "__main__":
    main()
