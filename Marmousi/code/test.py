import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from U_Net import UNet
from UNet_3Plus import UNet_3Plus_DeepSup


MODEL_PATH = r'D:\桌面\随机噪声压制\Marmousi\conventional_model\model_epoch_10.pth'
CLEAN_PATH = r"D:\桌面\随机噪声压制\Marmousi\data\MAMI_clean.npy"
NOISY_PATH = r"D:\桌面\随机噪声压制\Marmousi\data\MAMI_noised(k=3).npy"
NOISE_SAVE_PATH = r"D:\桌面\随机噪声压制\noise111"
FIGURE_SIZE = None
IMAGE_EXTENT = (0, 512, 512, 0)
RESULT_CLIM = (0, 1)
NOISE_CLIM = (-1, 1)
CMAP = "seismic"

# model = UNet_3Plus_DeepSup(in_channels=1, n_classes=1)
# model_path = r'D:\桌面\项目\开源数据集训练CODE\U_Net3+_model_4_10\model_epoch_431.pth'
# # model_path = r'D:\桌面\项目\开源数据集训练CODE\U_Net3+_model_4_10\model_epoch_70.pth'
# # model_path = r'D:\桌面\项目\开源数据集训练CODE\U_Net3+_model_4_10\model_epoch_600.pth'
# model.load_state_dict(torch.load(model_path, weights_only=True))
# marmousi_noisy = np.load(r"D:\桌面\U_Net3\开源数据集训练CODE\data\noised(k=1).npy")
# np.save(r"D:\桌面\随机噪声压制\Marmousi\data\MAMI_U-Net_denoised(k=1)", result)
# plt.savefig(r"D:\桌面\U_Net3\开源数据集训练CODE\训练日志(15-55hz)\model_725.png", dpi = 2400)


def setup_font():
    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False


def calculate_snr(predictions, targets, use_variance=False):
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    noise = predictions - targets

    if use_variance:
        signal_power = np.var(targets)
        noise_power = np.var(noise)
    else:
        signal_power = np.mean(targets**2)
        noise_power = np.mean(noise**2)

    if noise_power == 0:
        return 0.0 if signal_power == 0 else np.inf

    return 10 * np.log10(signal_power / noise_power)


def load_model():
    model = UNet()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    return model


def predict(model, noisy, clean):
    input_tensor = noisy.unsqueeze(0).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]

    initial = input_tensor.squeeze(0).squeeze(0).numpy()
    pure = clean.numpy()
    result = prediction.squeeze(0).squeeze(0).numpy()
    noise = initial - result
    residual = pure - result
    return initial, pure, result, noise, residual


def plot_result(initial, pure, result, noise):
    if FIGURE_SIZE:
        plt.figure(figsize=FIGURE_SIZE)
    else:
        plt.figure()

    plt.subplot(2, 2, 1)
    plt.title('Noisy Data')
    plt.imshow(initial, aspect='equal', cmap=CMAP, extent=IMAGE_EXTENT)
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')

    plt.subplot(2, 2, 3)
    plt.title('Clean Label')
    plt.imshow(pure, aspect='equal', cmap=CMAP, extent=IMAGE_EXTENT)
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')

    plt.subplot(2, 2, 2)
    plt.title('Denoised Result')
    plt.imshow(result, aspect='equal', cmap=CMAP, extent=IMAGE_EXTENT)
    plt.clim(*RESULT_CLIM)
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')

    plt.subplot(2, 2, 4)
    plt.title('Residual')
    plt.imshow(noise, aspect='equal', cmap=CMAP, extent=IMAGE_EXTENT)
    plt.clim(*NOISE_CLIM)
    plt.colorbar(label='Amplitude (VDD)')
    plt.ylabel('Samples')

    plt.tight_layout()
    plt.show()

def main():
    setup_font()
    model = load_model()
    clean = torch.tensor(np.load(CLEAN_PATH), dtype=torch.float32)
    noisy = torch.tensor(np.load(NOISY_PATH), dtype=torch.float32)
    initial, pure, result, noise, _ = predict(model, noisy, clean)

    np.save(NOISE_SAVE_PATH, noise)
    snr_noised = calculate_snr(initial, pure)
    snr_denoised = calculate_snr(result, pure)
    print("before:", snr_noised, "\n", "after:", snr_denoised)
    plot_result(initial, pure, result, noise)


if __name__ == "__main__":
    main()
