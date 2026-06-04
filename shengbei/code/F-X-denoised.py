import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, fftfreq
from matplotlib import rcParams
import os

# === 中文和负号支持 ===
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# === 参数设置 ===
dt = 0.001
dx = 3
f_max_display = 250
k_max_display = 0.5
threshold_ratio = 0.6  # F-K谱保留能量百分比（0~1）

# === 输入数据路径 ===
file_path = r"D:\桌面\项目\shengbei\data\shengbei_pstm_cb.npy"
save_dir = r"D:\桌面\项目\shengbei\可视化"
os.makedirs(save_dir, exist_ok=True)

# === 加载数据 ===
data = np.load(file_path)
nt, nx = data.shape
filename = os.path.splitext(os.path.basename(file_path))[0]

# === 去均值 + 加窗 ===
data = data - np.mean(data, axis=0)
window = np.hanning(nt)[:, None]
data_win = data * window

# === 进行 F-K 变换 ===
fk_data = fftshift(fft2(data_win))

# === 阈值处理（去噪） ===
# 构建物理域掩码：只保留低频+低波数部分
f = fftshift(fftfreq(nt, dt))[:, None]   # shape (nt, 1)
k = fftshift(fftfreq(nx, dx))[None, :]   # shape (1, nx)

f_max_keep = 120  # 只保留120Hz以下
k_max_keep = 0.10  # 波数范围限制

mask = (np.abs(f) < f_max_keep) & (np.abs(k) < k_max_keep)
fk_denoised = fk_data * mask


# === 反变换回时域 ===
data_denoised = np.real(ifft2(fftshift(fk_denoised)))
data_noise = data - data_denoised

# === 公共频率和波数轴 ===
f = fftshift(fftfreq(nt, dt))
k = fftshift(fftfreq(nx, dx))
f_mask = (f > 0) & (f < f_max_display)
k_mask = np.abs(k) < k_max_display

# === F-K谱计算函数 ===
def compute_fk_log(data_in):
    win = np.hanning(data_in.shape[0])[:, None]
    data_w = (data_in - np.mean(data_in, axis=0)) * win
    fk_spec = np.abs(fftshift(fft2(data_w)))**2
    return np.log10(fk_spec + 1e-10)

fk_log_original = compute_fk_log(data)
fk_log_denoised = compute_fk_log(data_denoised)
fk_log_noise = compute_fk_log(data_noise)

# === 设置统一色标范围 ===
all_fk = np.stack([fk_log_original, fk_log_denoised, fk_log_noise])
vmin, vmax = np.min(all_fk), np.max(all_fk)

# === 时域图绘制函数 ===
def plot_time_section(data_in, title, save_name):
    plt.figure(figsize=(6, 6))
    plt.imshow(data_in, cmap='seismic', aspect='auto')
    plt.title(title)
    plt.xlabel("道号")
    plt.ylabel("时间采样点")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{save_name}.png"), dpi=600)
    plt.show()

# === F-K谱图绘制函数 ===
def plot_fk_spectrum(fk_log, title, save_name):
    f_display = f[f_mask]
    k_display = k[k_mask]
    fk_display = fk_log[f_mask][:, k_mask]

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(k_display, f_display, fk_display, shading='auto',
                   cmap='jet', vmin=vmin, vmax=vmax)
    plt.xlabel("Wavenumber / m$^{-1}$")
    plt.ylabel("Frequency / Hz")
    plt.title(title)
    plt.colorbar(label="log10(Amplitude)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{save_name}_FK.png"), dpi=600)
    plt.show()

# === 绘图保存 ===
plot_time_section(data, "原始数据", f"{filename}_original")
plot_time_section(data_denoised, "去噪后数据", f"{filename}_denoised")
plot_time_section(data_noise, "去除的噪声", f"{filename}_noise")

plot_fk_spectrum(fk_log_original, "原始 F-K谱", f"{filename}_original")
plot_fk_spectrum(fk_log_denoised, "去噪后 F-K谱", f"{filename}_denoised")
plot_fk_spectrum(fk_log_noise, "噪声 F-K谱", f"{filename}_noise")

print("✅ 所有图像已保存。")
