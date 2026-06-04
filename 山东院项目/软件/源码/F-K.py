import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, fftfreq
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import os

# 中文和负号显示设置
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# === 参数设置 ===
dt = 0.00025  # 时间采样间隔（秒）
dx = 3        # 道距（米）
f_max_display = 250  # 最大频率显示范围
k_max_display = 0.5  # 最大波数显示范围

# === 手动设置colorbar范围 ===
manual_symmetric_range = 6  # 指定范围为 [-10, 10]
vmin_final = -manual_symmetric_range
vmax_final = manual_symmetric_range

# === 自定义 colormap：深蓝 → 浅蓝（0）→ 红 ===
colors = ["#000074", "#00008B", "#0000B5", "#0000CE", "#0000DC", "#0000FF", "#E9EC0C", "#D44538","#E62210"]  # 深蓝 - 亮蓝 - 橙红
custom_cmap = LinearSegmentedColormap.from_list('blue-red', colors, N=256)

# === 创建归一化器，以 0 为中心 ===
norm = TwoSlopeNorm(vmin=vmin_final, vcenter=0, vmax=vmax_final)

# === 文件列表 ===
file_list = [
    r"D:\桌面\项目\shandong\data\shandongnoised.npy",
    r"D:\桌面\项目\shandong\data\conventional_U_Net_denoised.npy",
    r"D:\桌面\项目\shandong\data\conventional_U_Net_noise.npy",
    r"D:\桌面\项目\shandong\data\U_Net3+_4_10_67_denoised.npy",
    r"D:\桌面\项目\shandong\data\U_Net3+_4_10_67_noise.npy",
    r"D:\桌面\项目\shandong\data\MSSA_denoised.npy",
    r"D:\桌面\项目\shandong\data\MSSA_noise.npy",
]

# === 第一遍：计算所有谱的最大最小值 ===
print("正在计算所有文件的F-K谱...")
fk_logs = []
global_min, global_max = np.inf, -np.inf

for i, file_path in enumerate(file_list):
    print(f"处理文件 {i+1}/{len(file_list)}: {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        print(f"警告：文件不存在 {file_path}")
        continue

    data = np.load(file_path)
    nt, nx = data.shape
    print(f"  数据形状: {nt} x {nx}")

    # 去均值，加窗
    data = data - np.mean(data, axis=0)
    window = np.hanning(nt)[:, None]
    data *= window

    # 计算F-K谱
    fk_spectrum = np.abs(fftshift(fft2(data)))**2
    fk_log = np.log10(fk_spectrum + 1e-10)
    fk_logs.append(fk_log)

    current_min = np.min(fk_log)
    current_max = np.max(fk_log)
    global_min = min(global_min, current_min)
    global_max = max(global_max, current_max)

    print(f"  当前文件: min={current_min:.2f}, max={current_max:.2f}")

print(f"\n原始数据全局范围: min={global_min:.2f}, max={global_max:.2f}")
print(f"使用手动对称colorbar范围: [{vmin_final}, {vmax_final}]")

# === 第二遍：绘图 ===
print("\n开始绘制图像...")

for i, file_path in enumerate(file_list):
    if i >= len(fk_logs):
        continue

    filename = os.path.splitext(os.path.basename(file_path))[0]
    fk_log = fk_logs[i]

    print(f"绘制图像 {i+1}: {filename}")

    f = fftshift(fftfreq(fk_log.shape[0], dt))  # Hz
    k = fftshift(fftfreq(fk_log.shape[1], dx))  # m^-1

    f_mask = (f > 0) & (f < f_max_display)
    k_mask = np.abs(k) < k_max_display

    f_display = f[f_mask]
    k_display = k[k_mask]
    fk_display = fk_log[f_mask][:, k_mask]

    plt.figure(figsize=(10, 6))
    im = plt.pcolormesh(k_display, f_display, fk_display,
                        cmap=custom_cmap, norm=norm, shading='auto')

    plt.xlabel("Wavenumber / m$^{-1}$", fontsize=12)
    plt.ylabel("Frequency / Hz", fontsize=12)
    plt.title(f"F-K Spectrum", fontsize=14)

    cbar = plt.colorbar(im, label="log10(Amplitude)", format='%.1f')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # 保存图像
    save_dir = r"D:\桌面\项目\shandong\可视化"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}_FK.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"  已保存: {save_path}")
    plt.show()

print("\n所有图像处理完成！")
