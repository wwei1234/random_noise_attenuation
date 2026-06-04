import numpy as np
from scipy.linalg import svd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

def calculate_snr(original: np.ndarray, denoised: np.ndarray) -> float:
    """
    计算信噪比 (SNR, Signal-to-Noise Ratio)。
    
    参数
    ----
    original : np.ndarray
        原始信号。
    denoised : np.ndarray
        去噪后的信号。
    返回
    ----
    snr : float
        信噪比，单位为 dB。
    """
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - denoised) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def hankel_matrix(vec: np.ndarray, L: int) -> np.ndarray:
    """
    将长度为 M 的一维向量 vec 构造成 L×K 的 Hankel 矩阵，
    其中 K = M - L + 1。
    """
    M = vec.shape[0]
    K = M - L + 1
    H = np.zeros((L, K), dtype=vec.dtype)
    for i in range(L):
        H[i, :] = vec[i:i+K]
    return H

def inverse_hankel(H: np.ndarray) -> np.ndarray:
    """
    对 Hankel 矩阵 H 的反对角线元素做平均，
    恢复长度为 M = L + K - 1 的一维向量。
    """
    L, K = H.shape
    M = L + K - 1
    vec = np.zeros(M, dtype=complex)
    count = np.zeros(M, dtype=int)
    for i in range(L):
        for j in range(K):
            vec[i+j] += H[i, j]
            count[i+j] += 1
    return vec / count

def dmssa_denoise_columns(
    data2d: np.ndarray,
    L: int = None,
    P: int = 5,
    damp: float = 2.0
) -> np.ndarray:
    """
    对原始 shape=(512,512) 矩阵进行 DMSSA 去噪，
    每列是一条地震道（Time×Trace）：
    
    参数
    ----
    data2d : np.ndarray
        输入数据，shape = (M, N)，M 行为时间采样点数，N 列为道数。
    L : int, optional
        Hankel 矩阵行数，默认 L = M//2 + 1。
    P : int, optional
        保留的主模态数（奇异值个数），其余模态参与阻尼处理。
    damp : float, optional
        阻尼指数 D（式(9)），越大对小奇异值的抑制越强。
    
    返回
    ----
    denoised : np.ndarray
        去噪后数据，shape 同输入。
    """
    M, N = data2d.shape
    if L is None:
        L = M // 2 + 1

    # 1. 沿时间轴（行方向）做 FFT → 频域
    F = np.fft.fft(data2d, axis=0)
    F_denoised = np.zeros_like(F)

    # 2. 对每条道（每列）做 DMSSA
    for k in range(N):
        f = F[:, k]                  # 第 k 条道的频谱，长度 M
        H = hankel_matrix(f, L)      # 构造 Hankel 矩阵
        U, s, Vh = svd(H, full_matrices=False)

        # —— DMSSA 核心：构造阻尼后的奇异值 Σ'^(D) —— 
        # σ_p 为第 P 大奇异值
        sigma_p = s[P-1]
        # 阻尼因子 D_p(i) = (σ_i/σ_p)^d for i=1..P
        Dp = (s[:P] / sigma_p) ** damp
        # 构造阻尼奇异值向量
        s_damped = np.zeros_like(s)
        s_damped[:P] = s[:P] * Dp
        # 其余 σ_i (i>P) 保持为 0

        # 重构阻尼 Hankel 矩阵
        H_dmssa = (U * s_damped) @ Vh

        # 逆 Hankel → 重建频谱
        f_rec = inverse_hankel(H_dmssa)
        F_denoised[:, k] = f_rec

    # 3. 频域 → 时域
    denoised = np.fft.ifft(F_denoised, axis=0).real
    return denoised

# ===========================
if __name__ == "__main__":
    # 例：读取 NumPy 格式地震数据，shape = (512,512)，每列是一条地震道
    clean = np.load(r"D:\桌面\raw_data\data\clean.npy")  # 原始数据
    raw = np.load(r"D:\桌面\raw_data\data\noised(k=3)(15-55).npy")   
    # DMSSA 去噪
    denoised = dmssa_denoise_columns(raw, P=3, damp=1)
    # np.save("seismic_dmssa_denoised.npy", denoised)
    print("DMSSA 去噪完成，结果已保存为 seismic_dmssa_denoised.npy")
    # 计算并打印信噪比
    snr_value = calculate_snr(clean, denoised)
    print(f"Signal-to-Noise Ratio (SNR): {snr_value:.2f} dB")
    # 可视化去噪前后的对比
    plt.figure(figsize=(12, 6))
    # 原始数据
    plt.subplot(1, 2, 1)
    plt.imshow(raw, cmap='seismic', aspect='equal')
    plt.colorbar(label='Amplitude')
    plt.title('Original Data')
    plt.xlabel('Trace')
    plt.ylabel('Time')

    # 去噪后数据
    plt.subplot(1, 2, 2)
    plt.imshow(denoised, cmap='seismic', aspect='equal')
    plt.colorbar(label='Amplitude')
    plt.title('Denoised Data')
    plt.xlabel('Trace')
    plt.ylabel('Time')

    plt.tight_layout()
    plt.show()
