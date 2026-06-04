if __name__ == '__main__':
    import numpy as np
    from scipy.fft import rfft, irfft
    from scipy.linalg import eigh
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import rcParams
    rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
    rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

    def normalize_patch(patch):
        min_val = patch.min()
        max_val = patch.max()
        if max_val - min_val < 1e-6:
            return np.zeros_like(patch)
        return (patch - min_val) / (max_val - min_val)

    def fx_sgd_denoise(data, dx, rank=None):

        # 1. 傅里叶变换到频率域
        freq_data = rfft(data, axis=0)
        nf, nx = freq_data.shape
        denoised_freq = np.zeros_like(freq_data, dtype=complex)
        
        for f_idx in range(nf):
            freq_slice = freq_data[f_idx, :]
            
            # 2. 构建Hankel轨迹矩阵
            L = nx - dx + 1
            H = np.zeros((dx, L), dtype=complex)
            for i in range(dx):
                H[i, :] = freq_slice[i:i+L]
            
            # 3. 计算协方差矩阵 A = H^T H
            A = H.T.conj() @ H  # A维度: L×L
            
            # 4. 构建Hamiltonian矩阵 M = [[A, 0], [0, -A^T]]
            M_upper = np.hstack([A, np.zeros((L, L))])
            M_lower = np.hstack([np.zeros((L, L)), -A.T])
            M = np.vstack([M_upper, M_lower])  # M维度: 2L×2L
            
            # 5. 计算N = M^2
            N = M @ M
            
            # 6. 特征分解
            eigvals, eigvecs = eigh(N)  # eigvecs维度: 2L×2L
            
            # 7. 自适应秩选择
            if rank is None:
                # 简化的自适应秩选择（实际应使用辛几何熵）
                svd_vals = np.sqrt(np.abs(eigvals))
                cumulative = np.cumsum(svd_vals[::-1])
                total = cumulative[-1]
                rank = np.argmax(cumulative > 0.95*total) + 1
            
            Q = eigvecs[:, -rank:]
            
            # 重构公式: \widetilde{H} = H @ (V @ V^H)
            # V是A的特征向量 (来自Q的子矩阵)
            V = Q[:L, :]  # 取前L行 (L×m)
            H_recon = H @ (V @ V.T.conj())  # (dx×L) @ (L×L) = dx×L
            
            # 9. 对角平均还原数据
            denoised_slice = np.zeros(nx, dtype=complex)
            count = np.zeros(nx)
            for i in range(dx):
                for j in range(L):
                    idx = i + j
                    denoised_slice[idx] += H_recon[i, j]
                    count[idx] += 1
            denoised_freq[f_idx, :] = denoised_slice / np.maximum(count, 1)
        
        return irfft(denoised_freq, axis=0, n=data.shape[0])

    # 使用示例
    if __name__ == "__main__":

        data = np.load(r"noised.npy_path")
        dx = 400 # x方向嵌入维度
        rank = 40 # 固定秩参数
        
        # 执行降噪
        denoised = fx_sgd_denoise(data, dx, rank)
        colors = [(0, 0, 0), (1, 1, 1), (1, 0, 0)]  # 黑 → 白 → 红
        black_white_red = LinearSegmentedColormap.from_list('black_white_red', colors, N=256)

        # np.save(r"D:\桌面\项目\Stratton\方法对比\BLACK_CONVENTIONAL_DENOISED", denoised)
        # np.save(r"D:\桌面\项目\Stratton\方法对比\BLACK_CONVENTIONAL_NOISE", data-denoised)

        # # 绘图对比
        plt.figure()
        # plt.imshow(data, cmap=black_white_red, aspect='auto', interpolation='nearest')
        plt.imshow(data, "seismic", aspect='auto')
        plt.title('带噪数据')
        plt.clim(0,1)
        plt.colorbar()

        plt.figure()
        plt.imshow(denoised, "seismic", aspect='auto')
        # plt.imshow(denoised, cmap=black_white_red, aspect='auto', interpolation='nearest')
        plt.title('SVD去噪后')
        plt.clim(0,1)
        plt.colorbar()

        plt.figure()
        # plt.imshow(data-denoised, cmap=black_white_red, aspect='auto', interpolation='nearest')
        plt.imshow(data-denoised, "seismic", aspect='auto')
        plt.title('去除的噪声')
        plt.clim(-1,1)
        plt.colorbar()
        plt.show()