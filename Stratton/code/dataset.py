# import numpy as np
# from torch.utils.data import Dataset
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt
# import scipy.signal as signal
# from matplotlib import rcParams
# rcParams['font.sans-serif'] = ['SimHei']
# rcParams['axes.unicode_minus'] = False

# def filter_data(data, lowcut, highcut):
#     """带通滤波函数"""
#     dt = 0.002
#     nyquist = 0.5 / dt
#     low = lowcut / nyquist
#     high = highcut / nyquist
    
#     # 确保频率范围在0-1之间
#     low = max(0.001, min(low, 0.99))
#     high = max(0.002, min(high, 0.99))
    
#     b, a = signal.butter(4, [low, high], btype='band', analog=False)
    
#     default_padlen = 3 * (max(len(b), len(a)) - 1)
#     if data.shape[0] <= default_padlen:
#         padlen = min(data.shape[0] - 1, 10)
#         filtered_data = signal.filtfilt(b, a, data, axis=0, padlen=padlen)
#     else:
#         filtered_data = signal.filtfilt(b, a, data, axis=0)
    
#     return filtered_data

# class DualBandNoiseDataset(Dataset):
#     def __init__(self, data3d, patch_size=(128, 128), stride=(64, 64), 
#                  include_border=True, noise_params1=None, noise_params2=None,
#                  depth_start=125, depth_end=800):

#         # 提取深度范围并转置维度 [crossline, depth, inline]
#         self.data = data3d[:, :, depth_start:depth_end].transpose(0, 2, 1)
#         self.patch_size = patch_size
#         self.stride = stride
#         self.include_border = include_border
        
#         self.noise_params1 = noise_params1 
#         self.noise_params2 = noise_params2 
#         self.patches = []  # 存储每个patch的信息
#         self.prepare_patches()

#     def prepare_patches(self):
#         """准备所有可能的patch位置"""
#         crosslines, depths, inlines = self.data.shape
        
#         depth_positions = list(range(0, depths - self.patch_size[0] + 1, self.stride[0]))
#         inline_positions = list(range(0, inlines - self.patch_size[1] + 1, self.stride[1]))
        
#         if self.include_border:
#             if depth_positions and depth_positions[-1] != depths - self.patch_size[0]:
#                 depth_positions.append(depths - self.patch_size[0])
#             if inline_positions and inline_positions[-1] != inlines - self.patch_size[1]:
#                 inline_positions.append(inlines - self.patch_size[1])
        
#         for cl in range(crosslines):
#             for depth in depth_positions:
#                 for inline in inline_positions:
#                     self.patches.append((cl, depth, inline))
    
#     def normalize_patch(self, patch):
#         """归一化patch到[0,1]范围"""
#         min_val = patch.min()
#         max_val = patch.max()
#         if max_val - min_val < 1e-6:
#             return np.zeros_like(patch)
#         return (patch - min_val) / (max_val - min_val)

#     def __len__(self):
#         return len(self.patches)

#     def __getitem__(self, idx):
#         cl, depth, inline = self.patches[idx]
#         clean_patch = self.data[cl, depth:depth+self.patch_size[0], inline:inline+self.patch_size[1]]
        
#         # 归一化处理
#         clean_patch_norm = self.normalize_patch(clean_patch)
#         base_std = np.std(clean_patch_norm)
        
#         # 生成带通噪声
#         k1 = np.random.uniform(*self.noise_params1['k_range'])
#         noise1 = np.random.randn(*clean_patch_norm.shape) * (base_std * k1)
#         noise1 = filter_data(noise1, self.noise_params1['lowcut'], self.noise_params1['highcut'])
        
#         k2 = np.random.uniform(*self.noise_params2['k_range'])
#         noise2 = np.random.randn(*clean_patch_norm.shape) * (base_std * k2)
#         noise2 = filter_data(noise2, self.noise_params2['lowcut'], self.noise_params2['highcut'])
        
#         # 合成噪声数据
#         noisy_patch = clean_patch_norm + noise1 + noise2

#         # 关键修复：确保内存连续性
#         clean_patch_norm = np.ascontiguousarray(clean_patch_norm)
#         noise1 = np.ascontiguousarray(noise1)
#         noise2 = np.ascontiguousarray(noise2)
#         noisy_patch = np.ascontiguousarray(noisy_patch)
        
#         return (
#             noisy_patch[np.newaxis, ...],
#             clean_patch_norm[np.newaxis, ...],
#         )

# # 使用示例
# if __name__ == "__main__":
#     # 加载实际数据
#     # data = np.load(r"D:\桌面\项目\Stratton\data\Stack_final.npy")
#     # data = data[:, 100:900, 0:600]

#     data = np.load(r"D:\桌面\项目\Stratton\data\Stratton3D_32bit.npy")

#     # 噪声参数配置（基于归一化后标准差）
#     noise_params1 = {
#         'k_range': (0.4, 0.8),   # 低频噪声强度范围
#         'lowcut': 25,            # 低频截止(Hz)
#         'highcut': 85            # 高频截止(Hz)
#     }
    
#     noise_params2 = {
#         'k_range': (0.6, 1.2),   # 高频噪声强度范围
#         'lowcut': 85,            # 低频截止(Hz)
#         'highcut': 200           # 高频截止(Hz)
#     }
    
#     # 创建数据集
#     dataset = DualBandNoiseDataset(
#         data,
#         patch_size=(128, 128),
#         stride=(128, 128),
#         include_border=True,
#         noise_params1=noise_params1,
#         noise_params2=noise_params2,
#         depth_start= 126,
#         depth_end = 800
#     )
    
#     print(f"数据集大小 (patch数量): {len(dataset)}")
#     print(f"转置后数据形状 (crossline, depth, inline): {dataset.data.shape}")
    
#     # 获取一个样本
#     noisy_patch, clean_patch= dataset[20]
#     print(f"噪声输入形状: {noisy_patch.shape}")
#     print(f"干净目标形状: {clean_patch.shape}")
    
#     # 计算归一化后标准差（用于噪声强度计算）
#     base_std = np.std(clean_patch)
#     print(f"归一化后patch标准差: {base_std:.4f}")
    
#     # 计算实际噪声强度（基于当前k值）
#     k1 = np.mean(noise_params1['k_range'])
#     actual_std1 = base_std * k1
#     k2 = np.mean(noise_params2['k_range'])
#     actual_std2 = base_std * k2
    
#     # 可视化样本
#     plt.figure(figsize=(15, 10))
    
#     # 1. 归一化干净数据
#     plt.subplot(1, 2, 1)
#     plt.imshow(np.squeeze(clean_patch), cmap='seismic', aspect='auto', vmin=0, vmax=1)
#     plt.title("归一化干净数据")
#     plt.colorbar()
#     plt.xlabel("Inline方向")
#     plt.ylabel("深度方向")
    
#     # 2. 加噪数据
#     plt.subplot(1, 2, 2)
#     plt.imshow(np.squeeze(noisy_patch), cmap='seismic', aspect='auto', vmin=0, vmax=1)
#     plt.title(f"加噪数据 (总噪声强度: {actual_std1+actual_std2:.4f})")
#     plt.colorbar()
#     plt.xlabel("Inline方向")
#     plt.ylabel("深度方向")

#     plt.tight_layout()
#     plt.show()
    
#     # 打印噪声参数
#     print("\n噪声参数1 (低频):")
#     print(f"强度范围: {noise_params1['k_range']}")
#     print(f"频率范围: {noise_params1['lowcut']}-{noise_params1['highcut']}Hz")
    
#     print("\n噪声参数2 (高频):")
#     print(f"强度范围: {noise_params2['k_range']}")
#     print(f"频率范围: {noise_params2['lowcut']}-{noise_params2['highcut']}Hz")




import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import scipy.signal as signal
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

def filter_data(data, lowcut, highcut):
    """带通滤波函数"""
    dt = 0.002
    nyquist = 0.5 / dt
    low = lowcut / nyquist
    high = highcut / nyquist

    low = max(0.001, min(low, 0.99))
    high = max(0.002, min(high, 0.99))

    b, a = signal.butter(4, [low, high], btype='band', analog=False)

    default_padlen = 3 * (max(len(b), len(a)) - 1)
    if data.shape[0] <= default_padlen:
        padlen = min(data.shape[0] - 1, 10)
        filtered_data = signal.filtfilt(b, a, data, axis=0, padlen=padlen)
    else:
        filtered_data = signal.filtfilt(b, a, data, axis=0)

    return filtered_data

def generate_coherent_noise(patch_shape, std, max_shift=10):
    """
    生成与反射轴一致的结构性噪声
    patch_shape: (depth, inline)
    std: 归一化后 patch 的标准差
    """
    depth, inline = patch_shape
    coherent = np.zeros((depth, inline))

    # 每个 inline trace 做一个微小的“相位扰动”模拟走时扰动
    shifts = np.random.randint(-max_shift, max_shift + 1, size=inline)
    for i in range(inline):
        idx = np.clip(np.arange(depth) - shifts[i], 0, depth - 1)
        trace = np.sin(2 * np.pi * 0.01 * idx)  # 频率可调，近似模拟反射扰动形态
        coherent[:, i] = trace

    coherent_std = np.std(coherent)
    if coherent_std > 1e-6:
        coherent = (coherent / coherent_std) * std * 0.6  # 结构性噪声强度因子
    else:
        coherent = np.zeros_like(coherent)
    return coherent

class DualBandNoiseDataset(Dataset):
    def __init__(self, data3d, patch_size=(128, 128), stride=(64, 64),
                 include_border=True, noise_params1=None, noise_params2=None,
                 depth_start=125, depth_end=800, use_coherent_noise=True, max = 200):  # 加入开关

        self.data = data3d[:, :, depth_start:depth_end].transpose(0, 2, 1)
        self.patch_size = patch_size
        self.stride = stride
        self.include_border = include_border

        self.noise_params1 = noise_params1
        self.noise_params2 = noise_params2
        self.use_coherent_noise = use_coherent_noise  # 保存开关

        self.patches = []
        self.prepare_patches()
        self.max_shift = max  # 最大相位扰动范围

    def prepare_patches(self):
        crosslines, depths, inlines = self.data.shape
        depth_positions = list(range(0, depths - self.patch_size[0] + 1, self.stride[0]))
        inline_positions = list(range(0, inlines - self.patch_size[1] + 1, self.stride[1]))

        if self.include_border:
            if depth_positions and depth_positions[-1] != depths - self.patch_size[0]:
                depth_positions.append(depths - self.patch_size[0])
            if inline_positions and inline_positions[-1] != inlines - self.patch_size[1]:
                inline_positions.append(inlines - self.patch_size[1])

        for cl in range(crosslines):
            for depth in depth_positions:
                for inline in inline_positions:
                    self.patches.append((cl, depth, inline))

    def normalize_patch(self, patch):
        min_val = patch.min()
        max_val = patch.max()
        if max_val - min_val < 1e-6:
            return np.zeros_like(patch)
        return (patch - min_val) / (max_val - min_val)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        cl, depth, inline = self.patches[idx]
        clean_patch = self.data[cl, depth:depth+self.patch_size[0], inline:inline+self.patch_size[1]]

        clean_patch_norm = self.normalize_patch(clean_patch)
        base_std = np.std(clean_patch_norm)

        if self.use_coherent_noise:
            coherent_noise = generate_coherent_noise(clean_patch_norm.shape, base_std, max_shift=self.max_shift)
        else:
            coherent_noise = np.zeros_like(clean_patch_norm)

        k1 = np.random.uniform(*self.noise_params1['k_range'])
        noise1 = np.random.randn(*clean_patch_norm.shape) * (base_std * k1)
        noise1 = filter_data(noise1, self.noise_params1['lowcut'], self.noise_params1['highcut'])

        k2 = np.random.uniform(*self.noise_params2['k_range'])
        noise2 = np.random.randn(*clean_patch_norm.shape) * (base_std * k2)
        noise2 = filter_data(noise2, self.noise_params2['lowcut'], self.noise_params2['highcut'])

        noisy_patch = clean_patch_norm + coherent_noise + noise1 + noise2

        clean_patch_norm = np.ascontiguousarray(clean_patch_norm)
        noisy_patch = np.ascontiguousarray(noisy_patch)

        return (
            noisy_patch[np.newaxis, ...],
            clean_patch_norm[np.newaxis, ...],
        )

if __name__ == "__main__":
    # data = np.load(r"D:\桌面\项目\Stratton\data\Stack_final.npy")
    # data = data[:, 150:900, 0:500]

    data = np.load(r"D:\桌面\项目\Stratton\data\Stratton3D_32bit.npy")

    noise_params1 = {
        'k_range': (0.8, 1.2),   # 低频噪声强度范围
        'lowcut': 25,            # 低频截止(Hz)
        'highcut': 85            # 高频截止(Hz)
    }

    noise_params2 = {
        'k_range': (1.2, 1.8),   # 高频噪声强度范围
        'lowcut': 85,            # 低频截止(Hz)
        'highcut': 200           # 高频截止(Hz)
    }

    dataset = DualBandNoiseDataset(
        data,
        patch_size=(128, 128),
        stride=(128, 128),
        include_border=True,
        noise_params1=noise_params1,
        noise_params2=noise_params2,
        depth_start=126,
        depth_end=800,
        use_coherent_noise=True
    )

    print(f"数据集大小 (patch数量): {len(dataset)}")
    print(f"数据维度 (crossline, depth, inline): {dataset.data.shape}")

    noisy_patch, clean_patch = dataset[200]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(clean_patch), cmap='seismic', aspect='auto', vmin=0, vmax=1)
    plt.title("归一化干净数据")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(noisy_patch), cmap='seismic', aspect='auto', vmin=0, vmax=1)
    plt.title("含结构+带通噪声数据")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
