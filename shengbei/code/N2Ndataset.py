import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class N2NDatasetFromCrossline(Dataset):
    def __init__(self, data3d, patch_size=(128, 128), stride=(128, 128)):
        self.data = data3d  # shape: (641, 331, 7000)
        self.patch_size = patch_size
        self.stride = stride
        self.pairs = []
        self.prepare_patches()

    def prepare_patches(self):
        crosslines = self.data.shape[0]
        for i in range(crosslines - 1):  # 相邻 crossline 剖面
            # 提取并转置剖面为 (7000, 331)
            h, w = 7000, 331
            ph, pw = self.patch_size
            sh, sw = self.stride

            for y in range(0, h - ph + 1, sh):
                for x in range(0, w - pw + 1, sw):
                    self.pairs.append((i, y, x))

    def normalize_patch(self, patch):
        min_val = patch.min()
        max_val = patch.max()
        if max_val - min_val < 1e-6:
            return np.zeros_like(patch)  # 防止除以0
        return (patch - min_val) / (max_val - min_val)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, y, x = self.pairs[idx]

        input_slice = self.data[i].T
        target_slice = self.data[i + 1].T

        input_patch = input_slice[y:y+128, x:x+128]
        target_patch = target_slice[y:y+128, x:x+128]

        input_patch = self.normalize_patch(input_patch)
        target_patch = self.normalize_patch(target_patch)

        # === 添加通道维 ===
        return input_patch[np.newaxis, ...], target_patch[np.newaxis, ...]  # shape: (1, 128, 128)

    

# # 加载地震数据
# data = np.load(r"D:\桌面\项目\shengbei\data\shengbei_pstm_cb.npy")  # (641, 331, 7000)

# # 构建 Dataset
# dataset = N2NDatasetFromCrossline(data, patch_size=(128, 128), stride=(128, 128))

# index = np.random.randint(0, len(dataset))
# input_patch, target_patch = dataset[index]  # shape: (1, 128, 128)


# # 去掉通道维以便显示
# input_img = input_patch[0]   # shape: (128, 128)
# target_img = target_patch[0]# shape: (128, 128)

# # 绘图
# plt.figure(figsize=(8, 4))

# plt.subplot(1, 2, 1)
# plt.imshow(input_img, cmap='seismic', aspect='auto')
# plt.title(f"Input Patch #{index}")
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(target_img, cmap='seismic', aspect='auto')
# plt.title(f"Target Patch #{index}")
# plt.axis('off')

# plt.tight_layout()
# plt.show()