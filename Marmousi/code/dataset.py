import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NumpyDataset(Dataset):
    def __init__(self, samples_dir, labels_dir):
        """
        初始化Dataset
        :param samples_dir: 样本文件夹路径
        :param labels_dir: 标签文件夹路径
        """
        self.samples_dir = samples_dir
        self.labels_dir = labels_dir
        self.samples = sorted(os.listdir(samples_dir))  # 假设文件名已经按顺序排列
        self.labels = sorted(os.listdir(labels_dir))

    def __len__(self):
        """返回数据集的大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引获取样本和标签
        :param idx: 索引
        :return: 样本和标签
        """
        sample_path = os.path.join(self.samples_dir, self.samples[idx])
        label_path = os.path.join(self.labels_dir, self.labels[idx])

        sample = np.load(sample_path)  # 读取npy文件
        label = np.load(label_path)

        # 确保样本和标签具有通道维度 (增加一个维度)
        sample = np.expand_dims(sample, axis=0)  # 将样本调整为 [1, 128, 128]
        label = np.expand_dims(label, axis=0)  # 将标签调整为 [1, 128, 128]

        # 转换为Tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return sample, label







