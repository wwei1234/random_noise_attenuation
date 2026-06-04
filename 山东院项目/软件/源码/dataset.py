import os
import numpy as np
import torch
from torch.utils.data import Dataset
import segyio
import random
from typing import List, Tuple, Optional

class SeismicDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 patch_size: int = 32,
                 step_size: int = 16,
                 k: float = 0.1,
                 augment: bool = True,
                 shotnum: int = 0,
                 normalize: bool = True):
        """
        地震数据Dataset类
        
        Args:
            data_dir: 包含sgy文件的目录路径
            patch_size: 切片大小 (默认32x32)
            step_size: 切片步长 (默认16，控制重叠程度)
            noise_k: 噪声强度系数 k，噪声标准差 = 干净数据标准差 * k
            augment: 是否进行数据增强
            shotnum: 炮数，0表示自动计算
            normalize: 是否对数据进行归一化
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.step_size = step_size
        self.k = k
        self.augment = augment
        self.shotnum = shotnum
        self.normalize = normalize
        
        # 获取所有sgy文件
        self.sgy_files = []
        for file in os.listdir(data_dir):
            if file.endswith('.sgy'):
                self.sgy_files.append(os.path.join(data_dir, file))
        
        if not self.sgy_files:
            raise ValueError(f"在目录 {data_dir} 中没有找到.sgy文件")
        
        print(f"找到 {len(self.sgy_files)} 个sgy文件")
        
        # 预处理所有数据
        self.processed_data = []
        self.patches_info = []  # 存储每个patch的信息 (file_idx, shot_idx, start_row, start_col)
        
        self._preprocess_data()
        
    def read_segy(self, data_dir: str, shotnum: int = 0) -> np.ndarray:
        """读取segy文件"""
        with segyio.open(data_dir, 'r', ignore_geometry=True) as f:
            sourceX = f.attributes(segyio.TraceField.SourceX)[:]
            trace_num = len(sourceX)
            if shotnum:
                shot_num = shotnum 
            else:
                shot_num = len(set(sourceX))
            len_shot = trace_num // shot_num
            time = f.trace[0].shape[0]
            print(f'开始读取 {os.path.basename(data_dir)}, shots: {shot_num}, traces: {len_shot}, samples: {time}')
            
            data = np.zeros((shot_num, time, len_shot))
            for j in range(shot_num):
                data[j, :, :] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
        return data
    
    def agc_enhanced(self, data: np.ndarray, window_len: int = 30, gamma: float = 0.5) -> np.ndarray:
        """AGC增强处理"""
        agc_data = np.zeros_like(data)
        half_win = window_len // 2
        for i in range(data.shape[1]):
            trace = data[:, i]
            agc_trace = np.zeros_like(trace)
            for j in range(len(trace)):
                start = max(0, j - half_win)
                end = min(len(trace), j + half_win)
                window = trace[start:end]
                rms = np.sqrt(np.mean(window ** 2)) + 1e-8
                agc_trace[j] = trace[j] / rms

            agc_trace = np.sign(agc_trace) * (np.abs(agc_trace) ** gamma)
            agc_data[:, i] = agc_trace

        # 归一化到 [0, 1]
        min_val = agc_data.min()
        max_val = agc_data.max()
        agc_data = (agc_data - min_val) / (max_val - min_val + 1e-8)
        return agc_data
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """数据归一化"""
        min_val = data.min()
        max_val = data.max()
        return (data - min_val) / (max_val - min_val + 1e-8)
    
    def add_noise(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """添加随机噪声，噪声标准差 = 干净数据标准差 * k"""
        clean_std = np.std(data)
        noise_std = clean_std * self.k
        noise = np.random.randn(*data.shape) * noise_std
        return data + noise, noise
    
    def _preprocess_data(self):
        """预处理所有数据并生成patch信息"""
        print("开始预处理数据...")
        
        for file_idx, sgy_file in enumerate(self.sgy_files):
            # 读取并处理数据
            data = self.read_segy(sgy_file, self.shotnum)
            
            # 对每个shot进行处理
            processed_shots = []
            for shot_idx in range(data.shape[0]):
                shot_data = data[shot_idx]
                
                # 应用归一化处理
                if self.normalize:
                    shot_data = self.normalize_data(shot_data)
                
                processed_shots.append(shot_data)
                
                # 生成patch信息
                h, w = shot_data.shape
                for i in range(0, h - self.patch_size + 1, self.step_size):
                    for j in range(0, w - self.patch_size + 1, self.step_size):
                        self.patches_info.append((file_idx, shot_idx, i, j))
            
            self.processed_data.append(processed_shots)
        
        print(f"数据预处理完成，共生成 {len(self.patches_info)} 个patches")
    
    def apply_augmentation(self, patch: np.ndarray) -> np.ndarray:
        """应用数据增强"""
        if not self.augment:
            return patch.copy()  # 确保返回副本
        
        # 随机选择增强方法
        augment_type = random.randint(0, 4)  # 0: 不变, 1: 90°, 2: 180°, 3: 270°, 4: 翻转
        
        if augment_type == 0:
            return patch.copy()
        elif augment_type == 1:  # 90度旋转
            return np.rot90(patch, k=1).copy()
        elif augment_type == 2:  # 180度旋转
            return np.rot90(patch, k=2).copy()
        elif augment_type == 3:  # 270度旋转
            return np.rot90(patch, k=3).copy()
        elif augment_type == 4:  # 水平翻转
            return np.fliplr(patch).copy()
        
    def __len__(self):
        return len(self.patches_info)
    
    def __getitem__(self, idx):
        file_idx, shot_idx, start_row, start_col = self.patches_info[idx]
        
        # 提取patch
        shot_data = self.processed_data[file_idx][shot_idx]
        clean_patch = shot_data[start_row:start_row + self.patch_size, 
                               start_col:start_col + self.patch_size].copy()
        
        # 应用数据增强
        clean_patch = self.apply_augmentation(clean_patch)
        
        # 添加噪声
        noisy_patch, noise = self.add_noise(clean_patch)
        
        # 确保所有数组都是连续的，然后转换为tensor
        clean_patch = np.ascontiguousarray(clean_patch)
        noisy_patch = np.ascontiguousarray(noisy_patch)
        noise = np.ascontiguousarray(noise)
        
        clean_patch = torch.FloatTensor(clean_patch).unsqueeze(0)  # 添加channel维度
        noisy_patch = torch.FloatTensor(noisy_patch).unsqueeze(0)
        noise = torch.FloatTensor(noise).unsqueeze(0)
        
        # 返回训练代码期望的格式: (input, target)
        return noisy_patch, clean_patch


