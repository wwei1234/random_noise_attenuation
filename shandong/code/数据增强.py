import numpy as np
import os

# 定义数据增强函数
def flip_left_right(data):
    """
    左右翻转数据
    :param data: 输入的地震记录数据
    :return: 左右翻转后的数据
    """
    return np.flip(data, axis=1)  # 沿列（水平）轴翻转

def flip_up_down(data):
    """
    上下翻转数据
    :param data: 输入的地震记录数据
    :return: 上下翻转后的数据
    """
    return np.flip(data, axis=0)  # 沿行（垂直）轴翻转

def rotate_180(data):
    """
    旋转180度
    :param data: 输入的地震记录数据
    :return: 旋转180度后的数据
    """
    return np.rot90(data, k=2)  # 旋转180度，相当于旋转两次90度

# 数据存储目录
data_dir = r'D:\桌面\实际数据去噪\迁移学习训练集'  # 原始数据文件夹路径

# 遍历文件夹中的npy文件并进行增强
for file_name in os.listdir(data_dir):
    if file_name.endswith('.npy'):
        # 获取文件的完整路径
        file_path = os.path.join(data_dir, file_name)
        data = np.load(file_path)
        
        # 左右翻转并保存为001_1.npy
        augmented_data_lr = flip_left_right(data)
        augmented_file_name_lr = f"{file_name.split('.')[0]}_1.npy"
        augmented_file_path_lr = os.path.join(data_dir, augmented_file_name_lr)
        np.save(augmented_file_path_lr, augmented_data_lr)
        
        # 上下翻转并保存为001_2.npy
        augmented_data_ud = flip_up_down(data)
        augmented_file_name_ud = f"{file_name.split('.')[0]}_2.npy"
        augmented_file_path_ud = os.path.join(data_dir, augmented_file_name_ud)
        np.save(augmented_file_path_ud, augmented_data_ud)
        
        # 180度旋转并保存为001_3.npy
        augmented_data_180 = rotate_180(data)
        augmented_file_name_180 = f"{file_name.split('.')[0]}_3.npy"
        augmented_file_path_180 = os.path.join(data_dir, augmented_file_name_180)
        np.save(augmented_file_path_180, augmented_data_180)
