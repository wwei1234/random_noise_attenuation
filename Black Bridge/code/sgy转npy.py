import os
import numpy as np
import segyio

def read_segy(data_dir, shotnum=0):
    with segyio.open(data_dir, 'r', ignore_geometry=True) as f:
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX)  # 总道数
        shot_num = shotnum if shotnum else len(set(sourceX))  # shot 数
        len_shot = trace_num // shot_num  # 每个 shot 的道数
        time = f.trace[0].shape[0]  # 每道的采样点数
        print(f"Reading file: {os.path.basename(data_dir)}")
        data = np.zeros((shot_num, time, len_shot))
        for j in range(shot_num):
            data[j, :, :] = np.asarray([np.copy(x) for x in f.trace[j * len_shot:(j + 1) * len_shot]]).T
        return data

# 归一化
def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

# 批量处理函数（增强：原始+LR+UD+ROT）
def batch_augment_and_save(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    idx = 363  # 用于命名的索引，开始从001
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".sgy"):
            segy_path = os.path.join(input_dir, filename)
            try:
                # 使用你指定的读取方式，返回 shape = (1, time, len_shot)
                data = read_segy(segy_path, shotnum=1)[0]  # 只取第一个 shot，形状是 (4000, 48)
                
                # 使用顺序编号，去掉原文件名中的 "_1"
                base_name = f"{idx:03d}"  # 格式化为三位数编号：001, 002, 003, ...

                # 原始数据
                data_orig = normalize_data(data)
                np.save(os.path.join(output_dir, f"{base_name}_0.npy"), data_orig)

                # 左右翻转
                data_lr = normalize_data(np.fliplr(data))
                np.save(os.path.join(output_dir, f"{base_name}_1.npy"), data_lr)

                # 上下翻转
                data_ud = normalize_data(np.flipud(data))
                np.save(os.path.join(output_dir, f"{base_name}_2.npy"), data_ud)

                # 上下+左右（180度翻转）
                data_rot = normalize_data(np.rot90(data, k=2))  # rot90保持 shape 不变（仍是 4000×48）
                np.save(os.path.join(output_dir, f"{base_name}_3.npy"), data_rot)

                print(f"Saved: {base_name}_0.npy, {base_name}_1.npy, {base_name}_2.npy, {base_name}_3.npy")

                idx += 1  # 增加索引，确保下一个文件使用下一个编号

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

# 执行路径设置
input_folder = r"D:\桌面\Front end express 转化后"       # 输入SGY文件夹
output_folder = r"D:\桌面\实际数据去噪\data2\clean3"     # 输出npy保存文件夹

batch_augment_and_save(input_folder, output_folder)
