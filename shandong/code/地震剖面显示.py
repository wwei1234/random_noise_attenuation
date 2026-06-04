import segyio
import matplotlib.pyplot as plt
import numpy as np
import torch
from obspy.io.segy.segy import _read_segy

def read_segy(data_dir,shotnum=0):
    with segyio.open(data_dir,'r',ignore_geometry=True) as f:
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX) #number of all trace
        if shotnum:
            shot_num = shotnum 
        else:
            shot_num = len(set(sourceX)) #shot number 
        len_shot = trace_num//shot_num   #The length of the data in each shot data
        time = f.trace[0].shape[0]
        print('start read segy data')
        data = np.zeros((shot_num,time,len_shot))
        for j in range(0,shot_num):
            data[j,:,:] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
        return data

def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

def agc_enhanced(data, window_len=30, gamma=0.5):
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

        # 添加 gamma 非线性压缩：增强弱振幅，抑制强振幅
        agc_trace = np.sign(agc_trace) * (np.abs(agc_trace) ** gamma)
        agc_data[:, i] = agc_trace

    # 归一化到 [0, 1]
    min_val = agc_data.min()
    max_val = agc_data.max()
    agc_data = (agc_data - min_val) / (max_val - min_val + 1e-8)
    return agc_data

segyfile = r"D:\桌面\随机噪声压制\shandong\雷达\ldsn06.sgy"
data = read_segy(segyfile,shotnum=1)
data = data[0]
data = normalize_data(data)
print(data.shape)
# data = data[0][60:796, 0:336]
# data = normalize_data(data)
# print(data.shape)

plt.figure()
plt.imshow(data, "seismic", aspect = "auto")
plt.colorbar()
plt.show()






