import segyio
import matplotlib.pyplot as plt
import numpy as np
import torch

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

def seismic_to_3d_array(segyfile):
    with segyio.open(segyfile, "r", 9, 21) as sgydata:

        print('start read segy data')
        # 获取行列数和样本数
        ilines = sgydata.ilines
        xlines = sgydata.xlines
        samples = sgydata.samples

        # 创建一个三维 NumPy 数组
        # 形状为 (ilines数量, xlines数量, 样本数量)
        seismic_data = np.zeros((len(ilines), len(xlines), len(samples)))

        # 填充数据
        for i, iline in enumerate(ilines):
            for j, xline in enumerate(xlines):
                # 读取每个道的样本数据
                seismic_data[i, j, :] = sgydata.iline[iline][j]  # 这里假设是以道数索引方式访问

    return seismic_data

def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

def agc(data, window_len=30):
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
        agc_data[:, i] = agc_trace

    # 归一化到 [0, 1]
    min_val = agc_data.min()
    max_val = agc_data.max()
    agc_data = (agc_data - min_val) / (max_val - min_val + 1e-8)
    return agc_data

segyfile = r"D:\桌面\项目\shengbei\data\shengbei_pstm_cb.sgy"
data = read_segy(segyfile,shotnum=1)
# data = seismic_to_3d_array(segyfile)
data = data[0]#[125:1885, 0:320]
# data = data
data = normalize_data(data)
# np.save(r"D:\桌面\项目\shengbei\data\shengbei_pstm_cb_local.npy", data)
print(data.shape)
# plt.figure()
# plt.imshow(data, "seismic", aspect='auto')
# plt.colorbar()
# # plt.savefig(r"D:\桌面\Denoise\可视化\合成地震记录", dpi =2400)
# plt.show()
#(641, 331, 7000)



 