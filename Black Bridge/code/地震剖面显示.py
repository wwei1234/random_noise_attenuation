import segyio
import matplotlib.pyplot as plt
import numpy as np
import torch

def read_segy_as_data(segyfile):
    with segyio.open(segyfile, "r", ignore_geometry=True) as sgy:
        sgy.mmap()
        # 正确获取 inline 和 crossline
        inlines = sgy.attributes(segyio.TraceField.INLINE_3D)[:]
        xlines  = sgy.attributes(segyio.TraceField.CROSSLINE_3D)[:]
        samples = len(sgy.samples)
        ntrace  = len(inlines)

        # 唯一值及其索引映射
        uniq_il = np.unique(inlines)
        uniq_xl = np.unique(xlines)
        ni, nx  = len(uniq_il), len(uniq_xl)
        cube = np.zeros((ni, nx, samples), dtype=sgy.trace.raw[0].dtype)

        il_map = {val: idx for idx, val in enumerate(uniq_il)}
        xl_map = {val: idx for idx, val in enumerate(uniq_xl)}

        # 填充立方体
        for tr in range(ntrace):
            i = il_map[inlines[tr]]
            j = xl_map[xlines[tr]]
            cube[i, j, :] = sgy.trace.raw[tr]
    return cube

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

segyfile = r"D:\桌面\项目\raw_data\data\Stack_final.sgy"
marmousi = read_segy(segyfile)
print(marmousi.shape)
# marmousi = marmousi[50][400:912,500:1012]
# marmousi = marmousi[50][150:662,150:662]
marmousi = marmousi[50][0:1000, 450:]
marmousi = marmousi.astype(np.float64)
# marmousi = normalize_data(marmousi)
print(marmousi.shape)

# np.save(r"D:\桌面\项目\Stratton\数据汇总\Black Bridge\实际地震剖面", marmousi)
plt.figure(figsize=(15,9))
plt.imshow(marmousi, cmap='seismic')
# plt.colorbar()
plt.clim(-6, 6)
# plt.savefig(r"D:\桌面\项目\raw_data\可视化\实际地震记录", dpi =600, bbox_inches='tight')
plt.show()
# input = torch.tensor(marmousi, dtype=torch.float32)
# print(input.shape)





