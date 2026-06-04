import numpy as np
import segyio

# ===================== #
# 1. 读取 npy 数据
# ===================== #
npy_file = r"data.npy"  # 请将路径替换为实际数据文件路径
data = np.load(npy_file)[0]   # data.shape = (nsamples, ntraces)
print(data.shape)
nsamples, ntraces = data.shape
print(f"数据维度: {nsamples} 个采样点, {ntraces} 道")

# ===================== #
# 2. 设置基本参数
# ===================== #
sgy_file = "output_with_headers.sgy"

dt = 4000  # 采样间隔 (微秒)，需要根据实际情况修改
iline = 0  # inline 起始编号，可自定义
xline = 0  # crossline 起始编号，可自定义

# ===================== #
# 3. 创建 SGY 文件
# ===================== #
spec = segyio.spec()
spec.format = 5                # IEEE 浮点格式 (常用)
spec.samples = range(nsamples) # 每道采样点数
spec.tracecount = ntraces      # 道数
offset = 25
with segyio.create(sgy_file, spec) as f:
    for i in range(ntraces):
        # 写入道数据
        f.trace[i] = data[:, i]

        # ===================== #
        # 设置道头信息
        # ===================== #
        f.header[i] = {
            segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,   # 道号（从1开始）
            segyio.TraceField.FieldRecord: 1,               # 炮号
            segyio.TraceField.TraceNumber: i + 1,           # 道号
            segyio.TraceField.CDP: i + 1,                   # CDP编号
            segyio.TraceField.offset: i * offset,               # ✅ 偏移距（小写 offset）
            segyio.TraceField.INLINE_3D: iline,             # inline编号
            segyio.TraceField.CROSSLINE_3D: xline + i,      # crossline编号
            segyio.TraceField.TRACE_SAMPLE_COUNT: nsamples, # 采样点数
            segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt     # 采样间隔
        }

    # ===================== #
    # 设置二进制头信息
    # ===================== #
    f.bin[segyio.BinField.Traces] = ntraces
    f.bin[segyio.BinField.Samples] = nsamples
    f.bin[segyio.BinField.Interval] = dt

print(f"✅ SGY 文件已保存: {sgy_file}")
