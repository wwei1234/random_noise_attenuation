import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 更清晰的汉字字体
rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 加载数据
clean = np.load(r"D:\桌面\U_Net3\开源数据集训练CODE\data\clean.npy")
noised = np.load(r"D:\桌面\U_Net3\开源数据集训练CODE\data\noised(k=3).npy")
denoised = np.load(r"D:\桌面\U_Net3\开源数据集训练CODE\训练日志(15-55hz)(3_31)\model_195.npy")

# 创建画布
plt.figure(figsize=(10, 6), dpi=100)  # 调整画布尺寸

# 检查数据维度并处理
def preprocess_data(data):
    if data.ndim > 1:
        print("检测到多维数据，自动取第一行绘制")
        return data[:][0]
    return data

# 绘制三条曲线
x = np.arange(len(preprocess_data(clean)))  # 生成x轴坐标
plt.plot(x, preprocess_data(clean), 
         label='干净信号', 
         color='#1f77b4', 
         linewidth=1.5,
         linestyle='-')

# plt.plot(x, preprocess_data(noised),
#          label='含噪信号 (k=3)',
#          color='#ff7f0e',
#          linewidth=1.2,
#          linestyle='--',
#          alpha=0.8)

plt.plot(x, preprocess_data(denoised),
         label='去噪结果',
         color='#ff7f0e',
         linewidth=1.5,
         linestyle='-.')

# 图表装饰
plt.title('单道去噪效果', fontsize=14, pad=20)
plt.xlabel('Samples', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)  # 添加网格线

# 智能图例位置
plt.legend(
    loc='upper right',
    frameon=True,
    shadow=True,
    fontsize=10,
    ncol=1,
    bbox_to_anchor=(1.12, 1)  # 防止图例覆盖曲线
)

# 自动调整坐标范围
plt.xlim([x.min(), x.max()])
plt.ylim([min(clean.min(), noised.min(), denoised.min())*1.1,
          max(clean.max(), noised.max(), denoised.max())*1.1])

# 优化布局
plt.tight_layout()

# 保存高清图像
plt.savefig(r'D:\桌面\U_Net3\开源数据集训练CODE\可视化\UNet_3+单道去噪效果.png', 
           dpi=2400, 
           bbox_inches='tight', 
           pad_inches=0.1)

plt.show()