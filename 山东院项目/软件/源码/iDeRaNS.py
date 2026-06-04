import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import segyio
import shutil
import time
import gc # 导入垃圾回收模块

# --- 导入您提供的核心模块 ---
from U_Net_CBAM import UNet 
from dataset import SeismicDataset 

# 注意: 为了让 F-K 谱绘制函数正常工作，这里需要从 scipy.fft 导入函数
from scipy.fft import fft2, fftshift, fftfreq

# 配置 Matplotlib 中文显示 (宋体)
# 确保中文显示正常，这是默认字体
rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Song Ti', 'Arial Unicode MS'] 
rcParams['font.family'] = 'SimSun' 
rcParams['axes.unicode_minus'] = False 
# 为了英文/数字的Times New Roman能生效，需要确保Times New Roman可用

# --- 核心数据处理函数 ---

import sys
import os

# 定义一个空的写入器，用于替代 None 的 stdout/stderr
class NullWriter:
    def write(self, *args, **kwargs):
        pass # 丢弃所有写入操作
    def flush(self):
        pass # 丢弃所有刷新操作

# 检查并修复 sys.stdout 和 sys.stderr
# 这能解决 --windowed 模式下 NoneType object has no attribute 'write' 的错误
if sys.stdout is None:
    sys.stdout = NullWriter()
if sys.stderr is None:
    sys.stderr = NullWriter()

def npy_to_sgy(data_npy: np.ndarray, output_sgy_path: str, dt_ms: float, dx_m: float):
    if data_npy.ndim != 2:
        raise ValueError("输入数据必须是二维数组 (nsamples, ntraces)")

    nsamples, ntraces = data_npy.shape
    dt_us = int(dt_ms * 1000) 

    spec = segyio.spec()
    spec.format = 5
    spec.samples = range(nsamples)
    spec.tracecount = ntraces
    
    try:
        with segyio.create(output_sgy_path, spec) as f:
            for i in range(ntraces):
                f.trace[i] = data_npy[:, i]
                f.header[i] = {
                    segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                    segyio.TraceField.FieldRecord: 1,
                    segyio.TraceField.TraceNumber: i + 1,
                    segyio.TraceField.CDP: i + 1,
                    segyio.TraceField.offset: int(i * dx_m), 
                    segyio.TraceField.TRACE_SAMPLE_COUNT: nsamples,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_us
                }
            f.bin[segyio.BinField.Traces] = ntraces
            f.bin[segyio.BinField.Samples] = nsamples
            f.bin[segyio.BinField.Interval] = dt_us
            
        print(f"SGY 文件已保存: {output_sgy_path}")
    except Exception as e:
        raise Exception(f"SGY 转换失败: {e}")

# **已修改：应用字体和字号**
def plot_data_png(data: np.ndarray, title: str, save_path: str, interval, dx, dpi: int = 600, 
                  start_time: float = 0.0, dt_unit: str = 'ms'):
    """
    绘制地震/雷达剖面图 (含噪或去噪结果)。
    新增参数 start_time (起始时间) 和 dt_unit (时间单位: 'ms' 或 'ns')。
    """
    
    # 根据时间单位计算纵坐标刻度
    if dt_unit == 'ms':
        time_unit_factor = interval  # ms
        dx = dx/2  # 地震数据道间距除以2
    elif dt_unit == 'ns':
        time_unit_factor = interval  # ns
        dx = dx
    else:
        time_unit_factor = 1.0
        
    num_samples = data.shape[0]
    
    # extent: (xmin, xmax, ymax, ymin)
    # ymin是起始时间 (start_time)，ymax是结束时间 (start_time + num_samples * dt)
    
    # 纵坐标标题
    ylabel_text = f"Time/{dt_unit}"
    
    plt.figure(figsize=(10, 6))
    plt.imshow(data, aspect='auto', cmap='seismic', 
               extent=(0, data.shape[1]*dx, start_time + num_samples * time_unit_factor, start_time)) 
    
    # 标题、轴标签：中文内容，使用默认宋体，字号 24
    plt.title(title, fontsize=24, fontname='Times New Roman') 
    plt.xlabel('X/m', fontsize=20, fontname='Times New Roman') 

    plt.ylabel(ylabel_text, fontsize=20, fontname='Times New Roman' if dt_unit == 'ns' else 'SimSun') 
    
    # 刻度：数字和英文内容，使用 Times New Roman，字号 20
    plt.xticks(fontsize=14, fontname='Times New Roman')
    plt.yticks(fontsize=14, fontname='Times New Roman')
    plt.clim(0, 1) # 
    
    # Colorbar 标签：中文内容，使用默认宋体，字号 24
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"图片已保存: {save_path}")


def plot_residual_png(data: np.ndarray, title: str, save_path: str, interval, dx,  dpi: int = 600,
                      start_time: float = 0.0, dt_unit: str = 'ms'):
    """
    绘制去除的噪声剖面图。
    新增参数 start_time (起始时间) 和 dt_unit (时间单位: 'ms' 或 'ns')。
    """
    
    # 根据时间单位计算纵坐标刻度
    if dt_unit == 'ms':
        time_unit_factor = interval  # ms
        dx = dx/2  # 地震数据道间距除以2
    elif dt_unit == 'ns':
        time_unit_factor = interval  # ns
        dx = dx
    else:
        time_unit_factor = 1.0
        
    num_samples = data.shape[0]
    
    # 纵坐标标题
    ylabel_text = f"Time/{dt_unit}"
    
    plt.figure(figsize=(10, 6))
    plt.imshow(data, aspect='auto', cmap='seismic', 
               extent=(0, data.shape[1]*dx, start_time + num_samples * time_unit_factor, start_time)) 
    
    # 标题、轴标签：中文内容，使用默认宋体，字号 24
    plt.title(title, fontsize=24, fontname='Times New Roman') 
    plt.xlabel('X/m', fontsize=20, fontname='Times New Roman') 

    plt.ylabel(ylabel_text, fontsize=20, fontname='Times New Roman' if dt_unit == 'ns' else 'SimSun') 
    
    # 刻度：数字和英文内容，使用 Times New Roman，字号 20
    plt.xticks(fontsize=14, fontname='Times New Roman')
    plt.yticks(fontsize=14, fontname='Times New Roman')
    plt.clim(-1, 1)#  关键修改点：去除的噪声使用 (-1, 1)
    
    # Colorbar 标签：中文内容，使用默认宋体，字号 24
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"✅ 图片已保存: {save_path}")


def plot_fk_spectrum(data: np.ndarray, dt: float, dx: float, save_path: str, dpi: int = 600, title: str = "F-K 谱", data_type: str = "seismic"):
    """
    绘制 F-K 谱图。
    新增参数 data_type 用于切换频率单位 (Hz 或 MHz)。
    修改轴标签为 'f' 和 'k'。
    dt 必须是秒 (s) 或纳秒 (ns)。
    """
    # 频率和波数的显示范围
    f_max_display = 200 # 默认为地震数据 (Hz)
    k_max_display = 2
    manual_symmetric_range = 6
    vmin_final = -manual_symmetric_range
    vmax_final = manual_symmetric_range
    
    if data_type == 'radar':

        f_max_display = 2500 
        # dt 已经从 ns 转换成了秒 s，fftfreq计算出的频率是 Hz。
        # 绘图时需要将 Hz 转换为 MHz
        
    # 自定义 colormap
    colors = ["#000074", "#00008B", "#0000B5", "#0000CE", "#0000DC", "#0000FF", "#E9EC0C", "#D44538","#E62210"]
    custom_cmap = LinearSegmentedColormap.from_list('blue-red', colors, N=256)
    norm = TwoSlopeNorm(vmin=vmin_final, vcenter=0, vmax=vmax_final)

    # 去平均值和加窗 
    data = data - np.mean(data, axis=0)
    window = np.hanning(data.shape[0])[:, None]
    data *= window

    # 计算 F-K 谱
    fk_spectrum = np.abs(fftshift(fft2(data)))**2
    fk_log = np.log10(fk_spectrum + 1e-10)

    if data_type == 'radar':
        dx = dx
    else: # seismic
        dx = dx/2  # 地震数据道间距除以2
    # dt 已经是以秒 (s) 或纳秒 (ns) 为单位传入的。
    # 如果 data_type 是 radar, dt 是 ns, 我们假设它被转换成了 s 才能得到 Hz 频率
    # 如果 data_type 是 seismic, dt 是 ms, 我们假设它被转换成了 s 才能得到 Hz 频率

    f = fftshift(fftfreq(data.shape[0], dt)) # 频率 (单位为 1/dt 的倒数)
    k = fftshift(fftfreq(data.shape[1], dx)) # 波数

    # 裁剪显示范围
    if data_type == 'radar':
        f_display_unit = f / 1e6
        ylabel_text = "f / MHz"
    else: # seismic
        f_display_unit = f
        ylabel_text = "f / Hz"
        
    f_mask = (f_display_unit > 0) & (f_display_unit < f_max_display)
    k_mask = np.abs(k) < k_max_display

    f_display = f_display_unit[f_mask]
    k_display = k[k_mask]
    fk_display = fk_log[f_mask][:, k_mask]
    
    plt.figure(figsize=(10, 6))
    im = plt.pcolormesh(k_display, f_display, fk_display,cmap=custom_cmap, norm=norm, shading='auto')

    plt.xlabel("k / m$^{-1}$", fontsize=20, family='Times New Roman')
    plt.ylabel(ylabel_text, fontsize=20, family='Times New Roman') 
    
    plt.title(title, fontsize=24, fontname='Times New Roman') 

    plt.xticks(fontsize=14, fontname='Times New Roman')
    plt.yticks(fontsize=14, fontname='Times New Roman')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"F-K 谱图已保存: {save_path}")

def read_sgy_to_numpy(data_dir: str, shotnum: int = 1) -> np.ndarray:
    with segyio.open(data_dir, 'r', ignore_geometry=True) as f:
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX)
        shot_num = shotnum 
        len_shot = trace_num // shot_num
        time = f.trace[0].shape[0]
        data = np.asarray([np.copy(x) for x in f.trace[0:len_shot]]).T
    return data

def normalize_data(data: np.ndarray) -> np.ndarray:
    """归一化到 [0, 1] 范围"""
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val + 1e-8)

class DenoisingApp:
    def __init__(self, master):
        self.master = master
        # 使用用户提供的窗口标题
        master.title("iDeRaNS去噪软件 V1.0")
        master.geometry("900x700")

        self.mode = tk.StringVar(value="pretrained")
        self.model_path = tk.StringVar(value="")
        self.sgy_path = tk.StringVar(value="")
        self.save_dir = tk.StringVar(value=os.path.join(os.getcwd(), "Denoising_Results"))
        self.plot_fk = tk.BooleanVar(value=True)

        self.data_type = tk.StringVar(value="seismic")
        

        self.start_time = tk.StringVar(value="0.0")

        self.sample_interval_ms = tk.StringVar(value="")
        self.trace_spacing_m = tk.StringVar(value="")
        self.dt_sec = 0.004 # 这是一个默认值，实际会根据用户输入更新

        self.clean_data_dirs_k = []
        self.train_vars = {}
        self.model_save_dir = tk.StringVar(value=os.path.join(os.getcwd(), "Trained_Models"))

        self.main_frame = ttk.Frame(master, padding="10")
        self.main_frame.pack(fill='both', expand=True)

        self.create_mode_selection(self.main_frame)
        
        self.pretrained_frame = ttk.Frame(self.main_frame, padding="10")
        self.create_pretrained_mode_widgets(self.pretrained_frame)
        
        self.train_frame = ttk.Frame(self.main_frame, padding="10")
        self.create_train_mode_widgets(self.train_frame)

        self.show_current_mode_frame()
        os.makedirs(self.save_dir.get(), exist_ok=True)
        os.makedirs(self.model_save_dir.get(), exist_ok=True)


    def create_mode_selection(self, parent):
        mode_frame = ttk.LabelFrame(parent, text="选择操作模式", padding="10")
        mode_frame.pack(fill='x', pady=10)

        ttk.Radiobutton(mode_frame, text="使用已训练模型去噪", 
                        variable=self.mode, value="pretrained", 
                        command=self.show_current_mode_frame).pack(anchor='w', side=tk.LEFT, padx=10)
        
        ttk.Radiobutton(mode_frame, text="训练新模型", 
                        variable=self.mode, value="train", 
                        command=self.show_current_mode_frame).pack(anchor='w', side=tk.LEFT, padx=10)

    def show_current_mode_frame(self):
        self.pretrained_frame.pack_forget()
        self.train_frame.pack_forget()

        if self.mode.get() == "pretrained":
            self.pretrained_frame.pack(fill='both', expand=True)
        elif self.mode.get() == "train":
            self.train_frame.pack(fill='both', expand=True)

    # --- 预训练模型模式 (Pretrained Mode) 界面 ---
    
    def create_pretrained_mode_widgets(self, parent):
        parent.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(parent, text="模型文件 (.pth):").grid(row=row, column=0, sticky='w', pady=5, padx=5)
        ttk.Entry(parent, textvariable=self.model_path, state='readonly').grid(row=row, column=1, sticky='ew', pady=5, padx=5)
        ttk.Button(parent, text="选择模型", command=self.select_model_file).grid(row=row, column=2, sticky='e', pady=5, padx=5)
        
        row += 1
        ttk.Label(parent, text="输入数据 (.sgy):").grid(row=row, column=0, sticky='w', pady=5, padx=5)
        ttk.Entry(parent, textvariable=self.sgy_path, state='readonly').grid(row=row, column=1, sticky='ew', pady=5, padx=5)
        ttk.Button(parent, text="选择 SGY", command=self.select_sgy_file).grid(row=row, column=2, sticky='e', pady=5, padx=5)

        row += 1
        ttk.Label(parent, text="结果保存目录:").grid(row=row, column=0, sticky='w', pady=5, padx=5)
        ttk.Entry(parent, textvariable=self.save_dir, state='readonly').grid(row=row, column=1, sticky='ew', pady=5, padx=5)
        ttk.Button(parent, text="选择目录", command=self.select_save_directory).grid(row=row, column=2, sticky='e', pady=5, padx=5)

        row += 1
        sgy_param_frame = ttk.LabelFrame(parent, text="SGY 转换和 F-K 谱参数", padding="10")
        sgy_param_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=10)
        sgy_param_frame.columnconfigure(1, weight=1)

    
        ttk.Label(sgy_param_frame, text="数据类型:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        data_type_frame = ttk.Frame(sgy_param_frame)
        data_type_frame.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        ttk.Radiobutton(data_type_frame, text="地震数据", variable=self.data_type, value="seismic", 
                        command=self.select_data_type).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(data_type_frame, text="雷达数据", variable=self.data_type, value="radar", 
                        command=self.select_data_type).pack(side=tk.LEFT, padx=5)
        
        self.interval_label = ttk.Label(sgy_param_frame, text="采样间隔 (ms):")
        self.interval_label.grid(row=1, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(sgy_param_frame, textvariable=self.sample_interval_ms).grid(row=1, column=1, sticky='ew', padx=5, pady=5)

        self.start_time_label = ttk.Label(sgy_param_frame, text="起始时间 (ms):")
        self.start_time_label.grid(row=2, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(sgy_param_frame, textvariable=self.start_time).grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        
        ttk.Label(sgy_param_frame, text="道间距 (m):").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(sgy_param_frame, textvariable=self.trace_spacing_m).grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        
        # 初始化单位显示
        self.select_data_type()

        row += 1
        ttk.Checkbutton(parent, text="绘制 F-K 谱", variable=self.plot_fk).grid(row=row, column=0, columnspan=3, sticky='w', pady=10)
        
        row += 1
        ttk.Button(parent, text="开始去噪", command=self.run_denoising).grid(row=row, column=0, columnspan=3, pady=20)
        
        row += 1
        self.status_label_pre = ttk.Label(parent, text="状态: 待命...", foreground="blue")
        self.status_label_pre.grid(row=row, column=0, columnspan=3, sticky='w', pady=5)
    
    def select_data_type(self):
        data_type = self.data_type.get()
        if data_type == "seismic":
            self.interval_label.config(text="采样间隔 (ms):")
            self.start_time_label.config(text="起始时间 (ms):")
        elif data_type == "radar":
            self.interval_label.config(text="采样间隔 (ns):")
            self.start_time_label.config(text="起始时间 (ns):")

    def select_model_file(self):
        filepath = filedialog.askopenfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch Model Files", "*.pth")],
            title="选择已训练的模型文件"
        )
        if filepath:
            self.model_path.set(filepath)
            
    def select_sgy_file(self):
        filepath = filedialog.askopenfilename(
            defaultextension=".sgy",
            filetypes=[("SEG-Y Files", "*.sgy")],
            title="选择需要去噪的 SGY 文件"
        )
        if filepath:
            self.sgy_path.set(filepath)

    def select_save_directory(self):
        dirpath = filedialog.askdirectory(title="选择去噪结果的保存目录")
        if dirpath:
            self.save_dir.set(dirpath)
            os.makedirs(dirpath, exist_ok=True)
            
    # --- 训练模型模式 (Train Mode) 界面 
    
    def create_train_mode_widgets(self, parent):
        """创建"自己训练模型"模式下的所有控件，包含 Batch Size"""
        param_frame = ttk.LabelFrame(parent, text="训练参数设置", padding="10")
        param_frame.pack(fill='x', pady=10)
        param_frame.columnconfigure(1, weight=1)

        labels = ["训练 Epoch:", "学习率 (LR):", "Batch Size:", "模型保存间隔 (Epochs):"]
        vars_names = ["epoch", "lr", "batch_size", "save_interval"]
        default_values = ["400", "1e-4", "32", "10"] 

        for i, (label, name, default) in enumerate(zip(labels, vars_names, default_values)):
            ttk.Label(param_frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            entry = ttk.Entry(param_frame)
            entry.insert(0, default)
            entry.grid(row=i, column=1, sticky='ew', padx=5, pady=2)
            self.train_vars[name] = entry
            
        save_model_frame = ttk.Frame(parent)
        save_model_frame.pack(fill='x', pady=5)
        save_model_frame.columnconfigure(1, weight=1)

        ttk.Label(save_model_frame, text="模型保存目录:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(save_model_frame, textvariable=self.model_save_dir, state='readonly').grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        ttk.Button(save_model_frame, text="选择目录", command=self.select_model_save_directory).grid(row=0, column=2, sticky='e', padx=5, pady=5)

        data_frame = ttk.LabelFrame(parent, text="干净数据文件夹及噪声强度设置 (每个设置将生成一个数据集)", padding="10")
        data_frame.pack(fill='x', pady=10)
        
        self.data_canvas = tk.Canvas(data_frame, height=150)
        self.data_canvas.pack(side="left", fill="both", expand=True)
        
        self.data_scrollbar = ttk.Scrollbar(data_frame, orient="vertical", command=self.data_canvas.yview)
        self.data_scrollbar.pack(side="right", fill="y")
        
        self.data_canvas.configure(yscrollcommand=self.data_scrollbar.set)
        self.data_canvas.bind('<Configure>', lambda e: self.data_canvas.configure(scrollregion = self.data_canvas.bbox("all")))

        self.data_list_frame = ttk.Frame(self.data_canvas)
        self.data_canvas.create_window((0, 0), window=self.data_list_frame, anchor="nw", width=800)
        self.data_list_frame.bind('<Configure>', lambda e: self.data_canvas.configure(scrollregion = self.data_canvas.bbox("all")))
        
        self.update_data_list_display()
        
        ttk.Button(data_frame, text="添加干净数据文件夹", command=self.add_clean_data_dir).pack(side=tk.BOTTOM, pady=5)
        
        ttk.Button(parent, text="开始训练", command=self.run_training).pack(pady=20)
        
        self.status_label_train = ttk.Label(parent, text="状态: 待命...", foreground="blue")
        self.status_label_train.pack(anchor='w', pady=5)
        
    def select_model_save_directory(self):
        dirpath = filedialog.askdirectory(title="选择训练模型的保存目录")
        if dirpath:
            self.model_save_dir.set(dirpath)
            os.makedirs(dirpath, exist_ok=True)

    def add_clean_data_dir(self):
        dirpath = filedialog.askdirectory(title="选择一个干净数据文件夹 (包含 .sgy 文件)")
        if not dirpath:
            return

        k_input = simpledialog.askstring("输入噪声强度", f"请输入文件夹 '{os.path.basename(dirpath)}' 对应的噪声强度 k (例如 0.6):")
        
        if k_input:
            try:
                k_value = float(k_input)
                self.clean_data_dirs_k.append((dirpath, k_value))
                self.update_data_list_display()
            except ValueError:
                messagebox.showerror("输入错误", "噪声强度 k 必须是一个有效的数字。")
            
    def update_data_list_display(self):
        for widget in self.data_list_frame.winfo_children():
            widget.destroy()

        if not self.clean_data_dirs_k:
            ttk.Label(self.data_list_frame, text="尚未添加干净数据文件夹。").pack(padx=5, pady=5, anchor='w')
            return

        for i, (path, k) in enumerate(self.clean_data_dirs_k):
            ttk.Label(self.data_list_frame, 
                      text=f"[数据集 {i+1}] 路径: .../{os.path.basename(path)} | 噪声强度 k: {k}").pack(fill='x', padx=5, pady=2, anchor='w')
            
    # --- 训练逻辑 (与您提供的代码一致) ---
    
    def run_training(self):
        """执行模型训练逻辑 (基于 train.py)"""
        
        # 1. 参数校验和获取
        try:
            epochs = int(self.train_vars['epoch'].get())
            lr = float(self.train_vars['lr'].get())
            # 从用户界面获取 Batch Size
            batch_size = int(self.train_vars['batch_size'].get()) 
            save_interval = int(self.train_vars['save_interval'].get())
        except ValueError:
            messagebox.showerror("参数错误", "请确保所有训练参数都是有效的整数或浮点数。")
            return
            
        model_save_dir = self.model_save_dir.get()
        if not model_save_dir or not self.clean_data_dirs_k:
            messagebox.showerror("输入错误", "请选择模型保存目录并至少添加一个干净数据文件夹。")
            return
        
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir, exist_ok=True)

        # 2. 训练设置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        
        # 混合精度训练：使用 torch.amp 导入 GradScaler 和 autocast，并处理兼容性
        try:
            from torch.amp import GradScaler, autocast
        except ImportError:
            # 如果 torch.amp 导入失败，回退到标准的 torch.cuda.amp (如果设备是CUDA)
            if device.type == 'cuda':
                from torch.cuda.amp import GradScaler, autocast
            else:
                # 针对非 CUDA 设备，定义空操作的类以避免崩溃
                class GradScaler:
                    def scale(self, loss): return loss
                    def step(self, optimizer): optimizer.step()
                    def update(self): pass
                class autocast(object):
                    def __init__(self, *args, **kwargs): pass
                    def __enter__(self): pass
                    def __exit__(self, exc_type, exc_val, exc_tb): pass

        scaler = GradScaler()

        train_losses, val_losses, snr_values = [], [], []
        
        def calculate_snr(predictions, targets):
            noise = predictions - targets
            signal_power = torch.mean(targets ** 2)
            noise_power = torch.mean(noise ** 2)
            return 10 * torch.log10(signal_power / noise_power)

        # 3. 数据集创建
        self.status_label_train.config(text="状态: 正在生成数据集...", foreground="orange")
        self.master.update()
        
        try:
            datasets = []
            for data_dir, k_value in self.clean_data_dirs_k:
                datasets.append(SeismicDataset(
                    data_dir=data_dir,
                    patch_size=32, step_size=16,
                    k=k_value, augment=True,
                    shotnum=1, normalize=True
                ))
            dataset = ConcatDataset(datasets)
            train_size = int(0.95 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            # 使用用户设置的 Batch Size
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
        except Exception as e:
            messagebox.showerror("数据加载错误", f"数据集生成失败: {e}")
            self.status_label_train.config(text="状态: 待命...", foreground="blue")
            return
            
        # 4. 训练循环
        self.status_label_train.config(text="状态: 开始训练...", foreground="green")
        log_dir = os.path.join(model_save_dir, "Training_Logs")
        os.makedirs(log_dir, exist_ok=True)
        self.master.update()
        
        try:
            for epoch in range(epochs):
                model.train()
                epoch_train_loss = 0.0
                
                # 训练阶段
                for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
                    inputs = inputs.to(device).float()
                    targets = targets.to(device).float()
                    
                    optimizer.zero_grad()
                    
                    with autocast(device_type=device.type if device.type != 'cpu' else 'cpu'): 
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    epoch_train_loss += loss.item()
                
                # 验证阶段
                model.eval()
                epoch_val_loss = 0.0
                epoch_snr = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(device).float()
                        targets = targets.to(device).float()
                        
                        with autocast(device_type=device.type if device.type != 'cpu' else 'cpu'):
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                        
                        epoch_val_loss += loss.item()
                        epoch_snr += calculate_snr(outputs, targets).item()
                
                # 记录指标
                avg_train_loss = epoch_train_loss / len(train_loader)
                avg_val_loss = epoch_val_loss / len(val_loader)
                avg_snr = epoch_snr / len(val_loader)
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                snr_values.append(avg_snr)
                
                # 更新状态
                status_text = f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, SNR: {avg_snr:.2f} dB"
                self.status_label_train.config(text="状态: " + status_text, foreground="green")
                self.master.update()
                
                # 保存模型
                if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
                    save_path = os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f"模型已保存: {save_path}")

            # 5. 训练结束后的日志和可视化
            np.save(os.path.join(log_dir, 'train_losses.npy'), np.array(train_losses))
            np.save(os.path.join(log_dir, 'val_losses.npy'), np.array(val_losses))
            np.save(os.path.join(log_dir, 'snr_values.npy'), np.array(snr_values))

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            # 统一使用宋体和 Times New Roman 的标准字号
            plt.plot(range(1, epochs + 1), train_losses, label='训练损失')
            plt.plot(range(1, epochs + 1), val_losses, label='验证损失')
            plt.title('每轮损失', fontsize=24) # Title size 24
            plt.xlabel('Epoch', fontsize=20, family='Times New Roman') # Label size 24, English
            plt.ylabel('Loss (MSE)', fontsize=20, family='Times New Roman') # Label size 24, English
            plt.xticks(fontsize=20, fontname='Times New Roman') # Tick size 20
            plt.yticks(fontsize=20, fontname='Times New Roman') # Tick size 20
            plt.legend(fontsize=14)
            
            plt.subplot(1, 2, 2)
            plt.plot(range(1, epochs + 1), snr_values, color='green', label='信噪比 (dB)')
            plt.title('每轮信噪比', fontsize=24) # Title size 24
            plt.xlabel('Epoch', fontsize=20, family='Times New Roman') # Label size 24, English
            plt.ylabel('SNR (dB)', fontsize=20, family='Times New Roman') # Label size 24, English
            plt.xticks(fontsize=20, fontname='Times New Roman') # Tick size 20
            plt.yticks(fontsize=20, fontname='Times New Roman') # Tick size 20
            
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, 'training_metrics.png'))
            plt.close()

            self.status_label_train.config(text=f"状态: 训练完成! 最终模型: model_epoch_{epochs}.pth", foreground="purple")
            messagebox.showinfo("训练完成", "模型训练已完成，日志和模型已保存。")

        except Exception as e:
            self.status_label_train.config(text="状态: 训练失败!", foreground="red")
            messagebox.showerror("训练失败", f"训练过程中发生错误: {e}")
            return

    def run_denoising(self):
        """执行去噪逻辑，已修改：添加了去除的噪声数据的计算、保存和绘图"""
        
        model_path = self.model_path.get()
        sgy_path = self.sgy_path.get()
        save_dir = self.save_dir.get()
        plot_fk = self.plot_fk.get()
        data_type = self.data_type.get()
        
        try:
            # 采样间隔 (ms 或 ns)
            sample_interval_unit = float(self.sample_interval_ms.get())
            trace_spacing_m = float(self.trace_spacing_m.get())
            start_time_unit = float(self.start_time.get()) 
            

            if data_type == "seismic":
                dt_sec = sample_interval_unit / 1000.0 # ms -> s
                dt_unit = 'ms'
                time_unit_factor_sgy = 1000 # ms -> us (SGY header)
            elif data_type == "radar":
                dt_sec = sample_interval_unit / 1e9 # ns -> s 
                dt_unit = 'ns'
                time_unit_factor_sgy = 1 # ns -> ns (SGY header，segyio默认us，但对于雷达ns数据，这个参数的准确性可能需要外部配置文件支持，这里暂时用1us作为默认)

        except ValueError:
            messagebox.showerror("参数错误", "SGY 转换参数必须是有效的数字。")
            return

        if not all([model_path, sgy_path, save_dir]):
            messagebox.showerror("输入错误", "请选择模型文件、输入 SGY 文件和结果保存目录。")
            return

        self.status_label_pre.config(text="状态: 正在加载模型和数据...", foreground="orange")
        self.master.update()
        
        base_name = os.path.splitext(os.path.basename(sgy_path))[0]
        # Denoised Output Paths
        output_sgy_path = os.path.join(save_dir, f"{base_name}_denoised.sgy")
        output_denoised_png_path = os.path.join(save_dir, f"{base_name}_denoised.png")
        # Noisy Output Path 
        output_noisy_png_path = os.path.join(save_dir, f"{base_name}_noisy.png")
        

        output_residual_sgy_path = os.path.join(save_dir, f"{base_name}_residual.sgy")
        output_residual_png_path = os.path.join(save_dir, f"{base_name}_residual.png")
        
        try:
            # 1. 读取数据
            noisy_data_np = read_sgy_to_numpy(sgy_path, shotnum=1)
            noisy_data_norm = normalize_data(noisy_data_np)
            
            # 2. 绘制含噪数据的 PNG 剖面图 
            self.status_label_pre.config(text="状态: 正在绘制含噪数据剖面图...", foreground="blue")
            self.master.update()
     
            plot_data_png(noisy_data_norm, "Noisy Data", output_noisy_png_path, interval=sample_interval_unit, dx = trace_spacing_m, dpi=600, 
                          start_time=start_time_unit, dt_unit=dt_unit)

            # 3. 绘制含噪数据的 F-K 谱
            if plot_fk:
                self.status_label_pre.config(text="状态: 正在绘制含噪数据 F-K 谱...", foreground="blue")
                self.master.update()
                output_noisy_fk_path = os.path.join(save_dir, f"{base_name}_noisy_FK.png")
     
                plot_fk_spectrum(noisy_data_norm, dt_sec, trace_spacing_m, output_noisy_fk_path,  dpi=600, 
                                 title="F-K (Noisy Data)", data_type=data_type)
            
            # 4. 准备模型输入
            input_tensor = torch.tensor(noisy_data_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # 5. 加载模型 & 执行去噪
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = UNet().to(device)
       
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            self.status_label_pre.config(text="状态: 正在进行模型去噪...", foreground="purple")
            self.master.update()
            
            model.eval()
            with torch.no_grad():
                input_tensor = input_tensor.to(device)
                prediction = model(input_tensor)
                denoised_data_npy = prediction.squeeze(0).squeeze(0).cpu().numpy()
                
    
            residual_data_npy = noisy_data_norm - denoised_data_npy 
            
            self.status_label_pre.config(text="状态: 正在保存结果...", foreground="blue")
            self.master.update()

            # 6. 保存去噪结果文件和图片 
        
            npy_to_sgy(denoised_data_npy, output_sgy_path, sample_interval_unit * (time_unit_factor_sgy/1000), trace_spacing_m) # 恢复到 ms/ns -> us (SGY header)

            plot_data_png(denoised_data_npy, "Denoised Result", output_denoised_png_path, interval=sample_interval_unit, dx = trace_spacing_m, dpi=600, 
                          start_time=start_time_unit, dt_unit=dt_unit)

            npy_to_sgy(residual_data_npy, output_residual_sgy_path, sample_interval_unit * (time_unit_factor_sgy/1000), trace_spacing_m)

            plot_residual_png(residual_data_npy, "Noise", output_residual_png_path, interval=sample_interval_unit,dx = trace_spacing_m,  dpi=600, 
                              start_time=start_time_unit, dt_unit=dt_unit)

            # 7. 绘制去噪数据的 F-K 谱
            if plot_fk:
                output_denoised_fk_path = os.path.join(save_dir, f"{base_name}_denoised_FK.png")

                plot_fk_spectrum(denoised_data_npy, dt_sec, trace_spacing_m, output_denoised_fk_path, dpi=600, 
                                 title="F-K (Denoised Result)", data_type=data_type)
                

                self.status_label_pre.config(text="状态: 正在绘制去除的噪声 F-K 谱...", foreground="blue")
                self.master.update()
                output_residual_fk_path = os.path.join(save_dir, f"{base_name}_residual_FK.png")
         
                plot_fk_spectrum(residual_data_npy, dt_sec, trace_spacing_m, output_residual_fk_path, dpi=600, 
                                 title="F-K (Noise)", data_type=data_type)

            # 8. 显存和内存清理
            del model 
            del input_tensor 
            del prediction
            if device.type == 'cuda':
                gc.collect() 
                torch.cuda.empty_cache()

            self.status_label_pre.config(text="状态: 去噪完成!", foreground="green")
            messagebox.showinfo("完成", "去噪处理和文件保存已成功完成!")

        except Exception as e:
            self.status_label_pre.config(text="状态: 错误发生!", foreground="red")
            messagebox.showerror("错误", f"去噪过程中发生错误: {e}")
            print(f"错误详情: {e}")


# --- 运行应用 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = DenoisingApp(root)
    root.mainloop()