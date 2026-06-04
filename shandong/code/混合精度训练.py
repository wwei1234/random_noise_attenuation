import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from U_Net_CBAM import UNet
# 使用新的双带通噪声数据集类
from dataset import SeismicDataset  

model_dir = "CU_Net_828"
log_dir = r"D:\桌面\项目\shandong\训练日志(CUNet)(8_28)"
patch_size = 32
batch_size = 32
learning_rate = 1e-4
num_epochs = 400
seed = 30

dataset1 = SeismicDataset(
    data_dir=r"D:\桌面\项目\shandong\叠加与偏移剖面",  # 你的数据目录
    patch_size=patch_size,          # 32x32切片
    step_size=16,           # 步长16（50%重叠）
    k=0.6,                  # 噪声强度系数，噪声标准差 = 干净数据标准差 * k
    augment=True,           # 启用数据增强
    shotnum=1,              # 每个文件读取1个shot，设为0自动计算
    normalize=True          # 启用数据归一化
)

dataset2 = SeismicDataset(
    data_dir=r"D:\桌面\项目\shandong\叠加与偏移剖面",  # 你的数据目录
    patch_size=patch_size,          # 32x32切片
    step_size=16,           # 步长16（50%重叠）
    k=0.3,                  # 噪声强度系数，噪声标准差 = 干净数据标准差 * k
    augment=True,           # 启用数据增强
    shotnum=1,              # 每个文件读取1个shot，设为0自动计算
    normalize=True          # 启用数据归一化
)
    
dataset3 = SeismicDataset(
    data_dir=r"D:\桌面\项目\shandong\叠加与偏移剖面",  # 你的数据目录
    patch_size=patch_size,          # 32x32切片
    step_size=16,           # 步长16（50%重叠）
    k=0.1,                  # 噪声强度系数，噪声标准差 = 干净数据标准差 * k
    augment=True,           # 启用数据增强
    shotnum=1,              # 每个文件读取1个shot，设为0自动计算
    normalize=True          # 启用数据归一化
)

dataset = ConcatDataset([dataset1, dataset2, dataset3])

# 设置随机种子
torch.manual_seed(seed)
np.random.seed(seed)
# 划分训练集和验证集
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()  # 使用均方误差损失
scaler = GradScaler()  # 混合精度训练

# 创建目录
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 记录训练指标
train_losses, val_losses, snr_values = [], [], []
def calculate_snr(predictions, targets):
    """计算信噪比(SNR)"""
    noise = predictions - targets
    signal_power = torch.mean(targets ** 2)
    noise_power = torch.mean(noise ** 2)
    return 10 * torch.log10(signal_power / noise_power)

# 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    epoch_train_loss = 0.0
    
    for inputs, targets in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch"):
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()
        
        optimizer.zero_grad()
        
        # 混合精度训练
        with autocast(device_type=device.type):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # 梯度缩放和更新
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_train_loss += loss.item()
    
    # 计算平均训练损失
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # 验证阶段
    model.eval()
    epoch_val_loss = 0.0
    epoch_snr = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            
            # 混合精度推理
            with autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            epoch_val_loss += loss.item()
            epoch_snr += calculate_snr(outputs, targets).item()
    
    # 计算平均验证指标
    avg_val_loss = epoch_val_loss / len(val_loader)
    avg_snr = epoch_snr / len(val_loader)
    val_losses.append(avg_val_loss)
    snr_values.append(avg_snr)
    
    # 打印进度
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_train_loss:.6f}, "
          f"Val Loss: {avg_val_loss:.6f}, "
          f"SNR: {avg_snr:.2f} dB")
    
    # 保存模型
    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        torch.save(model.state_dict(), os.path.join(model_dir, f"model_epoch_{epoch+1}.pth"))

# 保存训练日志
np.save(os.path.join(log_dir, 'train_losses.npy'), np.array(train_losses))
np.save(os.path.join(log_dir, 'val_losses.npy'), np.array(val_losses))
np.save(os.path.join(log_dir, 'snr_values.npy'), np.array(snr_values))

# 可视化训练过程
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.grid()

# SNR曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), snr_values, color='green', label='SNR (dB)')
plt.xlabel('Epoch')
plt.ylabel('SNR (dB)')
plt.title('SNR per Epoch')
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'training_metrics.png'))
plt.close()

print("✅ 训练完成，模型已保存，训练指标已输出至日志目录。")