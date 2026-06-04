import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from UNet_3Plus import UNet_3Plus_DeepSup  # 可替换为 UNet
# 使用新的双带通噪声数据集类
from dataset import DualBandNoiseDataset  # 替换原来的N2NDatasetFromCrossline

# 配置参数
input_data_path1 = r"D:\桌面\项目\Stratton\data\Stratton3D_32bit.npy"
input_data_path2 = r"D:\桌面\项目\Stratton\data\MAMI.npy"
input_data_path3 = r"D:\桌面\项目\Stratton\data\Stack_final.npy"

model_dir = "U_Net3+_718"
log_dir = r"D:\桌面\项目\Stratton\训练日志(UNet3+)(7_18)"
patch_size = (128, 128)
batch_size = 32
learning_rate = 1e-4
num_epochs = 400
seed = 30

noise_params1 = {
    'k_range': (0.8, 1.2),   # 低频噪声强度范围
    'lowcut': 25,            # 低频截止(Hz)
    'highcut': 85            # 高频截止(Hz)
}

noise_params2 = {
    'k_range': (1.2, 1.8),   # 高频噪声强度范围
    'lowcut': 85,            # 低频截止(Hz)
    'highcut': 200           # 高频截止(Hz)
}

# 加载数据并创建数据集
data1 = np.load(input_data_path1)
data2 = np.load(input_data_path2)
data3 = np.load(input_data_path3)
data3 = data3[:, 150:900, 0:500]

dataset1 = DualBandNoiseDataset(
    data1,
    patch_size=patch_size,
    stride=(128, 34),
    include_border=True,
    noise_params1=noise_params1,
    noise_params2=noise_params2,
    depth_start=126,  
    depth_end=800,
    use_coherent_noise=True    
)

dataset2 = DualBandNoiseDataset(
    data2,
    patch_size=patch_size,
    stride=(32, 32),
    include_border=True,
    noise_params1 = noise_params1,
    noise_params2 = noise_params2,
    depth_start = 0, 
    depth_end = 1001,
    use_coherent_noise=True     
)
      
dataset3 = DualBandNoiseDataset(
    data3,
    patch_size=patch_size,
    stride=(128, 128),
    include_border=True,
    noise_params1=noise_params1,
    noise_params2=noise_params2,
    depth_start= 0 , 
    depth_end=800 ,
    use_coherent_noise=True   
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
model = UNet_3Plus_DeepSup(in_channels=1, n_classes=1, dropout_prob=0)
model = model.to(device)
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
            total_loss = 0
            for i, pred in enumerate(outputs):
                # 深度监督权重衰减：深层输出权重更高
                weight = 1.0 / (2 ** (len(outputs)-1-i)) 
                total_loss += weight * criterion(pred, targets)
            loss = total_loss
        
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
                loss = criterion(outputs[0], targets)
            epoch_val_loss += loss.item()
            epoch_snr += calculate_snr(outputs[0], targets).item()
    
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