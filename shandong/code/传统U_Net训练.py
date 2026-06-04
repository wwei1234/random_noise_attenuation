import torch
import os
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dataset import NumpyDataset  
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
# from U_Net import UNet 
from U_Net_CBAM import UNet

# 数据路径
samples_dir1 = r'D:\桌面\项目\实际数据去噪\数据集\传统方法处理\data\noised(124)'
labels_dir1 = r'D:\桌面\项目\实际数据去噪\数据集\传统方法处理\data\clean'

samples_dir2 = r'D:\桌面\项目\实际数据去噪\数据集\未经处理\noised(123)'
labels_dir2 = r'D:\桌面\项目\实际数据去噪\数据集\未经处理\clean'

# # 数据路径
# samples_dir3 = r'D:\桌面\项目\raw_data\数据集(整个剖面)\noisy_npy'
# labels_dir3 = r'D:\桌面\项目\raw_data\数据集(整个剖面)\normalized_npy'

# samples_dir4 = r'D:\桌面\项目\U_Net3+ 合成记录测试\开源数据集训练CODE\数据集(整个剖面)\noisy_npy'
# labels_dir4 = r'D:\桌面\项目\U_Net3+ 合成记录测试\开源数据集训练CODE\数据集(整个剖面)\normalized_npy'

# 创建保存模型的文件夹
model_dir = "conventional_model_6_30"
folder_path = r'D:\桌面\项目\实际数据去噪\传统UNet训练日志_6_30'  # 请替换为你的目标文件夹路径
os.makedirs(folder_path, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
train_losses_path = os.path.join(folder_path, 'train_losses.npy')
test_losses_path = os.path.join(folder_path, 'test_losses.npy')
snr_values_path = os.path.join(folder_path, 'snr_values.npy')
loss_curve_path = os.path.join(folder_path, 'loss_curve.png')
snr_curve_path = os.path.join(folder_path, 'snr_curve.png')

# 创建数据集
dataset1 = NumpyDataset(samples_dir1, labels_dir1)
dataset2 = NumpyDataset(samples_dir2, labels_dir2)
# dataset3 = NumpyDataset(samples_dir3, labels_dir3)
# dataset4 = NumpyDataset(samples_dir4, labels_dir4)
dataset= torch.utils.data.ConcatDataset([dataset1, dataset2])  

torch.manual_seed(30)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# 创建 DataLoader
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 初始化网络
model = UNet()
model_path = r'D:\桌面\项目\实际数据去噪\conventional_model_6_29\model_epoch_200.pth'
model.load_state_dict(torch.load(model_path, weights_only=True))
scaler = GradScaler()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion_mse = torch.nn.MSELoss()

# 训练参数
num_epochs = 200

# 保存训练和测试损失
train_losses = []
test_losses = []
snr_values = []

# 计算信噪比（SNR）
def calculate_snr(predictions, targets):
    noise = predictions - targets
    signal_power = torch.mean(targets**2)
    noise_power = torch.mean(noise**2)
    return 10 * torch.log10(signal_power / noise_power)
   
device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
model = model.to(device)

# 训练和测试循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    train_loss = 0.0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)  # 将数据转移到GPU
        optimizer.zero_grad()  # 清空梯度
        # 前向传播
        outputs = model(inputs)  # 仅计算最终输出
        # 计算损失
        loss = criterion_mse(outputs, targets)
        # 反向传播和优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    
    # 记录训练集损失
    train_losses.append(train_loss / len(train_loader))

    with torch.no_grad():
        test_loss = 0.0
        epoch_snr = 0.0
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据转移到GPU
            outputs = model(inputs)
            loss = criterion_mse(outputs, targets)  # 测试时同样使用最终输出
            test_loss += loss.item()
            # 计算 SNR
            snr = calculate_snr(outputs, targets)
            epoch_snr += snr.item()

    # 记录测试集损失和平均 SNR
    test_losses.append(test_loss / len(test_loader))
    snr_values.append(epoch_snr / len(test_loader))

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Training Loss: {train_losses[-1]:.4f}, "
          f"Test Loss: {test_losses[-1]:.4f}, "
          f"SNR: {snr_values[-1]:.2f} dB")
    
    # 保存模型（每个 epoch 保存一个文件）
    model_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

# 使用 os.path.join 拼接文件夹路径与文件名
np.save(train_losses_path, np.array(train_losses))
np.save(test_losses_path, np.array(test_losses))
np.save(snr_values_path, np.array(snr_values))

# 绘制和保存损失图像
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.grid()
plt.savefig(loss_curve_path)
plt.close()

# 绘制和保存信噪比图像
plt.figure()
plt.plot(range(1, num_epochs + 1), snr_values, label='SNR (dB)', color='green')
plt.xlabel('Epoch')
plt.ylabel('SNR (dB)')
plt.title('SNR per Epoch')
plt.grid()
plt.savefig(snr_curve_path)
plt.close()
print("训练完成，损失和信噪比曲线已保存为 'loss_curve.png' 和 'snr_curve.png', 损失和SNR已保存为.npy文件。")
