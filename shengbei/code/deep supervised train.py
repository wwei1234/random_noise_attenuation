import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from UNet_3Plus import UNet_3Plus_DeepSup  
from dataset import NumpyDataset  

# 数据路径
samples_dir = r'D:\桌面\U_Net3\开源数据集训练CODE\训练集\noisy_npy'
labels_dir = r'D:\桌面\U_Net3\开源数据集训练CODE\训练集\normalized_npy'

test_samples_dir = r'D:\桌面\U_Net3\开源数据集训练CODE\测试集\noisy_npy'
test_labels_dir = r'D:\桌面\U_Net3\开源数据集训练CODE\测试集\normalized_npy'

# 创建完整数据集
train_dataset = NumpyDataset(samples_dir, labels_dir)
test_dataset = NumpyDataset(test_samples_dir, test_labels_dir)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化网络
model = UNet_3Plus_DeepSup(in_channels=1, n_classes=1)

# 选择优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 设置损失函数
criterion = torch.nn.MSELoss()

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

# 创建保存模型的文件夹
model_dir = "U_Net3+_model"
os.makedirs(model_dir, exist_ok=True)

# 将模型和数据迁移到GPU (如果GPU可用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 训练和测试循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    train_loss = 0.0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)  # 将数据转移到GPU

        optimizer.zero_grad()  # 清空梯度

        # 前向传播
        outputs = model(inputs)

        # 计算每个深度监督输出的损失
        losses = []
        for i, output in enumerate(outputs):
            loss = criterion(output, targets)
            weight = 1.0 - i * 0.2  # 权重策略
            losses.append(weight * loss)

        # 总损失
        total_loss = sum(losses)

        # 反向传播和优化
        total_loss.backward()
        optimizer.step()

        # 累加损失
        train_loss += total_loss.item()

    # 记录训练集损失
    train_losses.append(train_loss / len(train_loader))

    # 测试集评估
    model.eval()
    test_loss = 0.0
    epoch_snr = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据转移到GPU
            outputs = model(inputs)
            loss = criterion(outputs[0], targets)  # 测试时仅使用最终输出
            test_loss += loss.item()

            # 计算 SNR
            snr = calculate_snr(outputs[0], targets)
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

folder_path = r'D:\桌面\U_Net3\开源数据集训练CODE\训练日志'  # 请替换为你的目标文件夹路径

# 如果目标文件夹不存在，则创建
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 使用 os.path.join 拼接文件夹路径与文件名
np.save(os.path.join(folder_path, 'train_losses.npy'), np.array(train_losses))
np.save(os.path.join(folder_path, 'test_losses.npy'), np.array(test_losses))
np.save(os.path.join(folder_path, 'snr_values.npy'), np.array(snr_values))

# 绘制和保存损失图像
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.grid()
plt.savefig(r'D:\桌面\U_Net3\开源数据集训练CODE\训练日志\loss_curve.png')
plt.close()

# 绘制和保存信噪比图像
plt.figure()
plt.plot(range(1, num_epochs + 1), snr_values, label='SNR (dB)', color='green')
plt.xlabel('Epoch')
plt.ylabel('SNR (dB)')
plt.title('SNR per Epoch')
plt.grid()
plt.savefig(r'D:\桌面\U_Net3\开源数据集训练CODE\训练日志\snr_curve.png')
plt.close()

print("训练完成，损失和信噪比曲线已保存为 'loss_curve.png' 和 'snr_curve.png'，损失和SNR已保存为.npy文件。")
