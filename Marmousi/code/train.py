import torch
import os
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from UNet_3Plus import UNet_3Plus_DeepSup  
from dataset import NumpyDataset  
from torch.amp import GradScaler,autocast
from torch.utils.data import Dataset, DataLoader, random_split
from U_Net import UNet 
# 数据路径
samples_dir = r'D:\桌面\U_Net3\开源数据集训练CODE\数据集(整个剖面)\noisy_npy'
labels_dir = r'D:\桌面\U_Net3\开源数据集训练CODE\数据集(整个剖面)\normalized_npy'
# test_samples_dir = r'D:\桌面\U_Net3\开源数据集训练CODE\测试集2\noisy_npy'
# test_labels_dir = r'D:\桌面\U_Net3\开源数据集训练CODE\测试集2\normalized_npy'

# 创建完整数据集
dataset = NumpyDataset(samples_dir, labels_dir)
# test_dataset = NumpyDataset(test_samples_dir, test_labels_dir)
torch.manual_seed(30)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test = np.load(r"D:\桌面\U_Net3\开源数据集训练CODE\data\noised(k=3).npy")
test = torch.tensor(test, dtype=torch.float32)
test = test.unsqueeze(0).unsqueeze(0)

clean = np.load(r"D:\桌面\U_Net3\开源数据集训练CODE\data\clean.npy")
clean = torch.tensor(clean, dtype=torch.float32)
clean = clean.unsqueeze(0).unsqueeze(0)

# 初始化网络
model = UNet_3Plus_DeepSup(in_channels=1, n_classes=1, dropout_prob = 0)
# model_path = r'D:\桌面\U_Net3\开源数据集训练CODE\U_Net3+_model_4_1(80%)_2\model_epoch_200.pth'
# model.load_state_dict(torch.load(model_path, weights_only=True))

# model = UNet()
# 选择优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# 设置损失函数
criterion_mse = torch.nn.MSELoss()
# 训练参数
num_epochs = 600

# AMP 训练
scaler = GradScaler()

# 初始化保存训练和测试损失、SNR的变量
train_losses = []
test_losses = []
snr_values = []

# 初始化模型保存文件夹  保存训练日志
model_dir = "U_Net3+_model_4_10"
folder_path = r'D:\桌面\U_Net3\开源数据集训练CODE\训练日志(UNet3+)(4_10)'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(folder_path, exist_ok=True)

# 计算信噪比（SNR）的函数
def calculate_snr(predictions, targets):
    noise = predictions - targets
    signal_power = torch.mean(targets**2)
    noise_power = torch.mean(noise**2)
    return 10 * torch.log10(signal_power / noise_power)

# 将模型和数据迁移到设备（GPU 或 CPU）

device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
model = model.to(device)

# 训练和测试循环
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        with autocast(device_type = device_type):
            outputs = model(inputs)
            losses = []
            for i, output in enumerate(outputs):
                loss = criterion_mse(output, targets)
                weight = 0.5 ** i  # 权重策略
                losses.append(weight * loss)
                total_loss = sum(losses)
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += total_loss.item()

    train_losses.append(train_loss / len(train_loader))
    
    # #测试评估
    # model.eval()
    # test_loss = 0.0
    # epoch_snr = 0.0
    # with torch.no_grad():
    #     for inputs, targets in test_loader:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         with autocast(device_type = device_type):
    #             outputs = model(inputs)
    #             mse_loss = criterion_mse(outputs[0], targets)
    #         test_loss += mse_loss.item()
    #         snr = calculate_snr(outputs[0], targets)
    #         epoch_snr += snr.item()
    
    # test_losses.append(test_loss / len(test_loader))
    # snr_values.append(epoch_snr / len(test_loader))

    model.eval()
    with torch.no_grad():
        inputs, targets = test.to(device), clean.to(device)
        with autocast(device_type = device_type):
            outputs = model(inputs)
            mse_loss = criterion_mse(outputs[0], targets)
            mse_loss = mse_loss.item()
            snr = calculate_snr(outputs[0], targets)
            snr = snr.item()

    test_losses.append(mse_loss)
    snr_values.append(snr)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Training Loss: {train_losses[-1]:.4f}, "
          f"Test Loss: {test_losses[-1]:.4f}, "
          f"SNR: {snr_values[-1]:.2f} dB")
    
    # 保存当前 epoch 的模型
    model_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

np.save(os.path.join(folder_path, 'train_losses.npy'), np.array(train_losses))
np.save(os.path.join(folder_path, 'test_losses.npy'), np.array(test_losses))
np.save(os.path.join(folder_path, 'snr_values.npy'), np.array(snr_values))

# 绘制并保存损失曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.grid()
plt.savefig(os.path.join(folder_path, 'loss_curve.png'))
plt.close()
# 绘制并保存 SNR 曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), snr_values, label='SNR (dB)', color='green')
plt.xlabel('Epoch')
plt.ylabel('SNR (dB)')
plt.title('SNR per Epoch')
plt.grid()
plt.savefig(os.path.join(folder_path, 'snr_curve.png'))
plt.close()
print("训练完成，损失和信噪比曲线已保存为 'loss_curve.png' 和 'snr_curve.png'，训练日志保存为 .npy 文件。")
