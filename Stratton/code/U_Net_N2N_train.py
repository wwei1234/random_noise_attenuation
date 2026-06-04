import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from U_Net import UNet
from N2Ndataset import N2NDatasetFromCrossline

input_data_path = r"D:\桌面\项目\Stratton\data\Stratton3D_32bit.npy"
model_dir = "U_Net_N2N_714"
log_dir = r"D:\桌面\项目\shengbei\训练日志(UNet)(7_14)"
patch_size = (128, 128)
stride = (128, 34)
batch_size = 32
learning_rate = 1e-4
num_epochs = 200
seed = 30
data = np.load(input_data_path)
dataset = N2NDatasetFromCrossline(data, patch_size=patch_size, stride=stride, include_border=True)
torch.manual_seed(seed)
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion_mse = torch.nn.MSELoss()
scaler = GradScaler()

os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
train_losses, test_losses, snr_values = [], [], []

def calculate_snr(predictions, targets):
    noise = predictions - targets
    signal_power = torch.mean(targets ** 2)
    noise_power = torch.mean(noise ** 2)
    return 10 * torch.log10(signal_power / noise_power)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch"):
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()
        optimizer.zero_grad()

        with autocast(device_type=device_type):
            outputs = model(inputs)
            loss = criterion_mse(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证阶段
    model.eval()
    test_loss = 0.0
    epoch_snr = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            with autocast(device_type=device_type):
                outputs = model(inputs)
                mse_loss = criterion_mse(outputs, targets)

            test_loss += mse_loss.item()
            epoch_snr += calculate_snr(outputs, targets).item()

    avg_test_loss = test_loss / len(test_loader)
    avg_snr = epoch_snr / len(test_loader)
    test_losses.append(avg_test_loss)
    snr_values.append(avg_snr)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Test Loss: {avg_test_loss:.4f}, "
          f"SNR: {avg_snr:.2f} dB")

    # 模型保存
    torch.save(model.state_dict(), os.path.join(model_dir, f"model_epoch_{epoch+1}.pth"))

np.save(os.path.join(log_dir, 'train_losses.npy'), np.array(train_losses))
np.save(os.path.join(log_dir, 'test_losses.npy'), np.array(test_losses))
np.save(os.path.join(log_dir, 'snr_values.npy'), np.array(snr_values))

# 绘制损失曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.grid()
plt.savefig(os.path.join(log_dir, 'loss_curve.png'))
plt.close()

# 绘制SNR曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), snr_values, color='green', label='SNR (dB)')
plt.xlabel('Epoch')
plt.ylabel('SNR (dB)')
plt.title('SNR per Epoch')
plt.grid()
plt.savefig(os.path.join(log_dir, 'snr_curve.png'))
plt.close()
print("✅ 训练完成，模型已保存，损失与SNR曲线已输出至日志目录。")
