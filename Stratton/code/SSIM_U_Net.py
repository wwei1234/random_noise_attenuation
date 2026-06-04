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
import torch.nn.functional as F

input_data_path = r"D:\桌面\项目\Stratton\data\Stratton3D_32bit.npy"
model_dir = "U_Net_SSIM_715"
log_dir = r"D:\桌面\项目\shengbei\训练日志(UNet)(7_14)"
patch_size = (128, 128)
stride = (128, 34)
batch_size = 32
learning_rate = 1e-4
num_epochs = 200
seed = 30

# 损失函数权重
mse_weight = 1.0
ssim_weight = 0.1  # 结构相似性损失权重

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

def gaussian_kernel(kernel_size, sigma):
    """生成高斯核"""
    x = torch.arange(kernel_size, dtype=torch.float32)
    x = x - kernel_size // 2
    gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    return gauss

def create_2d_gaussian_kernel(kernel_size, sigma):
    """创建2D高斯核"""
    kernel_1d = gaussian_kernel(kernel_size, sigma)
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    return kernel_2d

def ssim_loss(img1, img2, kernel_size=11, sigma=1.5, k1=0.01, k2=0.03):
    """
    计算结构相似性指数损失
    img1: 去除的噪声 (input - output)
    img2: 去噪结果 (output)
    返回: 1 - SSIM (相似性越小，损失越小)
    """
    # 确保输入为float类型
    img1 = img1.float()
    img2 = img2.float()
    
    # 数据范围
    data_range = 1.0  # 假设数据已归一化到[0,1]
    
    # 创建高斯核
    kernel = create_2d_gaussian_kernel(kernel_size, sigma)
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(img1.device)
    
    # 计算均值
    mu1 = F.conv2d(img1, kernel, padding=kernel_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, kernel, padding=kernel_size//2, groups=img2.shape[1])
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=kernel_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=kernel_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel_size//2, groups=img1.shape[1]) - mu1_mu2
    
    # SSIM常数
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # 返回1-SSIM作为损失（相似性越小，损失越小）
    return 1 - ssim_map.mean()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch"):
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()
        optimizer.zero_grad()

        with autocast(device_type=device_type):
            outputs = model(inputs)
            
            # MSE损失
            mse_loss = criterion_mse(outputs, targets)
            
            # 计算去除的噪声
            removed_noise = inputs - outputs
            
            # 结构相似性损失（去除的噪声与去噪结果的相似性）
            ssim_loss_value = ssim_loss(removed_noise, outputs)
            
            # 总损失
            total_loss = mse_weight * mse_loss + ssim_weight * ssim_loss_value

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += total_loss.item()

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
                
                # MSE损失
                mse_loss = criterion_mse(outputs, targets)
                
                # 计算去除的噪声
                removed_noise = inputs - outputs
                
                # 结构相似性损失
                ssim_loss_value = ssim_loss(removed_noise, outputs)
                
                # 总损失
                total_loss = mse_weight * mse_loss + ssim_weight * ssim_loss_value

            test_loss += total_loss.item()
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