import torch
import os
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dataset import NumpyDataset  
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
from U_Net_CBAM import UNet

SAMPLES_DIR = r'D:\桌面\实际数据去噪\data\noised2'
LABELS_DIR = r'D:\桌面\实际数据去噪\data\clean'
MODEL_DIR = "conventional_model_6_23"
FOLDER_PATH = r'D:\桌面\实际数据去噪\传统UNet训练日志_6_23'  # 请替换为你的目标文件夹路径
NUM_EPOCHS = 200

# from U_Net import UNet 

def calculate_snr(predictions, targets):
    noise = predictions - targets
    signal_power = torch.mean(targets**2)
    noise_power = torch.mean(noise**2)
    return 10 * torch.log10(signal_power / noise_power)

def main():
    os.makedirs(FOLDER_PATH, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_losses_path = os.path.join(FOLDER_PATH, 'train_losses.npy')
    test_losses_path = os.path.join(FOLDER_PATH, 'test_losses.npy')
    snr_values_path = os.path.join(FOLDER_PATH, 'snr_values.npy')
    loss_curve_path = os.path.join(FOLDER_PATH, 'loss_curve.png')
    snr_curve_path = os.path.join(FOLDER_PATH, 'snr_curve.png')
    dataset = NumpyDataset(SAMPLES_DIR, LABELS_DIR)
    torch.manual_seed(30)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    model = UNet()
    scaler = GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion_mse = torch.nn.MSELoss()
    train_losses = []
    test_losses = []
    snr_values = []
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    model = model.to(device)
    for epoch in range(NUM_EPOCHS):
        model.train()  # 设置模型为训练模式
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", unit="batch"):
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

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
              f"Training Loss: {train_losses[-1]:.4f}, "
              f"Test Loss: {test_losses[-1]:.4f}, "
              f"SNR: {snr_values[-1]:.2f} dB")
        
        # 保存模型（每个 epoch 保存一个文件）
        model_path = os.path.join(MODEL_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到 {model_path}")
    np.save(train_losses_path, np.array(train_losses))
    np.save(test_losses_path, np.array(test_losses))
    np.save(snr_values_path, np.array(snr_values))
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid()
    plt.savefig(loss_curve_path)
    plt.close()
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), snr_values, label='SNR (dB)', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('SNR (dB)')
    plt.title('SNR per Epoch')
    plt.grid()
    plt.savefig(snr_curve_path)
    plt.close()
    print("训练完成，损失和信噪比曲线已保存为 'loss_curve.png' 和 'snr_curve.png', 损失和SNR已保存为.npy文件。")


if __name__ == "__main__":
    main()
