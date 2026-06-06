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
from U_Net_CBAM import UNet 

SAMPLES_DIR = r'D:\桌面\随机噪声压制\Black Bridge\数据集(整个剖面)\noisy_npy'
LABELS_DIR = r'D:\桌面\随机噪声压制\Black Bridge\数据集(整个剖面)\normalized_npy'
NUM_EPOCHS = 800
MODEL_DIR = "transfer_model_11_5"
FOLDER_PATH = r'D:\桌面\raw_data\训练日志(transfer_11_5)'

# model = UNet_3Plus_DeepSup(in_channels=1, n_classes=1, dropout_prob = 0)

def calculate_snr(predictions, targets):
    noise = predictions - targets
    signal_power = torch.mean(targets**2)
    noise_power = torch.mean(noise**2)
    return 10 * torch.log10(signal_power / noise_power)

def main():
    dataset = NumpyDataset(SAMPLES_DIR, LABELS_DIR)
    torch.manual_seed(30)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion_mse = torch.nn.MSELoss()
    scaler = GradScaler()
    train_losses = []
    test_losses = []
    snr_values = []
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FOLDER_PATH, exist_ok=True)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    model = model.to(device)
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", unit="batch"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with autocast(device_type = device_type):
                outputs = model(inputs)
                losses =  []
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
        
        
        # 测试评估
        model.eval()
        test_loss = 0.0
        epoch_snr = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast(device_type = device_type):
                    outputs = model(inputs)
                    mse_loss = criterion_mse(outputs[0], targets)
                test_loss += mse_loss.item()
                snr = calculate_snr(outputs[0], targets)
                epoch_snr += snr.item()
        
        test_losses.append(test_loss / len(test_loader))
        snr_values.append(epoch_snr / len(test_loader))
        
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
              f"Training Loss: {train_losses[-1]:.4f}, "
              f"Test Loss: {test_losses[-1]:.4f}, "
              f"SNR: {snr_values[-1]:.2f} dB")
        
        # 保存当前 epoch 的模型
        model_path = os.path.join(MODEL_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到 {model_path}")
    np.save(os.path.join(FOLDER_PATH, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(FOLDER_PATH, 'test_losses.npy'), np.array(test_losses))
    np.save(os.path.join(FOLDER_PATH, 'snr_values.npy'), np.array(snr_values))
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(FOLDER_PATH, 'loss_curve.png'))
    plt.close()
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), snr_values, label='SNR (dB)', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('SNR (dB)')
    plt.title('SNR per Epoch')
    plt.grid()
    plt.savefig(os.path.join(FOLDER_PATH, 'snr_curve.png'))
    plt.close()
    print("训练完成，损失和信噪比曲线已保存为 'loss_curve.png' 和 'snr_curve.png'，训练日志保存为 .npy 文件。")


if __name__ == "__main__":
    main()
