import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from dataset import NumpyDataset
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from U_Net import UNet


SAMPLES_DIR = r'D:\桌面\U_Net3\开源数据集训练CODE\数据集(整个剖面)\noisy_npy'
LABELS_DIR = r'D:\桌面\U_Net3\开源数据集训练CODE\数据集(整个剖面)\normalized_npy'
TEST_INPUT_PATH = r"D:\桌面\U_Net3\开源数据集训练CODE\data\noised(k=3).npy"
TEST_LABEL_PATH = r"D:\桌面\U_Net3\开源数据集训练CODE\data\clean.npy"
MODEL_DIR = "conventional_model"
LOG_DIR = r'D:\桌面\U_Net3\开源数据集训练CODE\传统UNet训练日志'

# test_samples_dir = r'D:\桌面\U_Net3\开源数据集训练CODE\测试集2\noisy_npy'
# test_labels_dir = r'D:\桌面\U_Net3\开源数据集训练CODE\测试集2\normalized_npy'

SEED = 30
TRAIN_RATIO = 0.8
BATCH_SIZE = 32
NUM_EPOCHS = 600
LEARNING_RATE = 1e-4


def calculate_snr(predictions, targets):
    noise = predictions - targets
    signal_power = torch.mean(targets**2)
    noise_power = torch.mean(noise**2)
    return 10 * torch.log10(signal_power / noise_power)


def load_test_tensor(path):
    data = np.load(path)
    data = torch.tensor(data, dtype=torch.float32)
    return data.unsqueeze(0).unsqueeze(0)


def build_train_loader():
    dataset = NumpyDataset(SAMPLES_DIR, LABELS_DIR)
    torch.manual_seed(SEED)
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, _ = random_split(dataset, [train_size, val_size])
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


def train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    train_loss = 0.0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]", unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    return train_loss / len(train_loader)


def evaluate(model, criterion, test_input, test_label, device, device_type):
    model.eval()
    with torch.no_grad():
        inputs, targets = test_input.to(device), test_label.to(device)
        with autocast(device_type=device_type):
            outputs = model(inputs)
            mse_loss = criterion(outputs, targets).item()
            snr = calculate_snr(outputs, targets).item()
    return mse_loss, snr


def save_curves(train_losses, test_losses, snr_values):
    np.save(os.path.join(LOG_DIR, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(LOG_DIR, 'test_losses.npy'), np.array(test_losses))
    np.save(os.path.join(LOG_DIR, 'snr_values.npy'), np.array(snr_values))

    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(LOG_DIR, 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), snr_values, label='SNR (dB)', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('SNR (dB)')
    plt.title('SNR per Epoch')
    plt.grid()
    plt.savefig(os.path.join(LOG_DIR, 'snr_curve.png'))
    plt.close()


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_loader = build_train_loader()
    test_input = load_test_tensor(TEST_INPUT_PATH)
    test_label = load_test_tensor(TEST_LABEL_PATH)

    model = UNet()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    model = model.to(device)

    scaler = GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_mse = torch.nn.MSELoss()

    train_losses = []
    test_losses = []
    snr_values = []

    # test_dataset = NumpyDataset(test_samples_dir, test_labels_dir)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # with torch.no_grad():
    #     for inputs, targets in test_loader:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         test_loss += loss.item()
    #         snr = calculate_snr(outputs, targets)
    #         epoch_snr += snr.item()
    #
    # test_losses.append(test_loss / len(test_loader))
    # snr_values.append(epoch_snr / len(test_loader))

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_mse, scaler, device, epoch)
        train_losses.append(train_loss)

        test_loss, snr = evaluate(model, criterion_mse, test_input, test_label, device, device_type)
        test_losses.append(test_loss)
        snr_values.append(snr)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], "
              f"Training Loss: {train_losses[-1]:.4f}, "
              f"Test Loss: {test_losses[-1]:.4f}, "
              f"SNR: {snr_values[-1]:.2f} dB")

        model_path = os.path.join(MODEL_DIR, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    save_curves(train_losses, test_losses, snr_values)
    print("Training finished. Curves and npy logs were saved.")


if __name__ == "__main__":
    main()
