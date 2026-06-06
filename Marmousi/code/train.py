import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from dataset import NumpyDataset
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from UNet_3Plus import UNet_3Plus_DeepSup


SAMPLES_DIR = r'D:\桌面\U_Net3\开源数据集训练CODE\数据集(整个剖面)\noisy_npy'
LABELS_DIR = r'D:\桌面\U_Net3\开源数据集训练CODE\数据集(整个剖面)\normalized_npy'
TEST_INPUT_PATH = r"D:\桌面\U_Net3\开源数据集训练CODE\data\noised(k=3).npy"
TEST_LABEL_PATH = r"D:\桌面\U_Net3\开源数据集训练CODE\data\clean.npy"
MODEL_DIR = "U_Net3+_model_4_10"
LOG_DIR = r'D:\桌面\U_Net3\开源数据集训练CODE\训练日志(UNet3+)(4_10)'

# test_samples_dir = r'D:\桌面\U_Net3\开源数据集训练CODE\测试集2\noisy_npy'
# test_labels_dir = r'D:\桌面\U_Net3\开源数据集训练CODE\测试集2\normalized_npy'
# model_path = r'D:\桌面\U_Net3\开源数据集训练CODE\U_Net3+_model_4_1(80%)_2\model_epoch_200.pth'

SEED = 30
TRAIN_RATIO = 0.8
BATCH_SIZE = 32
NUM_EPOCHS = 600
LEARNING_RATE = 1e-4
IN_CHANNELS = 1
N_CLASSES = 1
DROPOUT_PROB = 0
DEEP_SUPERVISION_WEIGHT_DECAY = 0.5


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


def train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, device_type, epoch):
    model.train()
    train_loss = 0.0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]", unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast(device_type=device_type):
            outputs = model(inputs)
            losses = []
            for i, output in enumerate(outputs):
                loss = criterion(output, targets)
                weight = DEEP_SUPERVISION_WEIGHT_DECAY ** i
                losses.append(weight * loss)
            total_loss = sum(losses)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += total_loss.item()

    return train_loss / len(train_loader)


def evaluate(model, criterion, test_input, test_label, device, device_type):
    model.eval()
    with torch.no_grad():
        inputs, targets = test_input.to(device), test_label.to(device)
        with autocast(device_type=device_type):
            outputs = model(inputs)
            mse_loss = criterion(outputs[0], targets).item()
            snr = calculate_snr(outputs[0], targets).item()
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
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_loader = build_train_loader()
    test_input = load_test_tensor(TEST_INPUT_PATH)
    test_label = load_test_tensor(TEST_LABEL_PATH)

    model = UNet_3Plus_DeepSup(
        in_channels=IN_CHANNELS,
        n_classes=N_CLASSES,
        dropout_prob=DROPOUT_PROB,
    )
    # model.load_state_dict(torch.load(model_path, weights_only=True))

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_mse = torch.nn.MSELoss()
    scaler = GradScaler()

    train_losses = []
    test_losses = []
    snr_values = []

    # test_dataset = NumpyDataset(test_samples_dir, test_labels_dir)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # #娴嬭瘯璇勪及
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
    #
    # test_losses.append(test_loss / len(test_loader))
    # snr_values.append(epoch_snr / len(test_loader))

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion_mse,
            scaler,
            device,
            device_type,
            epoch,
        )
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
