__Course__ = "CSE 574 - Intro To Machine Learning"
__author__ = "Viral Koshti"
__email__ = "viraljag@buffalo.edu"
__UBPersonNumber__ = "50761354"

from pathlib import Path
import argparse
import csv
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


class DepthLSTM(nn.Module):
    """Conv1d downsamples the 1,800-element vector before recurrent processing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=5, padding=7),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=3, padding=3),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=32, hidden_size=48, num_layers=1, batch_first=True
        )
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(48, 1))

    def forward(self, x):
        # (batch, 1800) -> (batch, 1, 1800) -> (batch, 120, 32)
        x = self.conv(x.unsqueeze(1)).transpose(1, 2)
        _, (h, _) = self.lstm(x)
        return self.classifier(h[-1]).squeeze(1)  # raw logits


def load_data():
    features = pd.read_excel(ROOT / "Features1.xlsx", header=None).to_numpy(
        dtype=np.float32
    )
    labels = pd.read_excel(ROOT / "Class1.xlsx", header=None).to_numpy().reshape(-1)
    if features.shape[0] != 1800:
        raise ValueError(f"Expected 1800 features per image; got {features.shape}")
    x = features.T.copy()  # (samples, 1800)
    if len(x) != len(labels):
        raise ValueError(f"Images/labels mismatch: {len(x)} vs {len(labels)}")
    if not np.isfinite(x).all():
        raise ValueError("Features contain missing or non-finite values")
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError(f"Expected 2 classes, got {classes}")
    y = (labels == classes[1]).astype(np.float32)
    return x, y, classes


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    count, squared_error, wrong = 0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        probabilities = torch.sigmoid(model(xb))
        squared_error += torch.sum((probabilities - yb) ** 2).item()
        wrong += torch.sum((probabilities >= 0.5) != (yb >= 0.5)).item()
        count += len(yb)
    return squared_error / count, 100 * wrong / count


def plot_history(rows, output):
    epochs = [r["epoch"] for r in rows]
    for key, title, ylabel in [
        ("mse", "Mean Squared Error", "MSE"),
        ("error", "Misclassification Rate", "Error (%)"),
    ]:
        plt.figure(figsize=(7, 4.6))
        plt.plot(
            epochs,
            [r[f"train_{key}"] for r in rows],
            label="Training",
            marker="o",
            markersize=3,
        )
        plt.plot(
            epochs,
            [r[f"test_{key}"] for r in rows],
            label="Testing (24 held-out)",
            marker="s",
            markersize=3,
        )
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"LSTM: {title}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"lstm_{key}.png", dpi=180)
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set a seed to reproduce a specific run; default chooses a new random seed",
    )
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("Epochs and batch size must be positive")
    seed = (
        args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    x, y, labels = load_data()
    rng = np.random.default_rng(seed)
    test_idx = rng.choice(len(y), 24, replace=False)
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[test_idx] = False
    x_train, x_test = x[train_mask], x[test_idx]
    y_train, y_test = y[train_mask], y[test_idx]
    # Use only training data to estimate normalization statistics (no test leakage).
    mu, sigma = x_train.mean(axis=0), x_train.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1, sigma)
    x_train = np.clip((x_train - mu) / sigma, -8, 8).astype(np.float32)
    x_test = np.clip((x_test - mu) / sigma, -8, 8).astype(np.float32)

    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    train_eval = DataLoader(train_data, batch_size=128)
    test_loader = DataLoader(test_data, batch_size=24)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.BCEWithLogitsLoss()
    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    print(
        "Seed:",
        seed,
        "Device:",
        device,
        "Class mapping:",
        dict(zip(map(str, labels), [0, 1])),
    )
    print(
        "Training:",
        len(y_train),
        "Testing:",
        len(y_test),
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
    )
    start = time.time()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_function(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_mse, train_error = evaluate(model, train_eval, device)
        test_mse, test_error = evaluate(model, test_loader, device)
        row = dict(
            epoch=epoch,
            train_mse=train_mse,
            train_error=train_error,
            test_mse=test_mse,
            test_error=test_error,
        )
        history.append(row)
        print(
            f"Epoch {epoch:02d}: train MSE {train_mse:.5f}, error {train_error:.2f}% | "
            f"test MSE {test_mse:.5f}, error {test_error:.2f}%"
        )

    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.savetxt(output / "test_indices_0_based.csv", test_idx, fmt="%d", delimiter=",")
    np.savez(output / "normalization.npz", mean=mu, std=sigma, label_values=labels)
    torch.save(model.state_dict(), output / "lstm_model.pt")
    plot_history(history, output)
    print(f"Finished in {time.time() - start:.1f} seconds. Outputs: {output}")


if __name__ == "__main__":
    main()
