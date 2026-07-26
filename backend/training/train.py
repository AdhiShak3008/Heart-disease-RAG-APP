from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from backend.training.rosanet import RosaNet
from backend.training.torch_dataset import HeartSoundTorchDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3


@torch.no_grad()
def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:

        recordings = batch["recordings"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(recordings)

        loss = criterion(outputs, labels)

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    average_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def compute_class_weights(train_csv):

    df = pd.read_csv(train_csv)

    patient_labels = df.groupby("patient_id")["murmur"].first()

    label_map = {
        "Absent": 0,
        "Present": 1,
        "Unknown": 2,
    }

    counts = [0, 0, 0]

    for label in patient_labels:

        counts[label_map[label]] += 1

    total = sum(counts)

    weights = [total / (3 * c) for c in counts]

    print("\nClass Distribution")
    print("------------------")

    print(f"Absent  : {counts[0]}")
    print(f"Present : {counts[1]}")
    print(f"Unknown : {counts[2]}")

    print("\nClass Weights")
    print("-------------")

    print(weights)

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


def train():

    base_dir = Path(__file__).resolve().parent.parent

    train_csv = base_dir / "data" / "processed" / "train.csv"
    val_csv = base_dir / "data" / "processed" / "val.csv"

    train_dataset = HeartSoundTorchDataset(train_csv)
    val_dataset = HeartSoundTorchDataset(val_csv)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = RosaNet().to(DEVICE)

    class_weights = compute_class_weights(train_csv)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    save_dir = Path(__file__).resolve().parent.parent.parent / "saved_models"

    save_dir.mkdir(exist_ok=True)

    best_val_acc = 0.0

    print("=" * 60)
    print(f"Training on: {DEVICE}")
    print("=" * 60)

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            recordings = batch["recordings"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(recordings)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
        )

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss      : {train_loss:.4f}")
        print(f"Validation Loss : {val_loss:.4f}")
        print(f"Validation Acc  : {val_acc:.2f}%")

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                save_dir / "rosanet.pt",
            )

            print("✅ Best model saved.")

    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Best Validation Accuracy : {best_val_acc:.2f}%")
    print(f"Model saved to : {save_dir / 'rosanet.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    train()
