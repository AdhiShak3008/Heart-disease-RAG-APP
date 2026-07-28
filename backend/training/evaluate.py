from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from backend.training.rosanet import RosaNet
from backend.training.torch_dataset import HeartSoundTorchDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():

    base_dir = Path(__file__).resolve().parent.parent

    test_csv = base_dir / "data" / "processed" / "test.csv"

    model_path = (
        Path(__file__).resolve().parent.parent.parent / "saved_models" / "rosanet.pt"
    )

    dataset = HeartSoundTorchDataset(test_csv)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
    )

    model = RosaNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
        )
    )

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for batch in loader:

            recordings = batch["recordings"].to(DEVICE)

            target = batch["label"]

            outputs = model(recordings)

            pred = outputs.argmax(dim=1).cpu()

            predictions.extend(pred.numpy())

            labels.extend(target.numpy())

    print("\n" + "=" * 60)

    print(f"Test Accuracy : {accuracy_score(labels, predictions)*100:.2f}%")

    print("=" * 60)

    print("\nClassification Report\n")

    print(
        classification_report(
            labels,
            predictions,
            target_names=[
                "Absent",
                "Present",
                "Unknown",
            ],
            digits=4,
        )
    )

    print("Confusion Matrix\n")

    print(
        confusion_matrix(
            labels,
            predictions,
        )
    )


if __name__ == "__main__":

    evaluate()
