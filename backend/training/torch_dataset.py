from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class HeartSoundTorchDataset(Dataset):

    def __init__(self, csv_path: Path):

        self.df = pd.read_csv(csv_path)

        self.patient_ids = sorted(self.df["patient_id"].unique())

    def __len__(self):

        return len(self.patient_ids)

    def _load_recording(self, patient_rows, location):

        row = patient_rows[patient_rows["recording_location"] == location]

        if len(row) == 0:

            return torch.zeros((1, 128, 256), dtype=torch.float32)

        spectrogram = np.load(row.iloc[0]["spectrogram_path"])

        tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)

        return tensor

    def __getitem__(self, index):

        patient_id = self.patient_ids[index]

        patient_rows = self.df[self.df["patient_id"] == patient_id]

        av = self._load_recording(patient_rows, "AV")
        mv = self._load_recording(patient_rows, "MV")
        pv = self._load_recording(patient_rows, "PV")
        tv = self._load_recording(patient_rows, "TV")

        murmur = patient_rows.iloc[0]["murmur"]

        label_map = {
            "Absent": 0,
            "Present": 1,
            "Unknown": 2,
        }

        label = torch.tensor(
            label_map[murmur],
            dtype=torch.long,
        )

        return {
            "patient_id": patient_id,
            "AV": av,
            "MV": mv,
            "PV": pv,
            "TV": tv,
            "label": label,
        }


def main():

    csv_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "processed"
        / "dataset_index.csv"
    )

    dataset = HeartSoundTorchDataset(csv_path)

    print("Patients:", len(dataset))

    sample = dataset[0]

    print()

    print("Patient:", sample["patient_id"])

    print("AV :", sample["AV"].shape)
    print("MV :", sample["MV"].shape)
    print("PV :", sample["PV"].shape)
    print("TV :", sample["TV"].shape)

    print()

    print("Label:", sample["label"])


if __name__ == "__main__":
    main()
