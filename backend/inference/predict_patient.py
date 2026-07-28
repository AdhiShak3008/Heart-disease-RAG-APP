from pathlib import Path

import torch

from backend.inference.predictor import RosaNetPredictor
from backend.preprocessing.preprocess_dataset import preprocess_audio

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_patient(patient_id):

    valves = ["AV", "MV", "PV", "TV"]

    spectrograms = []

    for valve in valves:

        wav_path = DATA_DIR / f"{patient_id}_{valve}.wav"

        spectrogram = preprocess_audio(wav_path)

        spectrograms.append(
            torch.tensor(
                spectrogram,
                dtype=torch.float32,
            )
        )

    recordings = torch.stack(spectrograms)

    recordings = recordings.unsqueeze(0)

    return recordings


def main():

    patient_id = "2530"

    predictor = RosaNetPredictor()

    recordings = load_patient(patient_id)

    print("Input Shape:", recordings.shape)

    result = predictor.predict(recordings)

    print()

    print("=" * 50)
    print(f"Patient ID : {patient_id}")
    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
