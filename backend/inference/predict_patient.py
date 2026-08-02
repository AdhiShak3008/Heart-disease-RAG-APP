from pathlib import Path

import torch

from backend.inference.predictor import RosaNetPredictor
from backend.preprocessing.preprocess_dataset import preprocess_audio

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_patient(patient_id: str) -> torch.Tensor:
    """Load all four valve recordings for a patient."""

    valves = ("AV", "MV", "PV", "TV")

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

    return recordings.unsqueeze(0)


def main():

    patient_id = "2530"

    predictor = RosaNetPredictor()

    recordings = load_patient(patient_id)

    print(f"Input Shape : {tuple(recordings.shape)}")

    clinical_context = predictor.predict(recordings)

    print()
    print("=" * 60)
    print(f"Patient ID : {patient_id}")
    print(f"Prediction : {clinical_context.prediction}")
    print(f"Confidence : {clinical_context.confidence:.2%}")

    print("\nClass Probabilities")

    for label, probability in clinical_context.probabilities.items():
        print(f"  {label:<8}: {probability:.2%}")

    print("=" * 60)


if __name__ == "__main__":
    main()