from pathlib import Path

import pandas as pd

from backend.preprocessing.dataset_parser import parse_patient

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"

PROCESSED_DIR = BASE_DIR / "data" / "processed" / "spectrograms"

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "dataset_index.csv"


def main():

    rows = []

    patient_files = sorted(RAW_DIR.glob("*.txt"))

    print(f"Found {len(patient_files)} patients.\n")

    for patient_file in patient_files:

        patient = parse_patient(patient_file)

        for recording in patient.recordings.values():

            spectrogram_file = PROCESSED_DIR / f"{recording.path.stem}.npy"

            if not spectrogram_file.exists():
                continue

            rows.append(
                {
                    "patient_id": patient.patient_id,
                    "recording_location": recording.location,
                    "spectrogram_path": str(spectrogram_file),
                    "murmur": patient.murmur,
                    "outcome": patient.outcome,
                    "age": patient.age,
                    "sex": patient.sex,
                    "height": patient.height,
                    "weight": patient.weight,
                    "pregnancy": patient.pregnancy,
                }
            )

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Created dataset index with {len(df)} samples.")

    print(f"Saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
