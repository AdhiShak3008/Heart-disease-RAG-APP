from pathlib import Path
from typing import Any

from backend.preprocessing.audio_loader import load_audio
from backend.preprocessing.dataset_parser import parse_patient


class HeartSoundDataset:
    """
    Dataset for loading heart sound patients and their recordings.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.patient_files = sorted(data_dir.glob("*.txt"))

    def __len__(self) -> int:
        return len(self.patient_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Returns one patient sample.

        Returns:
            {
                "patient": Patient,
                "recordings": {
                    "AV": np.ndarray,
                    "PV": np.ndarray,
                    "TV": np.ndarray,
                    "MV": np.ndarray
                },
                "sample_rate": int
            }
        """

        patient_file = self.patient_files[index]

        patient = parse_patient(patient_file)

        recordings = {}

        sample_rate = None

        for location, path in patient.recordings.items():
            signal, sample_rate = load_audio(path)
            recordings[location] = signal

        return {
            "patient": patient,
            "recordings": recordings,
            "sample_rate": sample_rate,
        }


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"

    dataset = HeartSoundDataset(data_dir)

    print("=" * 60)
    print(f"Total Patients : {len(dataset)}")
    print("=" * 60)

    sample = dataset[0]

    print("\nPatient Information")
    print("-------------------")
    print(sample["patient"])

    print("\nAvailable Recordings")
    print("--------------------")
    print(list(sample["recordings"].keys()))

    print("\nSample Rate")
    print("-----------")
    print(sample["sample_rate"], "Hz")

    print("\nRecording Lengths")
    print("-----------------")

    for location, signal in sample["recordings"].items():
        duration = len(signal) / sample["sample_rate"]
        print(f"{location:2} : {duration:.2f} seconds")


if __name__ == "__main__":
    main()
