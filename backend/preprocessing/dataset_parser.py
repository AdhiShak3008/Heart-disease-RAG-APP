from pathlib import Path

from backend.models.patient import Patient
from backend.models.recording import Recording

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def parse_patient(patient_file: Path) -> Patient:
    """
    Parse a patient metadata file into a Patient object.
    """

    with open(patient_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    patient_id = patient_file.stem

    metadata = {}
    recordings = {}

    for line in lines:

        if line.startswith("#"):
            key, value = line[1:].split(":", 1)
            metadata[key.strip()] = value.strip()

        elif "_AV" in line or "_MV" in line or "_PV" in line or "_TV" in line:

            parts = line.split()

            valve = parts[0]
            wav_file = parts[2]

            recordings[valve] = Recording(
                location=valve,
                path=DATA_DIR / wav_file,
            )

    return Patient(
        patient_id=patient_id,
        age=metadata["Age"],
        sex=metadata["Sex"],
        height=float(metadata["Height"]),
        weight=float(metadata["Weight"]),
        pregnancy=metadata["Pregnancy status"] == "True",
        murmur=metadata["Murmur"],
        outcome=metadata["Outcome"],
        recordings=recordings,
    )


if __name__ == "__main__":
    patient = parse_patient(DATA_DIR / "14241.txt")

    print(patient)

    print("\nRecordings")

    for recording in patient.recordings.values():
        print(f"{recording.location} -> {recording.path}")
