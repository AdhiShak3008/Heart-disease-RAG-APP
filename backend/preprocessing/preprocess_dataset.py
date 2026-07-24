from pathlib import Path

import numpy as np

from backend.preprocessing.audio_loader import load_audio
from backend.preprocessing.spectrogram import generate_mel_spectrogram

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "spectrograms"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_WIDTH = 256


def pad_or_crop(spectrogram: np.ndarray) -> np.ndarray:
    """
    Pad or crop a spectrogram to a fixed width.
    """

    height, width = spectrogram.shape

    if width > TARGET_WIDTH:
        return spectrogram[:, :TARGET_WIDTH]

    if width < TARGET_WIDTH:

        padding = TARGET_WIDTH - width

        return np.pad(
            spectrogram,
            ((0, 0), (0, padding)),
            mode="constant",
            constant_values=0,
        )

    return spectrogram


def normalize(spectrogram: np.ndarray) -> np.ndarray:
    """
    Normalize spectrogram values between 0 and 1.
    """

    minimum = spectrogram.min()
    maximum = spectrogram.max()

    return (spectrogram - minimum) / (maximum - minimum + 1e-8)


def preprocess_audio(audio_path: Path) -> np.ndarray:
    """
    Complete preprocessing pipeline.
    """

    signal, sample_rate = load_audio(audio_path)

    spectrogram = generate_mel_spectrogram(signal, sample_rate)

    spectrogram = pad_or_crop(spectrogram)

    spectrogram = normalize(spectrogram)

    return spectrogram.astype(np.float32)


def main():

    wav_files = sorted(RAW_DIR.glob("*_*.wav"))

    print(f"Found {len(wav_files)} recordings.\n")

    for index, wav_file in enumerate(wav_files, start=1):

        processed = preprocess_audio(wav_file)

        save_path = OUTPUT_DIR / f"{wav_file.stem}.npy"

        np.save(save_path, processed)

        if index % 100 == 0:
            print(f"Processed {index}/{len(wav_files)}")

    print("\nPreprocessing Complete.")
    print(f"Saved files to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
