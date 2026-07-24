from pathlib import Path

import librosa
import numpy as np


def generate_mel_spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
) -> np.ndarray:
    """
    Generate a Mel spectrogram from a heart sound signal.
    """

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_mels=n_mels,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db


if __name__ == "__main__":

    from backend.preprocessing.audio_loader import load_audio

    BASE_DIR = Path(__file__).resolve().parent.parent

    audio_path = BASE_DIR / "data" / "raw" / "2530_AV.wav"

    signal, sr = load_audio(audio_path)

    spectrogram = generate_mel_spectrogram(signal, sr)

    print(spectrogram.shape)
