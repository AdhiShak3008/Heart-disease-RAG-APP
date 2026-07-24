from pathlib import Path

import librosa
import numpy as np


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """
    Load a heart sound recording.

    Returns:
        signal : numpy array
        sample_rate : int
    """

    signal, sample_rate = librosa.load(str(audio_path), sr=None)

    return signal, sample_rate
