import torch

from backend.preprocessing.preprocess_dataset import preprocess_audio

VALVES = ["AV", "MV", "PV", "TV"]


def prepare_recordings(file_paths):

    spectrograms = []

    for valve in VALVES:

        spectrogram = preprocess_audio(file_paths[valve])

        spectrograms.append(
            torch.tensor(
                spectrogram,
                dtype=torch.float32,
            )
        )

    recordings = torch.stack(spectrograms)

    return recordings.unsqueeze(0)
