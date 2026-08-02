"""RosaNet inference."""

from pathlib import Path

import torch

from backend.rag.clinical_context import ClinicalContext
from backend.training.rosanet import RosaNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = (
    "Absent",
    "Present",
    "Unknown",
)


class RosaNetPredictor:
    """Load a trained RosaNet model and perform inference."""

    def __init__(self):

        self.model = RosaNet().to(DEVICE)

        model_path = (
            Path(__file__).resolve().parent.parent.parent
            / "saved_models"
            / "rosanet.pt"
        )

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        print(f"Loaded RosaNet on {DEVICE}.")

    @torch.no_grad()
    def predict(
        self,
        recordings: torch.Tensor,
    ) -> ClinicalContext:
        """
        Predict murmur class from preprocessed recordings.

        Parameters
        ----------
        recordings : torch.Tensor
            Shape: (batch, 4, 128, 256)

        Returns
        -------
        ClinicalContext
        """

        recordings = recordings.to(DEVICE)

        logits = self.model(recordings)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

        probs = probabilities.squeeze(0).cpu().tolist()

        return ClinicalContext(
            prediction=LABELS[prediction.item()],
            confidence=confidence.item(),
            probabilities={
                "Absent": probs[0],
                "Present": probs[1],
                "Unknown": probs[2],
            },
        )


if __name__ == "__main__":

    predictor = RosaNetPredictor()

    dummy = torch.randn(
        1,
        4,
        128,
        256,
    )

    result = predictor.predict(dummy)

    print(result)