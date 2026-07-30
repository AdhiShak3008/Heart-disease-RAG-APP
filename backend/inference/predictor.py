from pathlib import Path

import torch

from backend.training.rosanet import RosaNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RosaNetPredictor:

    def __init__(self):

        self.model = RosaNet().to(DEVICE)

        model_path = (
            Path(__file__).resolve().parent.parent.parent
            / "saved_models"
            / "rosanet.pt"
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=DEVICE,
            )
        )

        self.model.eval()

        print("Model loaded successfully.")

    @torch.no_grad()
    def predict(self, recordings):

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

        labels = [
            "Absent",
            "Present",
            "Unknown",
        ]

        probs = probabilities.squeeze(0).cpu().tolist()

        return {
            "prediction": labels[prediction.item()],
            "confidence": confidence.item(),
            "probabilities": {
                "Absent": round(probs[0], 4),
                "Present": round(probs[1], 4),
                "Unknown": round(probs[2], 4),
            },
        }


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
