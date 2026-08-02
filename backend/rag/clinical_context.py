"""Clinical context passed from AI models into the RAG pipeline."""

from dataclasses import dataclass


@dataclass
class ClinicalContext:
    """
    Represents structured clinical information produced
    by AI models prior to retrieval.
    """

    prediction: str

    confidence: float

    probabilities: dict[str, float]

    patient_age: int | None = None

    patient_sex: str | None = None

    recording_site: str | None = None

    notes: str | None = None