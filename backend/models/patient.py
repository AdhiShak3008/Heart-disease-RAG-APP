from dataclasses import dataclass
from backend.models.recording import Recording


@dataclass
class Patient:
    patient_id: str

    age: str
    sex: str
    height: float
    weight: float
    pregnancy: bool

    murmur: str
    outcome: str

    recordings: dict[str, Recording]
