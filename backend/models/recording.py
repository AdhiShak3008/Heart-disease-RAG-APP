from dataclasses import dataclass
from pathlib import Path


@dataclass
class Recording:
    location: str
    path: Path
