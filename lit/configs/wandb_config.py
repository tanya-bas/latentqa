from typing import List, Optional
from dataclasses import dataclass

@dataclass
class wandb_config:
    project: str = "Latent Steering"
    entity: Optional[str] = "krystiannowak212-minerva-university"
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None
