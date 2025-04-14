import time
from typing import List

from pydantic import BaseModel, Field


class Adapter(BaseModel):
    created_at: int = Field(default_factory=lambda: int(time.time()))
    name: str
    uri: str
    base_model: str
    owner: str
    epochs_trained: int = Field(default=0)
    last_trained: int = Field(default=0)
    lora_rank: int = Field(default=8)
    lora_alpha: int = Field(default=16)
    lora_dropout: float = Field(default=0.1)
    lora_target_modules: List[str] = Field(default=[])
    learning_rate: float = Field(default=0.0001)
