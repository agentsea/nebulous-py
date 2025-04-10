import time

from pydantic import BaseModel, Field


class Adapter(BaseModel):
    created_at: int = Field(default_factory=lambda: int(time.time()))
    name: str
    uri: str
    base_model: str
    owner: str
