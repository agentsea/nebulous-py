from pydantic import BaseModel


def print_same_level(text: str):
    print(f"same level: {text}")


class Adapter(BaseModel):
    name: str
    uri: str
    owner: str
    base_model: str
    epochs_trained: int
    last_trained: int
