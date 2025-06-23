import time

from pydantic import BaseModel

from nebulous.processors.decorate import processor
from nebulous.processors.models import Message

print("__name__", __name__)

VERSION = "v9"


class Input(BaseModel):
    greeting: str


class Output(BaseModel):
    response: str


# This mimics your setup: an async processor in a subdirectory.
# The decorator will run when this module is imported.
@processor(
    image="python:3.11",
    accelerators=["1:L40S"],
    wait_for_healthy=True,
    name=f"async-test-processor-{VERSION}",
)
async def my_async_processor(msg: Message[Input]) -> Output:
    """A simple async processor for testing."""
    if not msg.content:
        raise ValueError("Input message has no content")
    print(f"Processor received: {msg.content.greeting}")
    response_text = f"Hello back to you! You said: {msg.content.greeting}"
    return Output(response=response_text)


# --- Generator processor to test streaming ---


@processor(
    image="python:3.11",
    name=f"async-test-generator-processor-{VERSION}",
    accelerators=["1:L40S"],
    wait_for_healthy=True,
    stream=True,
)
def my_generator_processor(msg: Message[Input]) -> Output:  # type: ignore[misc]
    """Simple generator processor that yields three chunks."""
    for idx in range(3):
        yield Output(response=f"chunk {idx}: {msg.content.greeting}")  # type: ignore[arg-type]
        time.sleep(0.5)


# --- Async generator processor to test async streaming ---
@processor(
    image="python:3.11",
    name=f"async-test-async-generator-processor-{VERSION}",
    accelerators=["1:L40S"],
    wait_for_healthy=True,
    stream=True,
)
async def my_async_generator_processor(msg: Message[Input]) -> Output:  # type: ignore[misc]
    """Simple async generator processor that yields three chunks."""
    import asyncio

    for idx in range(3):
        yield Output(response=f"async chunk {idx}: {msg.content.greeting}")  # type: ignore[arg-type]
        await asyncio.sleep(0.5)


print("__name__", __name__)
if __name__ == "__main__":
    print("sending message")
    print(my_async_processor(Input(greeting="Hello, world!"), poll=True, wait=True))
    print("message sent")
