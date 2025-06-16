from pydantic import BaseModel

from nebulous.processors.decorate import processor
from nebulous.processors.models import Message

print("__name__", __name__)


class Input(BaseModel):
    greeting: str


class Output(BaseModel):
    response: str


# This mimics your setup: an async processor in a subdirectory.
# The decorator will run when this module is imported.
@processor(
    image="python:3.11",
    name="async-test-processor-3",
    accelerators=["1:L40S"],
    wait_for_healthy=True,
)
async def my_async_processor(msg: Message[Input]) -> Output:
    """A simple async processor for testing."""
    if not msg.content:
        raise ValueError("Input message has no content")
    print(f"Processor received: {msg.content.greeting}")
    response_text = f"Hello back to you! You said: {msg.content.greeting}"
    return Output(response=response_text)


print("__name__", __name__)
if __name__ == "__main__":
    print("sending message")
    print(my_async_processor(Input(greeting="Hello, world!"), poll=True, wait=True))
    print("message sent")
