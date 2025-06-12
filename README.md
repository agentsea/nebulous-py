# nebulous-py

A declarative python library for the [Nebulous runtime](https://github.com/agentsea/nebulous)

## Installation

```bash
pip install nebulous-sh
```

## Usage

Create a pytorch container on runpod with 1 A100 GPU

```python
from nebulous import Container, V1EnvVar

container = Container(
    name="pytorch-example",
    namespace="test",
    image="pytorch/pytorch:latest",
    platform="runpod",
    env=[V1EnvVar(name="MY_ENV_VAR", value="my-value")],
    command="nvidia-smi",
    accelerators=["1:A100_SXM"],
    proxy_port=8080,
)

while container.status.status.lower() != "running":
    print(f"Container '{container.metadata.name}' is not running, it is '{container.status.status}', waiting...")
    time.sleep(1)

print(f"Container '{container.metadata.name}' is running")

print(f"You can access the container at {container.status.tailnet_url}")
```

### Processors

Run a python function as a stream processor.

```python
from nebulous import Message, processor
from pydantic import BaseModel

class MyInput(BaseModel):
    a: str
    b: int

@processor(image="python:3.10-slim", accelerators=["1:A100_SXM"])
def my_function(msg: Message[MyInput]):
    return msg

msg = MyInput(a="foo", b=1)
result = my_function(msg)
print(result)
```

## Contributing

Please open an issue or a PR to contribute to the project.

## Development

```bash
make test
```
