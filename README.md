# nebulous-py

A declarative python library for the [Nebulous runtime](https://github.com/agentsea/nebulous)

## Installation

```bash
pip install nebu
```

## Usage

Create a pytorch container on runpod with 1 A100 GPU

```python
from nebu import Container, V1EnvVar

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

### Decorator

Run a python function in a container [in progress]

```python
from nebu import container

@container(image="python:3.10-slim", accelerators=["1:A100_SXM"])
def my_function(x: int, y: int) -> int:
    return x + y

result = my_function(1, 2)
print(result)
```

## Contributing

Please open an issue or a PR to contribute to the project.

## Development

```bash
make test
```
