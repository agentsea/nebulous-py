import base64
import pickle
import time
from typing import Any, Callable, List, Optional

import dill  # Import dill
import requests

from nebu.containers.container import Container


def container(
    image: str,
    name: Optional[str] = None,
    namespace: Optional[str] = None,
    accelerators: Optional[List[str]] = None,
    platform: str = "runpod",
    python_cmd: str = "python",
):
    def decorator(func: Callable):
        nonlocal name
        if name is None:
            name = func.__name__

        def wrapper(*args: Any, **kwargs: Any):
            nonlocal name
            # Create your container with the server script
            cont = Container(
                name=name,  # type: ignore
                namespace=namespace,
                platform=platform,
                image=image,
                accelerators=accelerators,
                # Command to start our function execution server
                command=f"{python_cmd} -m nebu.containers.server",  # TODO: need to get the server code into the container
                proxy_port=8080,
            )

            # Wait for container to be running
            while (
                cont.container.status
                and cont.container.status.status
                and cont.container.status.status.lower() != "running"
            ):
                print(
                    f"Container '{cont.container.metadata.name}' not running yet; waiting..."
                )
                time.sleep(1)

            # Get function source code using dill
            try:
                func_code = dill.source.getsource(func)
            except (OSError, TypeError) as e:
                raise RuntimeError(
                    f"Failed to retrieve source code for function '{func.__name__}'. "
                    "This can happen with functions defined dynamically or interactively "
                    "(e.g., in a Jupyter notebook or REPL). Ensure the function is defined "
                    f"in a standard Python module if possible. Original error: {e}"
                )

            # Serialize arguments using pickle for complex objects
            serialized_args = base64.b64encode(pickle.dumps(args)).decode("utf-8")
            serialized_kwargs = base64.b64encode(pickle.dumps(kwargs)).decode("utf-8")

            # Prepare payload
            payload = {
                "function_code": func_code,
                "args": serialized_args,
                "kwargs": serialized_kwargs,
            }

            # Get container URL
            container_url = (
                cont.status.tailnet_url
                if cont.status and hasattr(cont.status, "tailnet_url")
                else "http://localhost:8080"
            )

            # Send to container and get result
            response = requests.post(f"{container_url}/execute", json=payload)

            if response.status_code != 200:
                raise RuntimeError(f"Function execution failed: {response.text}")

            # Deserialize the result
            serialized_result = response.json()["result"]
            result = pickle.loads(base64.b64decode(serialized_result))

            return result

        return wrapper

    return decorator
