import base64
import inspect
import pickle
import time
from typing import Any, Callable, List, Optional

import requests

from nebu.containers.container import Container
from nebu.containers.models import V1ContainerRequest, V1EnvVar, V1ResourceMetaRequest


def container(
    image: str,
    name: Optional[str] = None,
    namespace: str = "default",
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

            # Get function source code
            func_code = inspect.getsource(func)

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


def on_feedback(
    human: Human,
    accelerators: Optional[List[str]] = None,
    platform: str = "runpod",
    python_cmd: str = "python",
    timeout: Optional[str] = None,
    env: Optional[List[V1EnvVar]] = None,
):
    def decorator(func: Callable):
        nonlocal name
        if name is None:
            name = func.__name__

        # Get function source code
        func_code = inspect.getsource(func)

        command = """

"""

        # Create the container request
        container_request = V1ContainerRequest(
            kind="Container",
            platform=platform,
            metadata=V1ResourceMetaRequest(
                name=name,
                namespace=namespace,
            ),
            image=image,
            env=env,
            command=f"{python_cmd} -m nebu.containers.server",
            accelerators=accelerators,
            timeout=timeout,
            proxy_port=8080,
            restart="Never",  # Jobs should not restart
        )

        def run(*args: Any, **kwargs: Any):
            # Create a container from the request
            cont = Container.from_request(container_request)

            # Wait for container to be running
            while (
                cont.status
                and cont.status.status
                and cont.status.status.lower() != "running"
            ):
                print(f"Job '{cont.metadata.name}' not running yet; waiting...")
                time.sleep(1)

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

        # Attach the run method to the container request
        container_request.run = run  # type: ignore

        return container_request

    return decorator
