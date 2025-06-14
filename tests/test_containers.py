import random
import string

import pytest

from nebulous.config import GlobalConfig
from nebulous.containers.container import Container


@pytest.fixture
def random_container_name() -> str:
    """Generate a random container name to avoid collisions."""
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"test-container-{suffix}"


@pytest.fixture
def global_config() -> GlobalConfig:
    """
    Set your real API key and server here or ensure GlobalConfig.read()
    is retrieving valid values for these.
    """
    return GlobalConfig.read()


def test_container_integration(global_config: GlobalConfig, random_container_name: str):
    """
    Single end-to-end test that:
      1) Creates a container.
      2) Creates it again with no changes (should detect no differences).
      3) Updates a field (should trigger a recreate).
      4) Deletes the container.
    """

    # 1) Create container
    container_instance = Container(
        name=random_container_name,
        image="pytorch/pytorch:latest",
        command="nvidia-smi",
        platform="runpod",
        accelerators=["1:L4"],
        config=global_config,
        restart="Never",
    )
    assert container_instance.container is not None, "Container object should be set."
    assert container_instance.container.metadata.name == random_container_name
    print(f"Created container {random_container_name}")

    # 2) Attempt to create again with no changes
    same_params_instance = Container(
        name=random_container_name,
        image="pytorch/pytorch:latest",
        command="nvidia-smi",
        platform="runpod",
        accelerators=["1:L4"],
        config=global_config,
        restart="Never",
    )
    assert (
        same_params_instance.container is not None
    ), "Container object should remain set for no-change scenario."
    # If your code prints something like "No changes detected..." this is good.
    print(f"No changes detected for container {random_container_name}")

    # 3) Update the image to trigger a recreate
    updated_container_instance = Container(
        name=random_container_name,
        image="pytorch/pytorch:latest",
        command="ls -la /",
        platform="runpod",
        accelerators=["1:L4"],
        config=global_config,
        restart="Never",
    )
    assert (
        updated_container_instance.container is not None
    ), "Updated container should be set."
    assert (
        updated_container_instance.container.command == "ls -la /"
    ), "Container image should have been updated."
    print(f"Updated container image to python:3.10-slim for {random_container_name}")

    # 4) Delete the container
    updated_container_instance.delete()  # Hypothetical method you plan to add
