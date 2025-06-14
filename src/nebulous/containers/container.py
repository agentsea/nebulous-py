from typing import List, Optional

import requests

from nebulous.config import GlobalConfig  # or wherever your GlobalConfig is defined
from nebulous.containers.models import (
    DEFAULT_RESTART_POLICY,
    V1AuthzConfig,
    V1Container,
    V1ContainerRequest,
    V1ContainerResources,
    V1Containers,
    V1ContainerSearch,
    V1EnvVar,
    V1Meter,
    V1PortRequest,
    V1ResourceMetaRequest,
    V1SSHKey,
    V1VolumePath,
)
from nebulous.logging import logger  # Import the logger
from nebulous.meta import V1ResourceReference


class Container:
    def __init__(
        self,
        name: str,
        image: str,
        namespace: Optional[str] = None,
        platform: Optional[str] = None,
        env: Optional[List[V1EnvVar]] = None,
        command: Optional[str] = None,
        volumes: Optional[List[V1VolumePath]] = None,
        accelerators: Optional[List[str]] = None,
        resources: Optional[V1ContainerResources] = None,
        meters: Optional[List[V1Meter]] = None,
        restart: str = DEFAULT_RESTART_POLICY,
        queue: Optional[str] = None,
        timeout: Optional[str] = None,
        ssh_keys: Optional[List[V1SSHKey]] = None,
        ports: Optional[List[V1PortRequest]] = None,
        proxy_port: Optional[int] = None,
        authz: Optional[V1AuthzConfig] = None,
        config: Optional[GlobalConfig] = None,
        api_key: Optional[str] = None,
    ):
        # Fallback to a default config if none is provided
        config = config or GlobalConfig.read()
        current_server = config.get_current_server_config()
        if not current_server:
            raise ValueError("No current server config found")
        self.api_key = api_key or current_server.api_key
        self.nebu_host = current_server.server
        self.config = config

        logger.debug(f"nebu_host: {self.nebu_host}")
        logger.debug(f"api_key: {self.api_key}")

        # Construct the containers base URL
        self.containers_url = f"{self.nebu_host}/v1/containers"

        # Attempt to find an existing container
        response = requests.get(
            self.containers_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()

        meta_request = V1ResourceMetaRequest(
            name=name,
            namespace=namespace,
        )

        containers = V1Containers.model_validate(response.json())
        logger.debug(f"containers: {containers}")
        existing = next(
            (
                c
                for c in containers.containers
                if c.metadata.name == name and c.metadata.namespace == namespace
            ),
            None,
        )

        logger.debug(f"existing: {existing}")

        if not existing:
            # If there's no existing container, create one:
            if not image:
                raise ValueError("An 'image' is required to create a new container.")

            create_request = V1ContainerRequest(
                kind="Container",
                platform=platform,
                metadata=meta_request,
                image=image,
                env=env,
                command=command,
                volumes=volumes,
                accelerators=accelerators,
                resources=resources,
                meters=meters,
                restart=restart,
                queue=queue,
                timeout=timeout,
                ssh_keys=ssh_keys,
                ports=ports,
                proxy_port=proxy_port,
                authz=authz,
            )
            create_response = requests.post(
                self.containers_url,
                json=create_request.model_dump(),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            create_response.raise_for_status()
            self.container = V1Container.model_validate(create_response.json())
            logger.info(f"Created container {self.container.metadata.name}")
        else:
            # If container is found, check if anything has changed
            # Gather the updated fields from the function arguments
            updated_image = image or existing.image
            updated_env = env if env is not None else existing.env
            updated_command = command if command is not None else existing.command
            updated_volumes = volumes if volumes is not None else existing.volumes
            updated_accelerators = (
                accelerators if accelerators is not None else existing.accelerators
            )
            updated_resources = (
                resources if resources is not None else existing.resources
            )
            updated_meters = meters if meters is not None else existing.meters
            updated_restart = restart if restart else existing.restart
            updated_queue = queue if queue else existing.queue
            updated_timeout = timeout if timeout else existing.timeout
            updated_proxy_port = proxy_port if proxy_port else existing.proxy_port
            updated_authz = authz if authz else existing.authz

            # Determine if fields differ. You can adapt these checks as needed
            # (for example, deep comparison for complex field structures).
            fields_changed = (
                existing.image != updated_image
                or existing.env != updated_env
                or existing.command != updated_command
                or existing.volumes != updated_volumes
                or existing.accelerators != updated_accelerators
                or existing.resources != updated_resources
                or existing.meters != updated_meters
                or existing.restart != updated_restart
                or existing.queue != updated_queue
                or existing.timeout != updated_timeout
                or existing.proxy_port != updated_proxy_port
                or existing.authz != updated_authz
            )

            if not fields_changed:
                # Nothing changed—do nothing
                logger.info(
                    f"No changes detected for container {existing.metadata.name}."
                )
                self.container = existing
                return

            logger.info(
                f"Detected changes for container {existing.metadata.name}, deleting and recreating."
            )

            # Construct the URL to delete the existing container
            delete_url = (
                f"{self.containers_url}/{existing.metadata.namespace}/{existing.metadata.name}"
                if existing.metadata.namespace
                else f"{self.containers_url}/{existing.metadata.name}"
            )

            # Delete the existing container
            delete_response = requests.delete(
                delete_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            delete_response.raise_for_status()
            logger.info(f"Deleted container {existing.metadata.name}")

            # Now recreate the container using the updated parameters
            create_request = V1ContainerRequest(
                kind="Container",
                platform=platform,
                metadata=meta_request,
                image=updated_image,
                env=updated_env,
                command=updated_command,
                volumes=updated_volumes,
                accelerators=updated_accelerators,
                resources=updated_resources,
                meters=updated_meters,
                restart=updated_restart,
                queue=updated_queue,
                timeout=updated_timeout,
                ssh_keys=ssh_keys,
                ports=ports,
                proxy_port=updated_proxy_port,
                authz=updated_authz,
            )
            create_response = requests.post(
                self.containers_url,
                json=create_request.model_dump(),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            create_response.raise_for_status()
            self.container = V1Container.model_validate(create_response.json())
            logger.info(f"Recreated container {self.container.metadata.name}")

        # Save constructor params to `self` for reference, like you do in ReplayBuffer.
        self.kind = "Container"
        self.namespace = self.container.metadata.namespace
        self.name = self.container.metadata.name
        self.platform = platform
        self.metadata = meta_request
        self.image = image
        self.env = env
        self.command = command
        self.volumes = volumes
        self.accelerators = accelerators
        self.resources = resources
        self.meters = meters
        self.restart = restart
        self.queue = queue
        self.timeout = timeout
        self.ssh_keys = ssh_keys
        self.status = self.container.status

    @classmethod
    def from_request(cls, request: V1ContainerRequest) -> V1Container:
        return V1Container(**request.model_dump())

    def delete(self) -> None:
        """
        Deletes the container by making a DELETE request to /v1/containers/:namespace/:name.
        """
        # Construct the url using instance attributes
        delete_url = f"{self.containers_url}/{self.namespace}/{self.name}"

        # Perform the deletion
        response = requests.delete(
            delete_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        logger.info(f"Deleted container {self.name} in namespace {self.namespace}")

    @classmethod
    def get(
        cls,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        config: Optional[GlobalConfig] = None,
        api_key: Optional[str] = None,
    ) -> List[V1Container]:
        """
        Get a list of containers that match the optional name and/or namespace filters.
        """
        config = config or GlobalConfig.read()
        current_server = config.get_current_server_config()
        if not current_server:
            raise ValueError("No current server config found")
        api_key = api_key or current_server.api_key
        nebu_host = current_server.server

        containers_url = f"{nebu_host}/v1/containers"

        response = requests.get(
            containers_url, headers={"Authorization": f"Bearer {api_key}"}
        )
        response.raise_for_status()

        containers_response = V1Containers.model_validate(response.json())
        filtered_containers = containers_response.containers

        if name:
            filtered_containers = [
                container
                for container in filtered_containers
                if container.metadata.name == name
            ]
        if namespace:
            filtered_containers = [
                container
                for container in filtered_containers
                if container.metadata.namespace == namespace
            ]

        return filtered_containers

    @classmethod
    def load(
        cls,
        name: str,
        namespace: Optional[str] = None,
        config: Optional[GlobalConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Get a container from the remote server.
        """
        containers = cls.get(namespace=namespace, name=name, config=config)
        if not containers:
            raise ValueError("Container not found")
        container_v1 = containers[0]

        out = cls.__new__(cls)
        out.container = container_v1
        out.config = config or GlobalConfig.read()
        current_server = out.config.get_current_server_config()
        if not current_server:
            raise ValueError("No current server config found")
        out.api_key = api_key or current_server.api_key
        out.nebu_host = current_server.server
        out.containers_url = f"{out.nebu_host}/v1/containers"

        out = cls.from_v1(container_v1)
        return out

    @classmethod
    def from_v1(cls, v1: V1Container) -> "Container":
        out = cls.__new__(cls)
        out.name = v1.metadata.name
        out.namespace = v1.metadata.namespace
        out.status = v1.status
        out.kind = v1.kind
        out.platform = v1.platform
        out.metadata = v1.metadata
        out.image = v1.image
        out.env = v1.env
        out.command = v1.command
        out.volumes = v1.volumes
        out.accelerators = v1.accelerators
        out.resources = v1.resources
        out.meters = v1.meters
        out.restart = v1.restart
        out.queue = v1.queue
        out.timeout = v1.timeout
        out.ssh_keys = v1.ssh_keys
        return out

    @classmethod
    def search(
        cls,
        params: V1ContainerSearch,
        config: Optional[GlobalConfig] = None,
    ) -> List[V1Container]:
        """
        Search for containers on the remote server.
        """
        config = config or GlobalConfig.read()
        current_server = config.get_current_server_config()
        if not current_server:
            raise ValueError("No current server config found")
        api_key = current_server.api_key
        nebu_host = current_server.server

        search_url = f"{nebu_host}/v1/containers/search"

        response = requests.post(
            search_url,
            headers={"Authorization": f"Bearer {api_key}"},
            json=params.model_dump(),
        )
        response.raise_for_status()

        containers_response = V1Containers.model_validate(response.json())
        return containers_response.containers

    def ref(self) -> V1ResourceReference:
        return V1ResourceReference(
            kind="Container",
            name=self.name,
            namespace=self.namespace,
        )
