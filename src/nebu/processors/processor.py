from typing import Any, Dict, List, Optional

import requests

from nebu.auth import get_user_profile
from nebu.config import GlobalConfig
from nebu.meta import V1ResourceMetaRequest
from nebu.processors.models import (
    V1ContainerRequest,
    V1Processor,
    V1ProcessorRequest,
    V1Processors,
    V1ProcessorScaleRequest,
    V1Scale,
    V1UpdateProcessor,
)


class Processor:
    """
    A class for managing Processor instances.
    """

    def __init__(
        self,
        name: str,
        stream: str,
        namespace: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        container: Optional[V1ContainerRequest] = None,
        schema_: Optional[Any] = None,
        common_schema: Optional[str] = None,
        min_replicas: Optional[int] = None,
        max_replicas: Optional[int] = None,
        scale_config: Optional[V1Scale] = None,
        config: Optional[GlobalConfig] = None,
        no_delete: bool = False,
    ):
        self.config = config or GlobalConfig.read()
        if not self.config:
            raise ValueError("No config found")
        current_server = self.config.get_current_server_config()
        if not current_server:
            raise ValueError("No server config found")
        self.current_server = current_server
        self.api_key = current_server.api_key
        self.orign_host = current_server.server
        self.name = name
        self.namespace = namespace
        self.labels = labels
        self.stream = stream
        self.container = container
        self.schema_ = schema_
        self.common_schema = common_schema
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.scale_config = scale_config
        self.processors_url = f"{self.orign_host}/v1/processors"

        # Fetch existing Processors
        response = requests.get(
            self.processors_url, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        response.raise_for_status()

        if not namespace:
            if not self.api_key:
                raise ValueError("No API key provided")

            user_profile = get_user_profile(self.api_key)
            namespace = user_profile.handle

            if not namespace:
                namespace = user_profile.email.replace("@", "-").replace(".", "-")

        print(f"Using namespace: {namespace}")

        existing_processors = V1Processors.model_validate(response.json())
        print(f"Existing processors: {existing_processors}")
        self.processor: Optional[V1Processor] = next(
            (
                processor_val
                for processor_val in existing_processors.processors
                if processor_val.metadata.name == name
                and processor_val.metadata.namespace == namespace
            ),
            None,
        )
        print(f"Processor: {self.processor}")

        # If not found, create
        if not self.processor:
            print("Creating processor")
            # Create metadata and processor request
            metadata = V1ResourceMetaRequest(
                name=name, namespace=namespace, labels=labels
            )

            processor_request = V1ProcessorRequest(
                metadata=metadata,
                container=container,
                schema_=schema_,
                common_schema=common_schema,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                scale=scale_config,
            )

            print("Request:")
            print(processor_request.model_dump(exclude_none=True))
            create_response = requests.post(
                self.processors_url,
                json=processor_request.model_dump(exclude_none=True),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            create_response.raise_for_status()
            self.processor = V1Processor.model_validate(create_response.json())
            print(f"Created Processor {self.processor.metadata.name}")
        else:
            # Else, update
            print(
                f"Found Processor {self.processor.metadata.name}, updating if necessary"
            )

            update_processor = V1UpdateProcessor(
                stream=stream,
                container=container,
                schema_=schema_,
                common_schema=common_schema,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                scale=scale_config,
                no_delete=no_delete,
            )

            print("Update request:")
            print(update_processor.model_dump(exclude_none=True))
            patch_response = requests.patch(
                f"{self.processors_url}/{self.processor.metadata.namespace}/{self.processor.metadata.name}",
                json=update_processor.model_dump(exclude_none=True),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            patch_response.raise_for_status()
            print(f"Updated Processor {self.processor.metadata.name}")

    def send(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send data to the processor.
        """
        if not self.processor or not self.processor.metadata.name:
            raise ValueError("Processor not found")

        url = f"{self.processors_url}/{self.processor.metadata.namespace}/{self.processor.metadata.name}/send"

        response = requests.get(
            url,
            params=data,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        return response.json()

    def scale(self, replicas: int) -> Dict[str, Any]:
        """
        Scale the processor.
        """
        if not self.processor or not self.processor.metadata.name:
            raise ValueError("Processor not found")

        url = f"{self.processors_url}/{self.processor.metadata.namespace}/{self.processor.metadata.name}/scale"
        scale_request = V1ProcessorScaleRequest(replicas=replicas)

        response = requests.post(
            url,
            json=scale_request.model_dump(exclude_none=True),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        return response.json()

    @classmethod
    def load(
        cls,
        name: str,
        namespace: Optional[str] = None,
        config: Optional[GlobalConfig] = None,
    ):
        """
        Get a Processor from the remote server.
        """
        processors = cls.get(namespace=namespace, name=name, config=config)
        if not processors:
            raise ValueError("Processor not found")
        processor_v1 = processors[0]

        out = cls.__new__(cls)
        out.processor = processor_v1
        out.config = config or GlobalConfig.read()
        if not out.config:
            raise ValueError("No config found")
        out.current_server = out.config.get_current_server_config()
        if not out.current_server:
            raise ValueError("No server config found")
        out.api_key = out.current_server.api_key
        out.orign_host = out.current_server.server
        out.processors_url = f"{out.orign_host}/v1/processors"
        out.name = name
        out.namespace = namespace

        # Set specific fields from the processor
        out.stream = processor_v1.stream
        out.container = processor_v1.container
        out.schema_ = processor_v1.schema_
        out.common_schema = processor_v1.common_schema
        out.min_replicas = processor_v1.min_replicas
        out.max_replicas = processor_v1.max_replicas
        out.scale_config = processor_v1.scale

        return out

    @classmethod
    def get(
        cls,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        config: Optional[GlobalConfig] = None,
    ) -> List[V1Processor]:
        """
        Get a list of Processors that match the optional name and/or namespace filters.
        """
        config = config or GlobalConfig.read()
        if not config:
            raise ValueError("No config found")
        current_server = config.get_current_server_config()
        if not current_server:
            raise ValueError("No server config found")
        processors_url = f"{current_server.server}/v1/processors"

        response = requests.get(
            processors_url,
            headers={"Authorization": f"Bearer {current_server.api_key}"},
        )
        response.raise_for_status()

        processors_response = V1Processors.model_validate(response.json())
        filtered_processors = processors_response.processors

        if name:
            filtered_processors = [
                p for p in filtered_processors if p.metadata.name == name
            ]
        if namespace:
            filtered_processors = [
                p for p in filtered_processors if p.metadata.namespace == namespace
            ]

        return filtered_processors

    def delete(self):
        """
        Delete the Processor.
        """
        if not self.processor or not self.processor.metadata.name:
            raise ValueError("Processor not found")

        url = f"{self.processors_url}/{self.processor.metadata.namespace}/{self.processor.metadata.name}"
        response = requests.delete(
            url, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        response.raise_for_status()
        return

    def ref(self) -> str:
        """
        Get the resource ref for the processor.
        """
        return f"{self.name}.{self.namespace}.Processor"
