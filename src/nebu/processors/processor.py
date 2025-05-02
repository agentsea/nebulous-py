import json
import threading
from typing import Any, Dict, Generic, List, Optional, TypeVar

import requests
from pydantic import BaseModel

from nebu.config import GlobalConfig
from nebu.logging import logger
from nebu.meta import V1ResourceMetaRequest, V1ResourceReference
from nebu.processors.models import (
    V1ContainerRequest,
    V1Processor,
    V1ProcessorRequest,
    V1Processors,
    V1ProcessorScaleRequest,
    V1Scale,
    V1StreamData,
    V1UpdateProcessor,
)

InputType = TypeVar("InputType", bound=BaseModel)
OutputType = TypeVar("OutputType", bound=BaseModel)


def _fetch_and_print_logs(log_url: str, api_key: str, processor_name: str):
    """Helper function to fetch logs in a separate thread."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        logger.info(
            f"--- Attempting to stream logs for {processor_name} from {log_url} ---"
        )
        # Use stream=True for potentially long-lived connections and timeout
        with requests.get(log_url, headers=headers, stream=True, timeout=300) as r:
            r.raise_for_status()
            logger.info(f"--- Streaming logs for {processor_name} ---")
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    # Decode bytes to string
                    decoded_line = line.decode("utf-8")
                    # Parse the JSON line
                    log_data = json.loads(decoded_line)

                    # Check if the parsed data is a dictionary (expected format)
                    if isinstance(log_data, dict):
                        for container, log_content in log_data.items():
                            # Ensure log_content is a string before printing
                            if isinstance(log_content, str):
                                logger.info(
                                    f"[{processor_name}][{container}] {log_content}"
                                )
                            else:
                                # Handle cases where log_content might not be a string
                                logger.warning(
                                    f"[{processor_name}][{container}] Unexpected log format: {log_content}"
                                )
                    else:
                        # If not a dict, print the raw line with a warning
                        logger.warning(
                            f"[{processor_name}] Unexpected log structure (not a dict): {decoded_line}"
                        )

                except json.JSONDecodeError:
                    # If JSON parsing fails, print the original line as fallback
                    logger.warning(
                        f"[{processor_name}] {line.decode('utf-8')} (raw/non-JSON)"
                    )
                except Exception as e:
                    # Catch other potential errors during line processing
                    logger.error(f"Error processing log line for {processor_name}: {e}")

        logger.info(f"--- Log stream ended for {processor_name} ---")
    except requests.exceptions.Timeout:
        logger.warning(f"Log stream connection timed out for {processor_name}.")
    except requests.exceptions.RequestException as e:
        # Handle potential API errors gracefully
        logger.error(f"Error fetching logs for {processor_name} from {log_url}: {e}")
        if e.response is not None:
            # Log response details at a debug level or keep as error if critical
            logger.error(
                f"Response status: {e.response.status_code}, Response body: {e.response.text}"
            )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while fetching logs for {processor_name}: {e}"
        )


class Processor(Generic[InputType, OutputType]):
    """
    A class for managing Processor instances.
    """

    def __init__(
        self,
        name: str,
        namespace: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        container: Optional[V1ContainerRequest] = None,
        schema_: Optional[Any] = None,
        common_schema: Optional[str] = None,
        min_replicas: Optional[int] = None,
        max_replicas: Optional[int] = None,
        scale_config: Optional[V1Scale] = None,
        config: Optional[GlobalConfig] = None,
        api_key: Optional[str] = None,
        no_delete: bool = False,
    ):
        self.config = config or GlobalConfig.read()
        if not self.config:
            raise ValueError("No config found")
        current_server = self.config.get_current_server_config()
        if not current_server:
            raise ValueError("No server config found")
        self.current_server = current_server
        self.api_key = api_key or current_server.api_key
        self.orign_host = current_server.server
        self.name = name
        self.namespace = namespace
        self.labels = labels
        self.container = container
        self.schema_ = schema_
        self.common_schema = common_schema
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.scale_config = scale_config
        self.processors_url = f"{self.orign_host}/v1/processors"
        self._log_thread: Optional[threading.Thread] = None

        # Fetch existing Processors
        response = requests.get(
            self.processors_url, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        response.raise_for_status()

        if not namespace:
            namespace = "-"

        logger.info(f"Using namespace: {namespace}")

        existing_processors = V1Processors.model_validate(response.json())
        logger.debug(f"Existing processors: {existing_processors}")
        self.processor: Optional[V1Processor] = next(
            (
                processor_val
                for processor_val in existing_processors.processors
                if processor_val.metadata.name == name
                and processor_val.metadata.namespace == namespace
            ),
            None,
        )
        logger.debug(f"Processor: {self.processor}")

        # If not found, create
        if not self.processor:
            logger.info("Creating processor")
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

            logger.debug("Request:")
            logger.debug(processor_request.model_dump(exclude_none=True))
            create_response = requests.post(
                self.processors_url,
                json=processor_request.model_dump(exclude_none=True),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            create_response.raise_for_status()
            self.processor = V1Processor.model_validate(create_response.json())
            logger.info(f"Created Processor {self.processor.metadata.name}")
        else:
            # Else, update
            logger.info(
                f"Found Processor {self.processor.metadata.name}, updating if necessary"
            )

            update_processor = V1UpdateProcessor(
                container=container,
                schema_=schema_,
                common_schema=common_schema,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                scale=scale_config,
                no_delete=no_delete,
            )

            logger.debug("Update request:")
            logger.debug(update_processor.model_dump(exclude_none=True))
            patch_response = requests.patch(
                f"{self.processors_url}/{self.processor.metadata.namespace}/{self.processor.metadata.name}",
                json=update_processor.model_dump(exclude_none=True),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            patch_response.raise_for_status()
            self.processor = V1Processor.model_validate(patch_response.json())
            logger.info(f"Updated Processor {self.processor.metadata.name}")

    def __call__(
        self,
        data: InputType,
        wait: bool = False,
        logs: bool = False,
        api_key: Optional[str] = None,
        user_key: Optional[str] = None,
    ) -> OutputType | Dict[str, Any] | None:
        """
        Allows the Processor instance to be called like a function, sending data.
        """
        return self.send(
            data=data, wait=wait, logs=logs, api_key=api_key, user_key=user_key
        )

    def send(
        self,
        data: InputType,
        wait: bool = False,
        logs: bool = False,
        api_key: Optional[str] = None,
        user_key: Optional[str] = None,
    ) -> OutputType | Dict[str, Any] | None:
        """
        Send data to the processor and optionally stream logs in the background.
        """
        if (
            not self.processor
            or not self.processor.metadata.name
            or not self.processor.metadata.namespace
        ):
            raise ValueError("Processor not found or missing metadata (name/namespace)")

        processor_name = self.processor.metadata.name
        processor_namespace = self.processor.metadata.namespace

        if not api_key:
            api_key = self.api_key

        # --- Send Data ---
        messages_url = (
            f"{self.processors_url}/{processor_namespace}/{processor_name}/messages"
        )
        stream_data = V1StreamData(
            content=data,
            wait=wait,
            user_key=user_key,
        )
        response = requests.post(
            messages_url,
            json=stream_data.model_dump(mode="json", exclude_none=True),
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        send_response_json = response.json()

        # --- Fetch Logs (if requested and not already running) ---
        if logs:
            if self._log_thread is None or not self._log_thread.is_alive():
                log_url = (
                    f"{self.processors_url}/{processor_namespace}/{processor_name}/logs"
                )
                self._log_thread = threading.Thread(
                    target=_fetch_and_print_logs,
                    args=(log_url, self.api_key, processor_name),  # Pass processor_name
                    daemon=True,
                )
                try:
                    self._log_thread.start()
                    logger.info(
                        f"Started background log fetching for {processor_name}..."
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to start log fetching thread for {processor_name}: {e}"
                    )
                    self._log_thread = None  # Reset if start fails
            else:
                logger.info(f"Log fetching is already running for {processor_name}.")

        return send_response_json

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
        api_key: Optional[str] = None,
    ):
        """
        Get a Processor from the remote server.
        """
        processors = cls.get(
            namespace=namespace, name=name, config=config, api_key=api_key
        )
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
        out.api_key = api_key or out.current_server.api_key
        out.orign_host = out.current_server.server
        out.processors_url = f"{out.orign_host}/v1/processors"
        out.name = name
        out.namespace = namespace

        # Set specific fields from the processor
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
        api_key: Optional[str] = None,
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
            headers={"Authorization": f"Bearer {api_key or current_server.api_key}"},
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

    def ref(self) -> V1ResourceReference:
        """
        Get the resource ref for the processor.
        """
        if not self.processor:
            raise ValueError("Processor not found")
        return V1ResourceReference(
            name=self.processor.metadata.name,
            namespace=self.processor.metadata.namespace,
            kind="Processor",
        )

    def stop_logs(self):
        """
        Signals the intent to stop the background log stream.
        Note: Interrupting a streaming requests.get cleanly can be complex.
              This currently allows a new log stream to be started on the next call.
        """
        if self._log_thread and self._log_thread.is_alive():
            # Attempting to stop a daemon thread directly isn't standard practice.
            # Setting the reference to None allows a new thread to be created if needed.
            # The OS will eventually clean up the daemon thread when the main process exits,
            # or potentially sooner if the network request completes or errors out.
            logger.info(
                f"Disassociating from active log stream for {self.name}. A new stream can be started."
            )
            self._log_thread = None
        else:
            logger.info(f"No active log stream to stop for {self.name}.")
