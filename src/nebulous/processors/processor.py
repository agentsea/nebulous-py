import json
import threading
import time

# import uuid # Removed unused import
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

import requests
from pydantic import BaseModel

from nebulous.config import GlobalConfig
from nebulous.logging import logger
from nebulous.meta import V1ResourceMetaRequest, V1ResourceReference
from nebulous.processors.models import (
    V1ContainerRequest,
    V1Processor,
    V1ProcessorHealthResponse,
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
        input_model_cls: Optional[type[BaseModel]] = None,
        output_model_cls: Optional[type[BaseModel]] = None,
        common_schema: Optional[str] = None,
        min_replicas: Optional[int] = None,
        max_replicas: Optional[int] = None,
        scale_config: Optional[V1Scale] = None,
        config: Optional[GlobalConfig] = None,
        api_key: Optional[str] = None,
        no_delete: bool = False,
        wait_for_healthy: bool = False,
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
        self.input_model_cls = input_model_cls
        self.output_model_cls = output_model_cls
        self.common_schema = common_schema
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.scale_config = scale_config
        self.processors_url = f"{self.orign_host}/v1/processors"
        self._log_thread: Optional[threading.Thread] = None

        # Infer OutputType Pydantic class if output_model_cls is not provided
        if self.output_model_cls is None and hasattr(self, "__orig_class__"):
            type_args = get_args(self.__orig_class__)  # type: ignore
            if len(type_args) == 2:
                output_type_candidate = type_args[1]
                if isinstance(output_type_candidate, type) and issubclass(
                    output_type_candidate, BaseModel
                ):
                    logger.debug(
                        f"Inferred output_model_cls {output_type_candidate.__name__} from generic arguments."
                    )
                    self.output_model_cls = output_type_candidate
                else:
                    logger.debug(
                        f"Second generic argument {output_type_candidate} is not a Pydantic BaseModel. "
                        "Cannot infer output_model_cls."
                    )
            else:
                logger.debug(
                    "Could not infer output_model_cls from generic arguments: wrong number of type args found "
                    f"(expected 2, got {len(type_args) if type_args else 0})."
                )

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

        # --- Wait for health check if requested ---
        if wait_for_healthy:
            self.wait_for_healthy()

    def __call__(
        self,
        data: InputType,
        wait: bool = False,
        logs: bool = False,
        api_key: Optional[str] = None,
        user_key: Optional[str] = None,
        timeout: Optional[float] = 3600,
        poll: bool = False,
        poll_interval_seconds: float = 2.0,
        stream: bool = False,
    ) -> Any:
        """
        Allows the Processor instance to be called like a function, sending data.
        """
        return self.send(
            data=data,
            wait=wait,
            logs=logs,
            stream=stream,
            api_key=api_key,
            user_key=user_key,
            timeout=timeout,
            poll=poll,
            poll_interval_seconds=poll_interval_seconds,
        )

    def send(
        self,
        data: InputType,
        wait: bool = False,
        logs: bool = False,
        api_key: Optional[str] = None,
        user_key: Optional[str] = None,
        timeout: Optional[float] = 3600,
        poll: bool = False,
        poll_interval_seconds: float = 2.0,
        stream: bool = False,
    ) -> Any:
        """
        Send data to the processor.
        If wait=True, the request to the /messages endpoint waits for the processing to complete.
        If wait=False and poll=True, sends the message and then polls the /return/:message_id endpoint for the result.
        If wait=False and poll=False, sends the message and returns the initial response (e.g., an acknowledgement).
        Optionally streams logs in the background if logs=True.
        """
        logger.debug(
            f"Sending data to processor {self.name}: {data}, wait={wait}, poll={poll}, logs={logs}, stream={stream}"
        )

        if (
            not self.processor
            or not self.processor.metadata.name
            or not self.processor.metadata.namespace
        ):
            raise ValueError("Processor not found or missing metadata (name/namespace)")

        processor_name = self.processor.metadata.name
        processor_namespace = self.processor.metadata.namespace

        # Determine the API key to use for this send operation
        current_op_api_key = api_key if api_key is not None else self.api_key
        if not current_op_api_key:
            logger.error(
                f"Processor {processor_name}: API key is missing for the send operation."
            )
            raise ValueError("API key not available for sending message.")

        # Determine if streaming should be enabled by default for this instance
        # Priority: explicit `stream` argument -> instance attribute `_stream_enabled` -> False
        if not stream and getattr(self, "_stream_enabled", False):
            stream = True

        # --- Send Initial Message ---
        messages_url = (
            f"{self.processors_url}/{processor_namespace}/{processor_name}/messages"
        )

        # The 'wait' parameter for V1StreamData dictates if the /messages endpoint itself should block.
        stream_data_wait_param = wait

        stream_data = V1StreamData(
            content=data,
            wait=stream_data_wait_param,
            user_key=user_key,
        )

        # Timeout for the initial POST request.
        # If stream_data_wait_param is True, use the overall timeout.
        # Otherwise (quick ack expected), use a shorter fixed timeout.
        initial_request_timeout = timeout if stream_data_wait_param else 30.0

        logger.debug(
            f"Processor {processor_name}: Posting to {messages_url} with stream_data_wait={stream_data_wait_param}, initial_timeout={initial_request_timeout}"
        )
        response = requests.post(
            messages_url,
            json=stream_data.model_dump(mode="json", exclude_none=True),
            headers={"Authorization": f"Bearer {current_op_api_key}"},
            timeout=initial_request_timeout,
        )
        response.raise_for_status()
        raw_response_json = response.json()
        logger.debug(
            f"Processor {processor_name}: Initial response JSON: {raw_response_json}"
        )

        if "error" in raw_response_json:
            logger.error(
                f"Processor {processor_name}: Error in initial response: {raw_response_json['error']}"
            )
            raise Exception(raw_response_json["error"])

        # Initialize raw_content. This will hold the final data payload.
        raw_content: Optional[Union[Dict[str, Any], List[Any], str]] = None

        # --- Handle Response: Streaming or Polling ---
        if stream:
            # --- Streaming / Generator behaviour ---
            if wait:
                logger.warning(
                    "Argument 'wait' is ignored when 'stream' is True. Proceeding with streaming behaviour (non-blocking)."
                )
            if poll:
                logger.warning(
                    "Argument 'poll' is ignored when 'stream' is True. Streaming implicitly handles polling internally."
                )

            message_id = raw_response_json.get("message_id")
            return_stream = raw_response_json.get("return_stream")

            if not (isinstance(message_id, str) and isinstance(return_stream, str)):
                raise ValueError(
                    "Streaming requested but server did not return 'message_id' and 'return_stream' in response."
                )

            polling_url = f"{self.orign_host}/v1/processors/{processor_namespace}/{processor_name}/return/{message_id}"

            poll_payload = {"consumer_group": return_stream}

            def _result_generator() -> (
                Generator[OutputType | Dict[str, Any] | None, None, None]
            ):
                last_content: Any = None
                start_ts = time.time()
                while True:
                    # timeout handling
                    if timeout is not None and (time.time() - start_ts) > timeout:
                        logger.warning(
                            f"Processor {processor_name}: Streaming generator timed out after {timeout}s while waiting for message {message_id}."
                        )
                        return  # Stop generator

                    try:
                        poll_response = requests.post(
                            polling_url,
                            headers={"Authorization": f"Bearer {current_op_api_key}"},
                            timeout=max(10.0, poll_interval_seconds * 2),
                            json=poll_payload,
                        )

                        status = poll_response.status_code

                        if status == 200:
                            data_json = poll_response.json()
                            content = data_json.get("content", data_json)

                            parsed = content
                            if (
                                self.output_model_cls
                                and isinstance(content, dict)
                                and isinstance(self.output_model_cls, type)
                            ):
                                try:
                                    parsed = self.output_model_cls.model_validate(
                                        content
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Processor {processor_name}: Failed to parse streaming content into model – yielding raw content. Error: {e}"
                                    )

                            yield parsed  # type: ignore[yield-value]
                            return  # Final result delivered, stop generator

                        elif status in (202, 404):
                            # 202 Processing or 404 Not ready yet – optionally yield progress
                            try:
                                data_json = poll_response.json()
                                progress_content = data_json.get("content")
                                if (
                                    progress_content is not None
                                    and progress_content != last_content
                                ):
                                    last_content = progress_content
                                    yield progress_content  # type: ignore[yield-value]
                            except Exception:
                                pass
                            time.sleep(poll_interval_seconds)
                        else:
                            poll_response.raise_for_status()
                    except requests.exceptions.Timeout:
                        logger.debug(
                            f"Processor {processor_name}: Streaming poll timed out – retrying."
                        )
                    except Exception as e:
                        logger.error(
                            f"Processor {processor_name}: Error during streaming poll: {e}"
                        )
                        return  # Abort generator on unrecoverable error

            return _result_generator()
        elif poll and not stream_data_wait_param:
            # --- Traditional polling behaviour (wait for final response) ---
            message_id = raw_response_json.get("message_id")
            return_stream = raw_response_json.get("return_stream")

            if not (isinstance(message_id, str) and isinstance(return_stream, str)):
                logger.error(
                    f"Processor {processor_name}: Polling requested but 'message_id' or 'return_stream' missing/invalid in initial response. Response: {raw_response_json}"
                )
                raise ValueError(
                    "Polling failed: message_id or return_stream missing in server response."
                )

            polling_url = f"{self.orign_host}/v1/processors/{processor_namespace}/{processor_name}/return/{message_id}"
            poll_payload = {"consumer_group": return_stream}

            logger.info(
                f"Processor {processor_name}: Starting polling loop for {message_id} every {poll_interval_seconds}s (timeout={timeout})."
            )
            start_ts = time.time()
            while True:
                if timeout is not None and (time.time() - start_ts) > timeout:
                    raise TimeoutError(
                        f"Polling for message_id {message_id} timed out after {timeout} seconds."
                    )

                try:
                    poll_resp = requests.post(
                        polling_url,
                        headers={"Authorization": f"Bearer {current_op_api_key}"},
                        timeout=max(10.0, poll_interval_seconds * 2),
                        json=poll_payload,
                    )

                    status = poll_resp.status_code
                    if status == 200:
                        try:
                            polled_json = poll_resp.json()
                        except ValueError:
                            logger.error(
                                f"Processor {processor_name}: Poll response for {message_id} not JSON. Text: {poll_resp.text[:200]}"
                            )
                            raise

                        raw_content = (
                            polled_json.get("content")
                            if isinstance(polled_json, dict)
                            else polled_json
                        )
                        logger.info(
                            f"Processor {processor_name}: Polling completed for {message_id}."
                        )
                        break
                    elif status in (202, 404):
                        logger.debug(
                            f"Processor {processor_name}: Message {message_id} not ready (status {status}). Sleeping {poll_interval_seconds}s."
                        )
                        time.sleep(poll_interval_seconds)
                        continue
                    else:
                        poll_resp.raise_for_status()
                except requests.exceptions.Timeout:
                    logger.debug(
                        f"Processor {processor_name}: Poll request timed out, retrying..."
                    )
                except Exception as poll_err:
                    logger.error(
                        f"Processor {processor_name}: Error during polling for {message_id}: {poll_err}"
                    )
                    raise

        else:
            # Handles: wait=True (blocking request) OR simple fire-and-forget
            raw_content = raw_response_json.get("content")
            logger.debug(
                f"Processor {processor_name}: No polling. Raw content from initial response: {str(raw_content)[:200]}..."
            )

        # --- Fetch Logs (if requested and not already running) ---
        if logs:
            if self._log_thread is None or not self._log_thread.is_alive():
                log_url = (
                    f"{self.processors_url}/{processor_namespace}/{processor_name}/logs"
                )
                self._log_thread = threading.Thread(
                    target=_fetch_and_print_logs,
                    args=(
                        log_url,
                        self.api_key,
                        processor_name,
                    ),  # Use self.api_key for logs
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

        # --- Attempt to parse into OutputType if conditions are met ---
        logger.debug(
            f"Processor {processor_name}: Attempting to parse result. output_model_cls: {self.output_model_cls}, raw_content type: {type(raw_content)}"
        )

        # Attempt to parse if the operation was intended to yield full content (either by waiting or polling),
        # and raw_content is a dictionary, and output_model_cls is a Pydantic model.
        should_attempt_parse = (
            wait or poll
        )  # True if client expects full content back from this method call

        if (
            should_attempt_parse
            and self.output_model_cls is not None
            and isinstance(self.output_model_cls, type)
            and isinstance(raw_content, dict)
        ):
            logger.debug(
                f"Processor {processor_name}: Valid conditions for parsing. Raw content (dict): {str(raw_content)[:200]}..."
            )
            try:
                parsed_model = self.output_model_cls.model_validate(raw_content)
                logger.debug(
                    f"Processor {processor_name}: Successfully parsed to {self.output_model_cls.__name__}. Parsed model: {str(parsed_model)[:200]}..."
                )
                parsed_output: OutputType = cast(OutputType, parsed_model)
                return parsed_output
            except Exception as e:
                model_name = getattr(
                    self.output_model_cls, "__name__", str(self.output_model_cls)
                )
                logger.error(
                    f"Processor {processor_name}: Failed to parse 'content' field into output type {model_name}. "
                    f"Error: {e}. Raw content was: {str(raw_content)[:500]}. Returning raw content instead."
                )
                return raw_content  # type: ignore
        else:
            if (
                not isinstance(raw_content, dict)
                and should_attempt_parse
                and self.output_model_cls
            ):
                logger.debug(
                    f"Processor {processor_name}: Skipping Pydantic parsing because raw_content is not a dict (type: {type(raw_content)})."
                )
            elif not (should_attempt_parse and self.output_model_cls):
                logger.debug(
                    f"Processor {processor_name}: Skipping Pydantic parsing due to conditions not met (should_attempt_parse: {should_attempt_parse}, output_model_cls: {self.output_model_cls is not None})."
                )

        logger.debug(
            f"Processor {processor_name}: Returning raw_content (type: {type(raw_content)}): {str(raw_content)[:200]}..."
        )
        return raw_content  # type: ignore

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
        input_model_cls: Optional[type[BaseModel]] = None,
        output_model_cls: Optional[type[BaseModel]] = None,
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

        # Try to infer Input/Output model classes if Processor.load is called as generic
        # e.g., MyProcessor = Processor[MyInput, MyOutput]; MyProcessor.load(...)
        loaded_input_model_cls: Optional[type[BaseModel]] = None
        loaded_output_model_cls: Optional[type[BaseModel]] = None

        # __orig_bases__ usually contains the generic version of the class if it was parameterized.
        # We look for Processor[...] in the bases.
        if hasattr(cls, "__orig_bases__"):
            for base in cls.__orig_bases__:  # type: ignore
                if get_origin(base) is Processor:
                    type_args = get_args(base)
                    if len(type_args) == 2:
                        input_arg, output_arg = type_args
                        if isinstance(input_arg, type) and issubclass(
                            input_arg, BaseModel
                        ):
                            loaded_input_model_cls = input_arg
                        if isinstance(output_arg, type) and issubclass(
                            output_arg, BaseModel
                        ):
                            loaded_output_model_cls = output_arg
                    break  # Found Processor generic base

        # Determine final model classes, prioritizing overrides
        final_input_model_cls = (
            input_model_cls if input_model_cls is not None else loaded_input_model_cls
        )
        final_output_model_cls = (
            output_model_cls
            if output_model_cls is not None
            else loaded_output_model_cls
        )

        out = cls.__new__(cls)  # type: ignore
        # If generic types were successfully inferred or overridden, pass them to init
        # Otherwise, they will be None, and __init__ might try __orig_class__ if called on instance
        out.__init__(  # type: ignore
            name=processor_v1.metadata.name,  # Use name from fetched metadata
            namespace=processor_v1.metadata.namespace,  # Use namespace from fetched metadata
            labels=processor_v1.metadata.labels,  # Use labels from fetched metadata
            container=processor_v1.container,
            input_model_cls=final_input_model_cls,  # Use determined input model
            output_model_cls=final_output_model_cls,  # Use determined output model
            common_schema=processor_v1.common_schema,
            min_replicas=processor_v1.min_replicas,
            max_replicas=processor_v1.max_replicas,
            scale_config=processor_v1.scale,
            config=config,  # Pass original config
            api_key=api_key,  # Pass original api_key
        )
        # The __init__ call above handles most setup. We store the fetched processor data.
        out.processor = processor_v1
        # self.schema_ was removed, so no assignment for it here from processor_v1.schema_
        # out.common_schema = processor_v1.common_schema # This is now set in __init__

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

    def wait_for_healthy(
        self, timeout: float = 3600.0, retry_interval: float = 5.0
    ) -> None:
        """
        Wait for the processor to respond to health checks using the health endpoint.

        Args:
            timeout: Maximum time to wait for health check in seconds
            retry_interval: Time between health check attempts in seconds
        """
        if not self.processor or not self.processor.metadata.name:
            raise ValueError("Processor not found, cannot perform health check")

        logger.info(
            f"Waiting for processor {self.processor.metadata.name} to be healthy via health endpoint..."
        )

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                health_response = self.health()  # Use the new health() method
                logger.info(
                    f">>> Health check response: {health_response.model_dump_json()}"
                )

                # Check if the response indicates health
                if health_response.status == "ok":  # Check for "ok" status
                    logger.info(f"Processor {self.processor.metadata.name} is healthy!")
                    return
                else:
                    logger.info(
                        f"Processor {self.processor.metadata.name} reported status: {health_response.status}. Retrying in {retry_interval}s..."
                    )

            except Exception as e:
                logger.info(
                    f"Health check failed with error: {e}, retrying in {retry_interval}s..."
                )

            time.sleep(retry_interval)

        # If we get here, we timed out
        raise TimeoutError(
            f"Processor {self.processor.metadata.name} failed to become healthy within {timeout} seconds"
        )

    def health(self) -> V1ProcessorHealthResponse:
        """
        Performs a health check on the processor by calling the health endpoint.
        """
        if (
            not self.processor
            or not self.processor.metadata.name
            or not self.processor.metadata.namespace
        ):
            raise ValueError(
                "Processor not found or missing metadata (name/namespace), cannot perform health check."
            )

        health_url = f"{self.orign_host}/v1/processors/{self.processor.metadata.namespace}/{self.processor.metadata.name}/health"
        logger.debug(f"Calling health check endpoint: {health_url}")

        try:
            response = requests.get(
                health_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,  # Standard timeout for a health check
            )
            response.raise_for_status()  # Raise an exception for HTTP errors
            health_response_data = response.json()
            return V1ProcessorHealthResponse.model_validate(health_response_data)
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check request to {health_url} failed: {e}")
            # Optionally, return a V1ProcessorHealthResponse indicating an error
            # For now, re-raising the exception or a custom one might be better
            raise RuntimeError(f"Failed to get health status: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during health check: {e}")
            raise RuntimeError(
                f"Unexpected error during health status retrieval: {e}"
            ) from e
