#!/usr/bin/env python3
import importlib
import json
import os
import socket
import sys

# import time # Removed unused import
import traceback
import types
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

import redis
import socks

from nebulous.errors import RetriableError
from nebulous.logging import logger  # Import the logger

# from redis import ConnectionError, ResponseError # Removed unused imports

# Define TypeVar for generic models
T = TypeVar("T")

# Environment variable name used as a guard in the decorator
_NEBU_INSIDE_CONSUMER_ENV_VAR = "_NEBU_INSIDE_CONSUMER_EXEC"


# --- Global variables for dynamically loaded code (in this process) ---
target_function: Optional[Callable] = None
init_function: Optional[Callable] = None
imported_module: Optional[types.ModuleType] = None
local_namespace: Dict[str, Any] = {}  # Namespace for included objects
last_load_mtime: float = (
    0.0  # Note: This worker doesn't auto-reload code. It loads once.
)
entrypoint_abs_path: Optional[str] = None


# --- Function to Load User Code (Copied from consumer.py, no reload needed) ---
def load_user_code(
    module_path: str,
    function_name: str,
    entrypoint_abs_path: str,
    init_func_name: Optional[str] = None,
    included_object_sources: Optional[List[Tuple[str, List[str]]]] = None,
) -> Tuple[
    Optional[Callable],
    Optional[Callable],
    Optional[types.ModuleType],
    Dict[str, Any],
]:
    """Loads the user code module, executes includes, and returns functions/module."""
    global _NEBU_INSIDE_CONSUMER_ENV_VAR  # Access the global guard var name

    loaded_target_func = None
    loaded_init_func = None
    loaded_module = None
    exec_namespace: Dict[str, Any] = {}  # Use a local namespace for this load attempt

    logger.info(
        f"[Worker Code Loader] Attempting to load module: '{module_path}'"
    )  # Changed from print
    os.environ[_NEBU_INSIDE_CONSUMER_ENV_VAR] = "1"  # Set guard *before* import
    logger.debug(  # Changed from print to debug
        f"[Worker Code Loader] Set environment variable {_NEBU_INSIDE_CONSUMER_ENV_VAR}=1"
    )

    try:
        # Execute included object sources FIRST (if any)
        if included_object_sources:
            logger.debug(
                "[Worker Code Loader] Executing @include object sources..."
            )  # Changed from print to debug
            # Include necessary imports for the exec context
            exec("from pydantic import BaseModel, Field", exec_namespace)
            exec(
                "from typing import Optional, List, Dict, Any, Generic, TypeVar",
                exec_namespace,
            )
            exec("T_exec = TypeVar('T_exec')", exec_namespace)
            exec("from nebulous.processors.models import *", exec_namespace)
            # ... add other common imports if needed by included objects ...

            for i, (obj_source, args_sources) in enumerate(included_object_sources):
                try:
                    exec(obj_source, exec_namespace)
                    logger.debug(  # Changed from print to debug
                        f"[Worker Code Loader] Successfully executed included object {i} base source"
                    )
                    for j, arg_source in enumerate(args_sources):
                        try:
                            exec(arg_source, exec_namespace)
                            logger.debug(  # Changed from print to debug
                                f"[Worker Code Loader] Successfully executed included object {i} arg {j} source"
                            )
                        except Exception as e_arg:
                            logger.error(  # Changed from print to error
                                f"Error executing included object {i} arg {j} source: {e_arg}"
                            )
                            logger.exception(
                                f"Included object {i} arg {j} source error:"
                            )  # Replaced traceback.print_exc()
                except Exception as e_base:
                    logger.error(
                        f"Error executing included object {i} base source: {e_base}"
                    )  # Changed from print to error
                    logger.exception(
                        f"Included object {i} base source error:"
                    )  # Replaced traceback.print_exc()
            logger.debug(
                "[Worker Code Loader] Finished executing included object sources."
            )  # Changed from print to debug

        # Import the main module (no reload needed in worker)
        loaded_module = importlib.import_module(module_path)
        logger.info(
            f"[Worker Code Loader] Successfully imported module: {module_path}"
        )  # Changed from print

        # Get the target function from the loaded module
        loaded_target_func = getattr(loaded_module, function_name)
        logger.info(  # Changed from print
            f"[Worker Code Loader] Successfully loaded function '{function_name}' from module '{module_path}'"
        )

        # Get the init function if specified
        if init_func_name:
            loaded_init_func = getattr(loaded_module, init_func_name)
            logger.info(  # Changed from print
                f"[Worker Code Loader] Successfully loaded init function '{init_func_name}' from module '{module_path}'"
            )
            # Execute init_func
            logger.info(
                f"[Worker Code Loader] Executing init_func: {init_func_name}..."
            )  # Changed from print
            loaded_init_func()  # Call the function
            logger.info(  # Changed from print
                f"[Worker Code Loader] Successfully executed init_func: {init_func_name}"
            )

        logger.info("[Worker Code Loader] Code load successful.")  # Changed from print
        return (
            loaded_target_func,
            loaded_init_func,
            loaded_module,
            exec_namespace,
        )

    except FileNotFoundError:
        logger.error(  # Changed from print to error
            f"[Worker Code Loader] Error: Entrypoint file not found at '{entrypoint_abs_path}'. Cannot load."
        )
        return None, None, None, {}  # Indicate failure
    except ImportError as e:
        logger.error(
            f"[Worker Code Loader] Error importing module '{module_path}': {e}"
        )  # Changed from print to error
        logger.exception("Module import error:")  # Replaced traceback.print_exc()
        return None, None, None, {}  # Indicate failure
    except AttributeError as e:
        logger.error(  # Changed from print to error
            f"[Worker Code Loader] Error accessing function '{function_name}' or '{init_func_name}' in module '{module_path}': {e}"
        )
        logger.exception("Function access error:")  # Replaced traceback.print_exc()
        return None, None, None, {}  # Indicate failure
    except Exception as e:
        logger.error(
            f"[Worker Code Loader] Unexpected error during code load: {e}"
        )  # Changed from print to error
        logger.exception(
            "Unexpected code load error:"
        )  # Replaced traceback.print_exc()
        return None, None, None, {}  # Indicate failure
    finally:
        # Unset the guard environment variable
        os.environ.pop(_NEBU_INSIDE_CONSUMER_ENV_VAR, None)
        logger.debug(  # Changed from print to debug
            f"[Worker Code Loader] Unset environment variable {_NEBU_INSIDE_CONSUMER_ENV_VAR}"
        )


# --- Helper to Send Error Response (Copied from consumer.py) ---
# Note: 'r' and 'REDIS_STREAM' will be global in this worker's context
def _send_error_response(
    message_id: str,
    error_msg: str,
    tb: str,
    return_stream: Optional[str],
    user_id: Optional[str],
):
    """Sends a standardized error response to Redis."""
    global r, redis_stream  # Use lowercase redis_stream

    # Check if Redis connection exists before trying to use it
    if r is None:
        logger.critical(  # Changed from print to critical
            "[Worker] CRITICAL: Cannot send error response, Redis connection is not available."
        )
        return
    # Assert REDIS_STREAM type here for safety, although it should be set if r is available
    if not isinstance(redis_stream, str):
        logger.critical(  # Changed from print to critical
            "[Worker] CRITICAL: Cannot send error response, REDIS_STREAM is not a valid string."
        )
        return

    error_response = {
        "kind": "StreamResponseMessage",
        "id": message_id,
        "content": {
            "error": error_msg,
            "traceback": tb,
        },
        "status": "error",
        "created_at": datetime.now(timezone.utc).isoformat(),  # Use UTC
        "user_id": user_id,
    }

    error_destination = f"{redis_stream}.errors"  # Default error stream
    if return_stream:  # Prefer return_stream if available
        error_destination = return_stream

    try:
        assert isinstance(error_destination, str)
        r.xadd(error_destination, {"data": json.dumps(error_response)})
        logger.info(  # Changed from print
            f"[Worker] Sent error response for message {message_id} to {error_destination}"
        )
    except Exception as e_redis:
        logger.critical(  # Changed from print to critical
            f"[Worker] CRITICAL: Failed to send error response for {message_id} to Redis: {e_redis}"
        )
        logger.exception(
            "Error sending response to Redis:"
        )  # Replaced traceback.print_exc()


# --- Main Worker Logic ---
if __name__ == "__main__":
    logger.info("[Worker] Starting subprocess worker...")  # Changed from print
    r: Optional[redis.Redis] = None  # Initialize Redis connection variable
    # Initialize potentially unbound variables
    message_id: Optional[str] = None
    message_data: Dict[str, Any] = {}
    # Use lowercase variables for mutable values from env
    redis_stream: Optional[str] = None
    redis_consumer_group: Optional[str] = None

    try:
        # --- 1. Read Input from Stdin ---
        logger.info("[Worker] Reading message data from stdin...")  # Changed from print
        input_data_str = sys.stdin.read()
        if not input_data_str:
            logger.critical(
                "[Worker] FATAL: No input data received from stdin."
            )  # Changed from print to critical
            sys.exit(1)

        try:
            input_data = json.loads(input_data_str)
            message_id = input_data["message_id"]
            message_data = input_data["message_data"]
            logger.info(
                f"[Worker] Received message_id: {message_id}"
            )  # Changed from print
        except (json.JSONDecodeError, KeyError) as e:
            logger.critical(
                f"[Worker] FATAL: Failed to parse input JSON from stdin: {e}"
            )  # Changed from print to critical
            # Cannot easily send error response without message_id/Redis info
            sys.exit(1)

        # --- 2. Read Configuration from Environment ---
        logger.info(
            "[Worker] Reading configuration from environment variables..."
        )  # Changed from print
        try:
            # Core function info
            _function_name = os.environ.get("FUNCTION_NAME")
            _entrypoint_rel_path = os.environ.get("NEBU_ENTRYPOINT_MODULE_PATH")

            # Type info
            is_stream_message = os.environ.get("IS_STREAM_MESSAGE") == "True"
            param_type_str = os.environ.get("PARAM_TYPE_STR")
            return_type_str = os.environ.get("RETURN_TYPE_STR")
            content_type_name = os.environ.get("CONTENT_TYPE_NAME")

            # Init func info
            _init_func_name = os.environ.get("INIT_FUNC_NAME")

            # Included object sources
            _included_object_sources = []
            i = 0
            while True:
                obj_source = os.environ.get(f"INCLUDED_OBJECT_{i}_SOURCE")
                if obj_source:
                    args = []
                    j = 0
                    while True:
                        arg_source = os.environ.get(
                            f"INCLUDED_OBJECT_{i}_ARG_{j}_SOURCE"
                        )
                        if arg_source:
                            args.append(arg_source)
                            j += 1
                        else:
                            break
                    _included_object_sources.append((obj_source, args))
                    i += 1
                else:
                    break

            if not _function_name or not _entrypoint_rel_path:
                raise ValueError(
                    "FUNCTION_NAME or NEBU_ENTRYPOINT_MODULE_PATH environment variables not set"
                )

            # Redis info
            REDIS_URL = os.environ.get("REDIS_URL", "")
            # Read into temporary uppercase vars first
            _redis_consumer_group_env = os.environ.get("REDIS_CONSUMER_GROUP")
            _redis_stream_env = os.environ.get("REDIS_STREAM")
            # Assign to lowercase mutable vars
            redis_consumer_group = _redis_consumer_group_env
            redis_stream = _redis_stream_env

            if not all([REDIS_URL, redis_consumer_group, redis_stream]):
                raise ValueError("Missing required Redis environment variables")

            # Calculate absolute path
            entrypoint_abs_path = os.path.abspath(_entrypoint_rel_path)
            if not os.path.exists(entrypoint_abs_path):
                python_path = os.environ.get("PYTHONPATH", "").split(os.pathsep)
                found_path = False
                for p_path in python_path:
                    potential_path = os.path.abspath(
                        os.path.join(p_path, _entrypoint_rel_path)
                    )
                    if os.path.exists(potential_path):
                        entrypoint_abs_path = potential_path
                        found_path = True
                        logger.info(  # Changed from print
                            f"[Worker] Found entrypoint absolute path via PYTHONPATH: {entrypoint_abs_path}"
                        )
                        break
                if not found_path:
                    raise ValueError(
                        f"Could not find entrypoint file via relative path '{_entrypoint_rel_path}' or in PYTHONPATH."
                    )

            # Convert entrypoint file path to module path
            _module_path = _entrypoint_rel_path.replace(os.sep, ".")
            if _module_path.endswith(".py"):
                _module_path = _module_path[:-3]
            if _module_path.endswith(".__init__"):
                _module_path = _module_path[: -len(".__init__")]
            elif _module_path == "__init__":
                raise ValueError(
                    f"Entrypoint '{_entrypoint_rel_path}' resolves to ambiguous top-level __init__."
                )
            if not _module_path:
                raise ValueError(
                    f"Could not derive a valid module path from entrypoint '{_entrypoint_rel_path}'"
                )

            logger.info(  # Changed from print
                f"[Worker] Config loaded. Module: '{_module_path}', Function: '{_function_name}'"
            )

        except ValueError as e:
            logger.critical(
                f"[Worker] FATAL: Configuration error: {e}"
            )  # Changed from print to critical
            # Cannot send error response without Redis connection
            sys.exit(1)
        except Exception as e:
            logger.critical(
                f"[Worker] FATAL: Unexpected error reading environment: {e}"
            )  # Changed from print to critical
            logger.exception(
                "Environment reading error:"
            )  # Replaced traceback.print_exc()
            sys.exit(1)

        # --- 3. Set up SOCKS Proxy ---
        logger.info("[Worker] Configuring SOCKS proxy...")  # Changed from print
        try:
            socks.set_default_proxy(socks.SOCKS5, "localhost", 1055)
            socket.socket = socks.socksocket
            logger.info(  # Changed from print
                "[Worker] Configured SOCKS5 proxy for socket connections via localhost:1055"
            )
        except Exception as e:
            logger.critical(
                f"[Worker] FATAL: Failed to configure SOCKS proxy: {e}"
            )  # Changed from print to critical
            logger.exception(
                "SOCKS proxy configuration error:"
            )  # Replaced traceback.print_exc()
            sys.exit(1)

        # --- 4. Connect to Redis ---
        logger.info("[Worker] Connecting to Redis...")  # Changed from print
        try:
            r = redis.from_url(REDIS_URL, decode_responses=True)
            r.ping()
            redis_info = REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL
            logger.info(
                f"[Worker] Connected to Redis via SOCKS proxy at {redis_info}"
            )  # Changed from print
        except Exception as e:
            logger.critical(
                f"[Worker] FATAL: Failed to connect to Redis: {e}"
            )  # Changed from print to critical
            logger.exception(
                "Redis connection error:"
            )  # Replaced traceback.print_exc()
            sys.exit(1)  # Cannot proceed without Redis

        # --- 5. Load User Code ---
        logger.info("[Worker] Loading user code...")  # Changed from print
        try:
            (
                target_function,
                init_function,
                imported_module,
                local_namespace,
            ) = load_user_code(
                _module_path,
                _function_name,
                entrypoint_abs_path,
                _init_func_name,
                _included_object_sources,
            )

            if target_function is None or imported_module is None:
                # load_user_code prints errors, just need to exit
                raise RuntimeError("User code loading failed.")
            logger.info("[Worker] User code loaded successfully.")  # Changed from print

        except Exception as e:
            logger.error(
                f"[Worker] Error during user code load: {e}"
            )  # Changed from print to error
            logger.exception("User code load error:")  # Replaced traceback.print_exc()
            # Send error response via Redis before exiting
            # Assert message_id is str before sending error
            assert isinstance(message_id, str)
            _send_error_response(
                message_id,
                f"User code load failed: {e}",
                traceback.format_exc(),
                message_data.get("return_stream"),
                message_data.get("user_id"),
            )
            # Acknowledge the message to prevent reprocessing a load failure
            try:
                assert isinstance(redis_stream, str)
                assert isinstance(redis_consumer_group, str)
                # message_id should be str here if code load failed after reading it
                assert isinstance(message_id, str)
                r.xack(redis_stream, redis_consumer_group, message_id)
                logger.warning(  # Changed from print to warning
                    f"[Worker] Acknowledged message {message_id} after code load failure."
                )
            except Exception as e_ack:
                logger.critical(  # Changed from print to critical
                    f"[Worker] CRITICAL: Failed to acknowledge message {message_id} after code load failure: {e_ack}"
                )
            sys.exit(1)  # Exit after attempting to report failure

        # --- 6. Execute Processing Logic (Adapted from consumer.py inline path) ---
        logger.info(
            f"[Worker] Processing message {message_id}..."
        )  # Changed from print
        return_stream = None
        user_id = None
        try:
            payload_str = message_data.get("data")
            if not payload_str:
                raise ValueError("Missing or invalid 'data' field")
            try:
                raw_payload = json.loads(payload_str)
            except json.JSONDecodeError as json_err:
                raise ValueError(
                    f"Failed to parse JSON payload: {json_err}"
                ) from json_err
            if not isinstance(raw_payload, dict):
                raise TypeError(
                    f"Expected parsed payload dictionary, got {type(raw_payload)}"
                )

            kind = raw_payload.get("kind", "")
            msg_id = raw_payload.get("id", "")  # ID from within the payload
            content_raw = raw_payload.get("content", {})
            created_at_str = raw_payload.get("created_at")
            try:
                created_at = (
                    datetime.fromisoformat(created_at_str)
                    if created_at_str and isinstance(created_at_str, str)
                    else datetime.now(timezone.utc)
                )
            except ValueError:
                created_at = datetime.now(timezone.utc)

            return_stream = raw_payload.get("return_stream")
            user_id = raw_payload.get("user_id")
            orgs = raw_payload.get("organizations")
            handle = raw_payload.get("handle")
            adapter = raw_payload.get("adapter")
            api_key = raw_payload.get("api_key")

            # --- Health Check Logic ---
            if kind == "HealthCheck":
                logger.info(
                    f"[Worker] Received HealthCheck message {message_id}"
                )  # Changed from print
                health_response = {
                    "kind": "StreamResponseMessage",
                    "id": message_id,  # Respond with original stream message ID
                    "content": {"status": "healthy", "checked_message_id": msg_id},
                    "status": "success",
                    "created_at": datetime.now().isoformat(),
                    "user_id": user_id,
                }
                if return_stream:
                    assert isinstance(return_stream, str)
                    r.xadd(return_stream, {"data": json.dumps(health_response)})
                    logger.info(
                        f"[Worker] Sent health check response to {return_stream}"
                    )  # Changed from print
                # Ack handled outside try/except block
                logger.info(
                    f"[Worker] HealthCheck for {message_id} processed successfully."
                )  # Changed from print
                result_content = None  # Indicate healthcheck success path
            else:
                # --- Normal Message Processing ---
                if isinstance(content_raw, str):
                    try:
                        content = json.loads(content_raw)
                    except json.JSONDecodeError:
                        content = content_raw
                else:
                    content = content_raw
                # logger.debug(f"[Worker] Content: {content}") # Changed from print to debug, commented out for less noise

                # --- Construct Input Object ---
                input_obj: Any = None
                input_type_class = None
                try:
                    from nebulous.processors.models import Message

                    if is_stream_message:
                        message_class = Message
                        content_model_class = None
                        if content_type_name:
                            try:
                                content_model_class = getattr(
                                    imported_module, content_type_name, None
                                )
                                if content_model_class is None:
                                    content_model_class = local_namespace.get(
                                        content_type_name
                                    )
                                    if content_model_class is None:
                                        logger.warning(  # Changed from print to warning
                                            f"[Worker] Warning: Content type class '{content_type_name}' not found."
                                        )
                                    else:
                                        logger.debug(  # Changed from print to debug
                                            f"[Worker] Found content model class: {content_model_class}"
                                        )
                            except Exception as e:
                                logger.warning(  # Changed from print to warning
                                    f"[Worker] Warning: Error resolving content type class '{content_type_name}': {e}"
                                )

                        if content_model_class:
                            try:
                                content_model = content_model_class.model_validate(
                                    content
                                )
                                # logger.debug( # Changed from print to debug, commented out
                                #     f"[Worker] Validated content model: {content_model}"
                                # )
                                input_obj = message_class(
                                    kind=kind,
                                    id=msg_id,
                                    content=content_model,
                                    created_at=int(created_at.timestamp()),
                                    return_stream=return_stream,
                                    user_id=user_id,
                                    orgs=orgs,
                                    handle=handle,
                                    adapter=adapter,
                                    api_key=api_key,
                                )
                            except Exception as e:
                                logger.error(  # Changed from print to error
                                    f"[Worker] Error validating/creating content model '{content_type_name}': {e}. Falling back."
                                )
                                input_obj = message_class(
                                    kind=kind,
                                    id=msg_id,
                                    content=cast(Any, content),
                                    created_at=int(created_at.timestamp()),
                                    return_stream=return_stream,
                                    user_id=user_id,
                                    orgs=orgs,
                                    handle=handle,
                                    adapter=adapter,
                                    api_key=api_key,
                                )
                        else:
                            input_obj = message_class(
                                kind=kind,
                                id=msg_id,
                                content=cast(Any, content),
                                created_at=int(created_at.timestamp()),
                                return_stream=return_stream,
                                user_id=user_id,
                                orgs=orgs,
                                handle=handle,
                                adapter=adapter,
                                api_key=api_key,
                            )
                    else:  # Not a stream message
                        param_type_name = param_type_str
                        try:
                            input_type_class = (
                                getattr(imported_module, param_type_name, None)
                                if param_type_name
                                else None
                            )
                            if input_type_class is None and param_type_name:
                                input_type_class = local_namespace.get(param_type_name)
                            if input_type_class is None:
                                if param_type_name:
                                    logger.warning(  # Changed from print to warning
                                        f"[Worker] Warning: Input type class '{param_type_name}' not found. Passing raw."
                                    )
                                input_obj = content
                            else:
                                logger.debug(  # Changed from print to debug
                                    f"[Worker] Found input model class: {input_type_class}"
                                )
                                input_obj = input_type_class.model_validate(content)
                                logger.debug(
                                    f"[Worker] Validated input model: {input_obj}"
                                )  # Changed from print to debug
                        except Exception as e:
                            logger.error(  # Changed from print to error
                                f"[Worker] Error resolving/validating input type '{param_type_name}': {e}. Passing raw."
                            )
                            input_obj = content

                except NameError as e:
                    raise RuntimeError(
                        f"Required class not found (e.g., Message or param type): {e}"
                    ) from e
                except Exception as e:
                    logger.error(
                        f"[Worker] Error constructing input object: {e}"
                    )  # Changed from print to error
                    raise

                # --- Execute the Function ---
                logger.info(
                    f"[Worker] Executing function '{_function_name}'..."
                )  # Changed from print
                result = target_function(input_obj)
                logger.debug(
                    f"[Worker] Result: {result}"
                )  # Changed from print to debug

                # --- Convert Result ---
                if hasattr(result, "model_dump"):
                    result_content = result.model_dump(mode="json")
                elif hasattr(result, "dict"):
                    result_content = result.dict()
                else:
                    result_content = result

            # --- 7. Send Result / Handle Success (outside HealthCheck specific block) ---
            if kind != "HealthCheck":  # Only send response for non-healthcheck messages
                response = {
                    "kind": "StreamResponseMessage",
                    "id": message_id,  # Use original stream message ID
                    "content": result_content,
                    "status": "success",
                    "created_at": datetime.now().isoformat(),
                    "user_id": user_id,
                }
                if return_stream:
                    assert isinstance(return_stream, str)
                    r.xadd(return_stream, {"data": json.dumps(response)})
                    logger.info(  # Changed from print
                        f"[Worker] Processed message {message_id}, result sent to {return_stream}"
                    )

            # --- 8. Acknowledge Original Message (on success) ---
            assert isinstance(redis_stream, str)
            assert isinstance(redis_consumer_group, str)
            assert isinstance(
                message_id, str
            )  # message_id is str if processing succeeded
            r.xack(redis_stream, redis_consumer_group, message_id)
            logger.info(
                f"[Worker] Acknowledged message {message_id} successfully."
            )  # Changed from print

            # --- 9. Exit Successfully ---
            logger.info("[Worker] Exiting with status 0.")  # Changed from print
            sys.exit(0)

        except RetriableError as e:
            # --- Handle Retriable Processing Error ---
            logger.warning(
                f"[Worker] Retriable error processing message {message_id}: {e}"
            )  # Changed from print to warning
            tb = traceback.format_exc()
            # logger.exception("Retriable error details:") # Use logger.exception instead of printing tb
            # Assert message_id is str before sending error
            assert isinstance(message_id, str)
            # Send error response (optional, consider suppressing later if too noisy)
            _send_error_response(message_id, str(e), tb, return_stream, user_id)

            # DO NOT Acknowledge the message for retriable errors

            # --- 9. Exit with specific code for retriable failure ---
            logger.warning(
                "[Worker] Exiting with status 3 due to retriable error."
            )  # Changed from print to warning
            sys.exit(3)

        except Exception as e:
            # --- Handle Non-Retriable Processing Error ---
            logger.error(
                f"[Worker] Error processing message {message_id}: {e}"
            )  # Changed from print to error
            tb = traceback.format_exc()
            # logger.exception("Processing error details:") # Use logger.exception instead of printing tb
            # Assert message_id is str before sending error
            assert isinstance(message_id, str)
            _send_error_response(message_id, str(e), tb, return_stream, user_id)

            # Acknowledge the message even if processing failed
            try:
                assert isinstance(redis_stream, str)
                assert isinstance(redis_consumer_group, str)
                # message_id is str if processing failed after reading it
                assert isinstance(message_id, str)
                r.xack(redis_stream, redis_consumer_group, message_id)
                logger.info(
                    f"[Worker] Acknowledged failed message {message_id}"
                )  # Changed from print
            except Exception as e_ack:
                logger.critical(  # Changed from print to critical
                    f"[Worker] CRITICAL: Failed to acknowledge failed message {message_id}: {e_ack}"
                )

            # --- 9. Exit with Failure ---
            logger.error(
                "[Worker] Exiting with status 1 due to processing error."
            )  # Changed from print to error
            sys.exit(1)

    except Exception as outer_e:
        # --- Handle Catastrophic Worker Error (e.g., setup failure) ---
        logger.critical(
            f"[Worker] FATAL outer error: {outer_e}"
        )  # Changed from print to critical
        tb = traceback.format_exc()
        # logger.exception("Fatal worker error:") # Use logger.exception instead of printing tb
        # If Redis was connected, try to send a generic error for the message_id read from stdin
        # Check that all required variables are not None before proceeding
        if (
            r is not None
            and message_id is not None
            and redis_stream is not None
            and redis_consumer_group is not None
        ):
            try:
                # Assert types explicitly before calls
                assert isinstance(message_id, str)
                assert isinstance(redis_stream, str)
                assert isinstance(redis_consumer_group, str)

                _send_error_response(
                    message_id,
                    f"Worker failed during setup or processing: {outer_e}",
                    tb,
                    message_data.get("return_stream"),
                    message_data.get("user_id"),
                )
                # Attempt to ack if possible, even though the main consumer *might* also try
                r.xack(redis_stream, redis_consumer_group, message_id)
                logger.warning(  # Changed from print to warning
                    f"[Worker] Attempted to acknowledge message {message_id} after fatal error."
                )
            except Exception as final_e:
                logger.critical(  # Changed from print to critical
                    f"[Worker] CRITICAL: Failed during final error reporting/ack: {final_e}"
                )
        else:
            logger.critical(  # Changed from print to critical
                "[Worker] CRITICAL: Could not report final error or ack message due to missing Redis connection or message details."
            )

        logger.critical(
            "[Worker] Exiting with status 1 due to fatal error."
        )  # Changed from print to critical
        sys.exit(1)
