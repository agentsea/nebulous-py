#!/usr/bin/env python3
import importlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import types
from datetime import datetime, timezone
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

import redis
import socks
from redis import ConnectionError, ResponseError

from nebu.errors import RetriableError
from nebu.logging import logger

# Define TypeVar for generic models
T = TypeVar("T")

# Environment variable name used as a guard in the decorator
_NEBU_INSIDE_CONSUMER_ENV_VAR = "_NEBU_INSIDE_CONSUMER_EXEC"

# --- Global variables for dynamically loaded code ---
target_function: Optional[Callable] = None
init_function: Optional[Callable] = None
imported_module: Optional[types.ModuleType] = None
local_namespace: Dict[str, Any] = {}  # Namespace for included objects
last_load_mtime: float = 0.0
entrypoint_abs_path: Optional[str] = None

REDIS_CONSUMER_GROUP = os.environ.get("REDIS_CONSUMER_GROUP")
REDIS_STREAM = os.environ.get("REDIS_STREAM")
NEBU_EXECUTION_MODE = os.environ.get("NEBU_EXECUTION_MODE", "inline").lower()
execution_mode = NEBU_EXECUTION_MODE

if execution_mode not in ["inline", "subprocess"]:
    logger.warning(
        f"Invalid NEBU_EXECUTION_MODE: {NEBU_EXECUTION_MODE}. Must be 'inline' or 'subprocess'. Defaulting to 'inline'."
    )
    execution_mode = "inline"

logger.info(f"Execution mode: {execution_mode}")


# --- Function to Load/Reload User Code ---
def load_or_reload_user_code(
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
    float,
]:
    """Loads or reloads the user code module, executes includes, and returns functions/module."""
    global _NEBU_INSIDE_CONSUMER_ENV_VAR  # Access the global guard var name

    current_mtime = 0.0
    loaded_target_func = None
    loaded_init_func = None
    loaded_module = None
    exec_namespace: Dict[str, Any] = {}  # Use a local namespace for this load attempt

    logger.info(f"[Code Loader] Attempting to load/reload module: '{module_path}'")
    os.environ[_NEBU_INSIDE_CONSUMER_ENV_VAR] = "1"  # Set guard *before* import/reload
    logger.debug(
        f"[Code Loader] Set environment variable {_NEBU_INSIDE_CONSUMER_ENV_VAR}=1"
    )

    try:
        current_mtime = os.path.getmtime(entrypoint_abs_path)

        # Execute included object sources FIRST (if any)
        if included_object_sources:
            logger.debug("[Code Loader] Executing @include object sources...")
            # Include necessary imports for the exec context
            exec("from pydantic import BaseModel, Field", exec_namespace)
            exec(
                "from typing import Optional, List, Dict, Any, Generic, TypeVar",
                exec_namespace,
            )
            exec("T_exec = TypeVar('T_exec')", exec_namespace)
            exec("from nebu.processors.models import *", exec_namespace)
            # ... add other common imports if needed by included objects ...

            for i, (obj_source, args_sources) in enumerate(included_object_sources):
                try:
                    exec(obj_source, exec_namespace)
                    logger.debug(
                        f"[Code Loader] Successfully executed included object {i} base source"
                    )
                    for j, arg_source in enumerate(args_sources):
                        try:
                            exec(arg_source, exec_namespace)
                            logger.debug(
                                f"[Code Loader] Successfully executed included object {i} arg {j} source"
                            )
                        except Exception as e_arg:
                            logger.error(
                                f"Error executing included object {i} arg {j} source: {e_arg}"
                            )
                            logger.exception(
                                f"Traceback for included object {i} arg {j} source error:"
                            )
                except Exception as e_base:
                    logger.error(
                        f"Error executing included object {i} base source: {e_base}"
                    )
                    logger.exception(
                        f"Traceback for included object {i} base source error:"
                    )
            logger.debug("[Code Loader] Finished executing included object sources.")

        # Check if module is already loaded and needs reload
        if module_path in sys.modules:
            logger.info(
                f"[Code Loader] Module '{module_path}' already imported. Reloading..."
            )
            # Pass the exec_namespace as globals? Usually reload works within its own context.
            # If included objects *modify* the module's global scope upon exec,
            # reload might not pick that up easily. Might need a fresh import instead.
            # Let's try reload first.
            loaded_module = importlib.reload(sys.modules[module_path])
            logger.info(f"[Code Loader] Successfully reloaded module: {module_path}")
        else:
            # Import the main module
            loaded_module = importlib.import_module(module_path)
            logger.info(
                f"[Code Loader] Successfully imported module for the first time: {module_path}"
            )

        # Get the target function from the loaded/reloaded module
        loaded_target_func = getattr(loaded_module, function_name)
        logger.info(
            f"[Code Loader] Successfully loaded function '{function_name}' from module '{module_path}'"
        )

        # Get the init function if specified
        if init_func_name:
            loaded_init_func = getattr(loaded_module, init_func_name)
            logger.info(
                f"[Code Loader] Successfully loaded init function '{init_func_name}' from module '{module_path}'"
            )
            # Execute init_func
            logger.info(f"[Code Loader] Executing init_func: {init_func_name}...")
            loaded_init_func()  # Call the function
            logger.info(
                f"[Code Loader] Successfully executed init_func: {init_func_name}"
            )

        logger.info("[Code Loader] Code load/reload successful.")
        return (
            loaded_target_func,
            loaded_init_func,
            loaded_module,
            exec_namespace,
            current_mtime,
        )

    except FileNotFoundError:
        logger.error(
            f"[Code Loader] Error: Entrypoint file not found at '{entrypoint_abs_path}'. Cannot load/reload."
        )
        return None, None, None, {}, 0.0  # Indicate failure
    except ImportError as e:
        logger.error(
            f"[Code Loader] Error importing/reloading module '{module_path}': {e}"
        )
        logger.exception("Import/Reload Error Traceback:")
        return None, None, None, {}, 0.0  # Indicate failure
    except AttributeError as e:
        logger.error(
            f"[Code Loader] Error accessing function '{function_name}' or '{init_func_name}' in module '{module_path}': {e}"
        )
        logger.exception("Attribute Error Traceback:")
        return None, None, None, {}, 0.0  # Indicate failure
    except Exception as e:
        logger.error(f"[Code Loader] Unexpected error during code load/reload: {e}")
        logger.exception("Unexpected Code Load/Reload Error Traceback:")
        return None, None, None, {}, 0.0  # Indicate failure
    finally:
        # Unset the guard environment variable
        os.environ.pop(_NEBU_INSIDE_CONSUMER_ENV_VAR, None)
        logger.debug(
            f"[Code Loader] Unset environment variable {_NEBU_INSIDE_CONSUMER_ENV_VAR}"
        )


# Print all environment variables before starting
logger.debug("===== ENVIRONMENT VARIABLES =====")
for key, value in sorted(os.environ.items()):
    logger.debug(f"{key}={value}")
logger.debug("=================================")

# --- Get Environment Variables ---
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
                arg_source = os.environ.get(f"INCLUDED_OBJECT_{i}_ARG_{j}_SOURCE")
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
        logger.critical(
            "FATAL: FUNCTION_NAME or NEBU_ENTRYPOINT_MODULE_PATH environment variables not set"
        )
        sys.exit(1)

    # Calculate absolute path for modification time checking
    # Assuming CWD or PYTHONPATH allows finding the relative path
    # This might need adjustment based on deployment specifics
    entrypoint_abs_path = os.path.abspath(_entrypoint_rel_path)
    if not os.path.exists(entrypoint_abs_path):
        # Try constructing path based on PYTHONPATH if direct abspath fails
        python_path = os.environ.get("PYTHONPATH", "").split(os.pathsep)
        found_path = False
        for p_path in python_path:
            potential_path = os.path.abspath(os.path.join(p_path, _entrypoint_rel_path))
            if os.path.exists(potential_path):
                entrypoint_abs_path = potential_path
                found_path = True
                logger.info(
                    f"[Consumer] Found entrypoint absolute path via PYTHONPATH: {entrypoint_abs_path}"
                )
                break
        if not found_path:
            logger.critical(
                f"FATAL: Could not find entrypoint file via relative path '{_entrypoint_rel_path}' or in PYTHONPATH."
            )
            # Attempting abspath anyway for the error message in load function
            entrypoint_abs_path = os.path.abspath(_entrypoint_rel_path)

    # Convert entrypoint file path to module path
    _module_path = _entrypoint_rel_path.replace(os.sep, ".")
    if _module_path.endswith(".py"):
        _module_path = _module_path[:-3]
    if _module_path.endswith(".__init__"):
        _module_path = _module_path[: -len(".__init__")]
    elif _module_path == "__init__":
        logger.critical(
            f"FATAL: Entrypoint '{_entrypoint_rel_path}' resolves to ambiguous top-level __init__. Please use a named file or package."
        )
        sys.exit(1)
    if not _module_path:
        logger.critical(
            f"FATAL: Could not derive a valid module path from entrypoint '{_entrypoint_rel_path}'"
        )
        sys.exit(1)

    logger.info(
        f"[Consumer] Initializing. Entrypoint: '{_entrypoint_rel_path}', Module: '{_module_path}', Function: '{_function_name}', Init: '{_init_func_name}'"
    )

    # --- Initial Load of User Code ---
    (
        target_function,
        init_function,
        imported_module,
        local_namespace,
        last_load_mtime,
    ) = load_or_reload_user_code(
        _module_path,
        _function_name,
        entrypoint_abs_path,
        _init_func_name,
        _included_object_sources,
    )

    if target_function is None or imported_module is None:
        logger.critical("FATAL: Initial load of user code failed. Exiting.")
        sys.exit(1)
    logger.info(
        f"[Consumer] Initial code load successful. Last modified time: {last_load_mtime}"
    )


except Exception as e:
    logger.critical(f"FATAL: Error during initial environment setup or code load: {e}")
    logger.exception("Initial Setup/Load Error Traceback:")
    sys.exit(1)

# Get Redis connection parameters from environment
REDIS_URL = os.environ.get("REDIS_URL", "")

if not all([REDIS_URL, REDIS_CONSUMER_GROUP, REDIS_STREAM]):
    logger.critical("Missing required Redis environment variables")
    sys.exit(1)

# Configure SOCKS proxy before connecting to Redis
# Use the proxy settings provided by tailscaled
socks.set_default_proxy(socks.SOCKS5, "localhost", 1055)
socket.socket = socks.socksocket
logger.info("Configured SOCKS5 proxy for socket connections via localhost:1055")

# Connect to Redis
try:
    # Parse the Redis URL to handle potential credentials or specific DBs if needed
    # Although from_url should work now with the patched socket
    r = redis.from_url(
        REDIS_URL, decode_responses=True
    )  # Added decode_responses for convenience
    r.ping()  # Test connection
    redis_info = REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL
    logger.info(f"Connected to Redis via SOCKS proxy at {redis_info}")
except Exception as e:
    logger.critical(f"Failed to connect to Redis via SOCKS proxy: {e}")
    logger.exception("Redis Connection Error Traceback:")
    sys.exit(1)

# Create consumer group if it doesn't exist
try:
    # Assert types before use
    assert isinstance(REDIS_STREAM, str)
    assert isinstance(REDIS_CONSUMER_GROUP, str)
    r.xgroup_create(REDIS_STREAM, REDIS_CONSUMER_GROUP, id="0", mkstream=True)
    logger.info(
        f"Created consumer group {REDIS_CONSUMER_GROUP} for stream {REDIS_STREAM}"
    )
except ResponseError as e:
    if "BUSYGROUP" in str(e):
        logger.info(f"Consumer group {REDIS_CONSUMER_GROUP} already exists")
    else:
        logger.error(f"Error creating consumer group: {e}")
        logger.exception("Consumer Group Creation Error Traceback:")


# Function to process messages
def process_message(message_id: str, message_data: Dict[str, str]) -> None:
    # Access the globally managed user code elements
    global target_function, imported_module, local_namespace
    global execution_mode, r, REDIS_STREAM, REDIS_CONSUMER_GROUP

    # --- Subprocess Execution Path ---
    if execution_mode == "subprocess":
        logger.info(f"Processing message {message_id} in subprocess...")
        process = None  # Initialize process variable

        # Helper function to read and print stream lines
        def stream_reader(stream: IO[str], prefix: str):
            try:
                for line in iter(stream.readline, ""):
                    logger.debug(f"{prefix}: {line.strip()}")
            except Exception as e:
                logger.error(f"Error reading stream {prefix}: {e}")
            finally:
                stream.close()

        try:
            worker_cmd = [
                sys.executable,
                "-u",  # Force unbuffered stdout/stderr in the subprocess
                "-m",
                "nebu.processors.consumer_process_worker",
            ]
            process_input = json.dumps(
                {"message_id": message_id, "message_data": message_data}
            )

            # Start the worker process
            process = subprocess.Popen(
                worker_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,  # Line buffered
                env=os.environ.copy(),
            )

            # Create threads to read stdout and stderr concurrently
            stdout_thread = threading.Thread(
                target=stream_reader,
                args=(process.stdout, f"[Subprocess STDOUT {message_id[:8]}]"),
            )
            stderr_thread = threading.Thread(
                target=stream_reader,
                args=(process.stderr, f"[Subprocess STDERR {message_id[:8]}]"),
            )

            stdout_thread.start()
            stderr_thread.start()

            # Send input data to the subprocess
            # Ensure process and stdin are valid before writing/closing
            if process and process.stdin:
                try:
                    process.stdin.write(process_input)
                    process.stdin.close()  # Signal end of input
                except (BrokenPipeError, OSError) as e:
                    # Handle cases where the process might have exited early
                    logger.warning(
                        f"Warning: Failed to write full input to subprocess {message_id}: {e}. It might have exited prematurely."
                    )
                    # Continue to wait and check return code
            else:
                logger.error(
                    f"Error: Subprocess stdin stream not available for {message_id}. Cannot send input."
                )
                # Handle this case - perhaps terminate and report error?
                # For now, we'll let it proceed to wait() which will likely show an error code.

            # Wait for the process to finish
            return_code = (
                process.wait() if process else -1
            )  # Handle case where process is None

            # Wait for reader threads to finish consuming remaining output
            stdout_thread.join()
            stderr_thread.join()

            if return_code == 0:
                logger.info(
                    f"Subprocess for {message_id} completed successfully (return code 0)."
                )
                # Assume success handling (ack/response) was done by the worker
            elif return_code == 3:
                logger.warning(
                    f"Subprocess for {message_id} reported a retriable error (exit code 3). Message will not be acknowledged."
                )
                # Optionally send an error response here, though the worker already did.
                # _send_error_response(...)
                # DO NOT Acknowledge the message here, let it be retried.
            else:
                logger.error(
                    f"Subprocess for {message_id} failed with exit code {return_code}."
                )
                # Worker likely failed, send generic error and ACK here
                _send_error_response(
                    message_id,
                    f"Subprocess execution failed with exit code {return_code}",
                    "See consumer logs for subprocess stderr.",  # stderr was already printed
                    message_data.get("return_stream"),
                    message_data.get("user_id"),
                )
                # CRITICAL: Acknowledge the message here since the subprocess failed
                try:
                    assert isinstance(REDIS_STREAM, str)
                    assert isinstance(REDIS_CONSUMER_GROUP, str)
                    r.xack(REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id)
                    logger.info(f"Acknowledged failed subprocess message {message_id}")
                except Exception as e_ack:
                    logger.critical(
                        f"CRITICAL: Failed to acknowledge failed subprocess message {message_id}: {e_ack}"
                    )

        except FileNotFoundError:
            logger.critical(
                "FATAL: Worker script 'nebu.processors.consumer_process_worker' not found. Check PYTHONPATH."
            )
            # Send error and ack if possible
            _send_error_response(
                message_id,
                "Worker script not found",
                traceback.format_exc(),
                message_data.get("return_stream"),
                message_data.get("user_id"),
            )
            try:
                assert isinstance(REDIS_STREAM, str)
                assert isinstance(REDIS_CONSUMER_GROUP, str)
                r.xack(REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id)
                logger.info(
                    f"Acknowledged message {message_id} after worker script not found failure"
                )
            except Exception as e_ack:
                logger.critical(
                    f"CRITICAL: Failed to acknowledge message {message_id} after worker script not found failure: {e_ack}"
                )

        except Exception as e:
            logger.error(
                f"Error launching or managing subprocess for message {message_id}: {e}"
            )
            logger.exception("Subprocess Launch/Manage Error Traceback:")
            # Also send an error and acknowledge
            _send_error_response(
                message_id,
                f"Failed to launch/manage subprocess: {e}",
                traceback.format_exc(),
                message_data.get("return_stream"),
                message_data.get("user_id"),
            )
            try:
                assert isinstance(REDIS_STREAM, str)
                assert isinstance(REDIS_CONSUMER_GROUP, str)
                r.xack(REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id)
                logger.info(
                    f"Acknowledged message {message_id} after subprocess launch/manage failure"
                )
            except Exception as e_ack:
                logger.critical(
                    f"CRITICAL: Failed to acknowledge message {message_id} after subprocess launch/manage failure: {e_ack}"
                )
            # Ensure process is terminated if it's still running after an error
            if process and process.poll() is None:
                logger.warning(
                    f"Terminating potentially lingering subprocess for {message_id}..."
                )
                process.terminate()
                process.wait(timeout=5)  # Give it a moment to terminate
                if process.poll() is None:
                    logger.warning(
                        f"Subprocess for {message_id} did not terminate gracefully, killing."
                    )
                    process.kill()
        finally:
            # Ensure streams are closed even if threads failed or process is None
            if process:
                if process.stdout:
                    try:
                        process.stdout.close()
                    except Exception:
                        pass  # Ignore errors during cleanup close
                if process.stderr:
                    try:
                        process.stderr.close()
                    except Exception:
                        pass  # Ignore errors during cleanup close
                # Stdin should already be closed, but doesn't hurt to be safe
                if process.stdin and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except Exception:
                        pass

        return  # Exit process_message after handling subprocess logic

    # --- Inline Execution Path (Original Logic) ---
    if target_function is None or imported_module is None:
        logger.error(
            f"Error processing message {message_id}: User code (target_function or module) is not loaded. Skipping."
        )
        _send_error_response(
            message_id,
            "User code is not loaded (likely due to a failed reload)",
            traceback.format_exc(),
            None,
            None,
        )  # Pass None for user_id if unavailable here
        # Acknowledge message with code load failure to prevent reprocessing loop
        try:
            assert isinstance(REDIS_STREAM, str)
            assert isinstance(REDIS_CONSUMER_GROUP, str)
            r.xack(REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id)
            logger.warning(
                f"Acknowledged message {message_id} due to code load failure."
            )
        except Exception as e_ack:
            logger.critical(
                f"CRITICAL: Failed to acknowledge message {message_id} after code load failure: {e_ack}"
            )
        return  # Skip processing

    return_stream = None
    user_id = None
    try:
        payload_str = message_data.get("data")
        if (
            not payload_str
        ):  # Covers None and empty string, isinstance check is redundant
            raise ValueError(
                f"Missing or invalid 'data' field (expected non-empty string): {message_data}"
            )
        try:
            raw_payload = json.loads(payload_str)
        except json.JSONDecodeError as json_err:
            raise ValueError(f"Failed to parse JSON payload: {json_err}") from json_err
        if not isinstance(raw_payload, dict):
            raise TypeError(
                f"Expected parsed payload to be a dictionary, but got {type(raw_payload)}"
            )

        logger.debug(f">> Raw payload: {raw_payload}")

        kind = raw_payload.get("kind", "")
        msg_id = raw_payload.get("id", "")
        content_raw = raw_payload.get("content", {})
        created_at_str = raw_payload.get("created_at")  # Get as string or None
        # Attempt to parse created_at, fallback to now()
        try:
            created_at = (
                datetime.fromisoformat(created_at_str)
                if created_at_str
                and isinstance(created_at_str, str)  # Check type explicitly
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
        logger.debug(f">> Extracted API key length: {len(api_key) if api_key else 0}")

        # --- Health Check Logic (Keep as is) ---
        if kind == "HealthCheck":
            logger.info(f"Received HealthCheck message {message_id}")
            health_response = {
                "kind": "StreamResponseMessage",  # Respond with a standard message kind
                "id": message_id,
                "content": {"status": "healthy", "checked_message_id": msg_id},
                "status": "success",
                "created_at": datetime.now().isoformat(),
                "user_id": user_id,  # Include user_id if available
            }
            if return_stream:
                # Assert type again closer to usage for type checker clarity
                assert isinstance(return_stream, str)
                r.xadd(return_stream, {"data": json.dumps(health_response)})
                logger.info(f"Sent health check response to {return_stream}")

            # Assert types again closer to usage for type checker clarity
            assert isinstance(REDIS_STREAM, str)
            assert isinstance(REDIS_CONSUMER_GROUP, str)
            r.xack(REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id)
            logger.info(f"Acknowledged HealthCheck message {message_id}")
            return  # Exit early for health checks
        # --- End Health Check Logic ---

        # Parse content if it's a string (e.g., double-encoded JSON)
        if isinstance(content_raw, str):
            try:
                content = json.loads(content_raw)
            except json.JSONDecodeError:
                content = content_raw  # Keep as string if not valid JSON
        else:
            content = content_raw

        # print(f"Content: {content}")

        # --- Construct Input Object using Imported Types ---
        input_obj: Any = None
        input_type_class = None

        try:
            # Try to get the actual model classes (they should be available via import)
            # Need to handle potential NameErrors if imports failed silently
            # Note: This assumes models are defined in the imported module scope
            # Or imported by the imported module.
            from nebu.processors.models import Message  # Import needed message class

            if is_stream_message:
                message_class = Message  # Use imported class
                content_model_class = None
                if content_type_name:
                    try:
                        # Assume content_type_name refers to a class available in the global scope
                        # (either from imported module or included objects)
                        # Use the globally managed imported_module and local_namespace
                        content_model_class = getattr(
                            imported_module, content_type_name, None
                        )
                        if content_model_class is None:
                            # Check in local_namespace from included objects as fallback
                            content_model_class = local_namespace.get(content_type_name)
                        if content_model_class is None:
                            logger.warning(
                                f"Warning: Content type class '{content_type_name}' not found in imported module or includes."
                            )
                        else:
                            logger.debug(
                                f"Found content model class: {content_model_class}"
                            )
                    except AttributeError:
                        logger.warning(
                            f"Warning: Content type class '{content_type_name}' not found in imported module."
                        )
                    except Exception as e:
                        logger.warning(
                            f"Warning: Error resolving content type class '{content_type_name}': {e}"
                        )

                if content_model_class:
                    try:
                        content_model = content_model_class.model_validate(content)
                        # print(f"Validated content model: {content_model}")
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
                        logger.error(
                            f"Error validating/creating content model '{content_type_name}': {e}. Falling back."
                        )
                        # Fallback to raw content in Message
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
                    # No content type name or class found, use raw content
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
            else:  # Not a stream message, use the function's parameter type
                param_type_name = (
                    param_type_str  # Assume param_type_str holds the class name
                )
                # Attempt to resolve the parameter type class
                try:
                    # Use the globally managed imported_module and local_namespace
                    input_type_class = (
                        getattr(imported_module, param_type_name, None)
                        if param_type_name
                        else None
                    )
                    if input_type_class is None and param_type_name:
                        input_type_class = local_namespace.get(param_type_name)
                    if input_type_class is None:
                        if param_type_name:  # Only warn if a name was expected
                            logger.warning(
                                f"Warning: Input type class '{param_type_name}' not found. Passing raw content."
                            )
                        input_obj = content
                    else:
                        logger.debug(f"Found input model class: {input_type_class}")
                        input_obj = input_type_class.model_validate(content)
                        logger.debug(f"Validated input model: {input_obj}")
                except AttributeError:
                    logger.warning(
                        f"Warning: Input type class '{param_type_name}' not found in imported module."
                    )
                    input_obj = content
                except Exception as e:
                    logger.error(
                        f"Error resolving/validating input type '{param_type_name}': {e}. Passing raw content."
                    )
                    input_obj = content

        except NameError as e:
            logger.error(
                f"Error: Required class (e.g., Message or parameter type) not found. Import failed? {e}"
            )
            # Can't proceed without types, re-raise or handle error response
            raise RuntimeError(f"Required class not found: {e}") from e
        except Exception as e:
            logger.error(f"Error constructing input object: {e}")
            raise  # Re-raise unexpected errors during input construction

        # print(f"Input object: {input_obj}") # Reduce verbosity
        # logger.debug(f"Input object: {input_obj}") # Could use logger.debug if needed

        # Execute the function
        logger.info("Executing function...")
        result = target_function(input_obj)
        # logger.debug(f"Raw Result: {result}") # Debugging

        result_content = None  # Default to None
        if result is not None:  # Only process if there's a result
            try:
                if hasattr(result, "model_dump"):
                    logger.debug("[Consumer] Result has model_dump, using it.")
                    # Use 'json' mode to ensure serializability where possible
                    result_content = result.model_dump(mode="json")
                    # logger.debug(f"[Consumer] Result after model_dump: {result_content}") # Debugging
                else:
                    # Try standard json.dumps as a fallback to check serializability
                    logger.debug(
                        "[Consumer] Result has no model_dump, attempting json.dumps check."
                    )
                    try:
                        # Test if it's serializable
                        json.dumps(result)
                        # If the above line doesn't raise TypeError, assign the original result
                        result_content = result
                        # logger.debug(f"[Consumer] Result assigned directly after json.dumps check passed: {result_content}") # Debugging
                    except TypeError as e:
                        logger.warning(
                            f"[Consumer] Warning: Result is not JSON serializable: {e}. Discarding result."
                        )
                        result_content = None  # Explicitly set to None on failure

            except (
                Exception
            ) as e:  # Catch other potential model_dump errors or unexpected issues
                logger.warning(
                    f"[Consumer] Warning: Unexpected error during result processing/serialization: {e}. Discarding result."
                )
                logger.exception("Result Processing/Serialization Error Traceback:")
                result_content = None

        # Prepare the response (ensure 'content' key exists even if None)
        response = {
            "kind": "StreamResponseMessage",
            "id": message_id,
            "content": result_content,  # Use the potentially None result_content
            "status": "success",
            "created_at": datetime.now(timezone.utc).isoformat(),  # Use UTC
            "user_id": user_id,  # Pass user_id back
        }

        # print(f"Final Response Content: {response['content']}") # Debugging

        # Send the result to the return stream
        if return_stream:
            assert isinstance(return_stream, str)
            r.xadd(return_stream, {"data": json.dumps(response)})
            logger.info(
                f"Processed message {message_id}, result sent to {return_stream}"
            )

        # Acknowledge the message
        assert isinstance(REDIS_STREAM, str)
        assert isinstance(REDIS_CONSUMER_GROUP, str)
        r.xack(REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id)

    except RetriableError as e:
        logger.warning(f"Retriable error processing message {message_id}: {e}")
        logger.exception("Retriable Error Traceback:")
        _send_error_response(
            message_id, str(e), traceback.format_exc(), return_stream, user_id
        )
        # DO NOT Acknowledge the message for retriable errors
        logger.info(f"Message {message_id} will be retried later.")

    except Exception as e:
        logger.error(f"Error processing message {message_id}: {e}")
        logger.exception("Message Processing Error Traceback:")
        _send_error_response(
            message_id, str(e), traceback.format_exc(), return_stream, user_id
        )

        # Acknowledge the message even if processing failed
        try:
            assert isinstance(REDIS_STREAM, str)
            assert isinstance(REDIS_CONSUMER_GROUP, str)
            r.xack(REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id)
            logger.info(f"Acknowledged failed message {message_id}")
        except Exception as e_ack:
            logger.critical(
                f"CRITICAL: Failed to acknowledge failed message {message_id}: {e_ack}"
            )


# --- Helper to Send Error Response ---
def _send_error_response(
    message_id: str,
    error_msg: str,
    tb: str,
    return_stream: Optional[str],
    user_id: Optional[str],
):
    """Sends a standardized error response to Redis."""
    global r, REDIS_STREAM  # Access global Redis connection and stream name

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

    error_destination = f"{REDIS_STREAM}.errors"  # Default error stream
    if return_stream:  # Prefer return_stream if available
        error_destination = return_stream

    try:
        assert isinstance(error_destination, str)
        r.xadd(error_destination, {"data": json.dumps(error_response)})
        logger.info(
            f"Sent error response for message {message_id} to {error_destination}"
        )
    except Exception as e_redis:
        logger.critical(
            f"CRITICAL: Failed to send error response for {message_id} to Redis: {e_redis}"
        )
        logger.exception("Redis Error Response Send Error Traceback:")


# Main loop
logger.info(
    f"Starting consumer for stream {REDIS_STREAM} in group {REDIS_CONSUMER_GROUP}"
)
consumer_name = f"consumer-{os.getpid()}-{socket.gethostname()}"  # More unique name
MIN_IDLE_TIME_MS = 60000  # Minimum idle time in milliseconds (e.g., 60 seconds)
CLAIM_COUNT = 10  # Max messages to claim at once

# Check if hot reloading should be disabled
disable_hot_reload = os.environ.get("NEBU_DISABLE_HOT_RELOAD", "0").lower() in [
    "1",
    "true",
]
logger.info(
    f"[Consumer] Hot code reloading is {'DISABLED' if disable_hot_reload else 'ENABLED'}."
)

try:
    while True:
        logger.debug(
            f"[{datetime.now(timezone.utc).isoformat()}] --- Top of main loop ---"
        )  # Added log
        # --- Check for Code Updates ---
        if not disable_hot_reload:
            logger.debug(
                f"[{datetime.now(timezone.utc).isoformat()}] Checking for code updates..."
            )  # Added log
            if entrypoint_abs_path:  # Should always be set after init
                try:
                    current_mtime = os.path.getmtime(entrypoint_abs_path)
                    if current_mtime > last_load_mtime:
                        logger.info(
                            f"[Consumer] Detected change in entrypoint file: {entrypoint_abs_path}. Reloading code..."
                        )
                        (
                            reloaded_target_func,
                            reloaded_init_func,
                            reloaded_module,
                            reloaded_namespace,
                            new_mtime,
                        ) = load_or_reload_user_code(
                            _module_path,
                            _function_name,
                            entrypoint_abs_path,
                            _init_func_name,
                            _included_object_sources,
                        )

                        if (
                            reloaded_target_func is not None
                            and reloaded_module is not None
                        ):
                            logger.info(
                                "[Consumer] Code reload successful. Updating functions."
                            )
                            target_function = reloaded_target_func
                            init_function = reloaded_init_func  # Update init ref too, though it's already run
                            imported_module = reloaded_module
                            local_namespace = (
                                reloaded_namespace  # Update namespace from includes
                            )
                            last_load_mtime = new_mtime
                        else:
                            logger.warning(
                                "[Consumer] Code reload failed. Continuing with previously loaded code."
                            )
                            # Optionally: Send an alert/log prominently that reload failed

                except FileNotFoundError:
                    logger.error(
                        f"[Consumer] Error: Entrypoint file '{entrypoint_abs_path}' not found during check. Cannot reload."
                    )
                    # Mark as non-runnable? Or just log?
                    target_function = None  # Stop processing until file reappears?
                    imported_module = None
                    last_load_mtime = 0  # Reset mtime to force check next time
                except Exception as e_reload_check:
                    logger.error(
                        f"[Consumer] Error checking/reloading code: {e_reload_check}"
                    )
                    logger.exception("Code Reload Check Error Traceback:")
            else:
                logger.warning(
                    "[Consumer] Warning: Entrypoint absolute path not set, cannot check for code updates."
                )
            logger.debug(
                f"[{datetime.now(timezone.utc).isoformat()}] Finished checking for code updates."
            )  # Added log
        else:
            # Log that hot reload is skipped if it's disabled
            logger.debug(
                f"[{datetime.now(timezone.utc).isoformat()}] Hot reload check skipped (NEBU_DISABLE_HOT_RELOAD=1)."
            )

        # --- Claim Old Pending Messages ---
        logger.debug(
            f"[{datetime.now(timezone.utc).isoformat()}] Checking for pending messages to claim..."
        )  # Added log
        try:
            if target_function is not None:  # Only claim if we can process
                assert isinstance(REDIS_STREAM, str)
                assert isinstance(REDIS_CONSUMER_GROUP, str)

                # Claim messages pending for longer than MIN_IDLE_TIME_MS for *this* consumer
                # xautoclaim returns (next_id, claimed_messages_list)
                # Note: We don't need next_id if we always start from '0-0'
                # but redis-py < 5 requires it to be handled.
                # We only get messages assigned to *this* consumer_name
                claim_result = r.xautoclaim(
                    name=REDIS_STREAM,
                    groupname=REDIS_CONSUMER_GROUP,
                    consumername=consumer_name,
                    min_idle_time=MIN_IDLE_TIME_MS,
                    start_id="0-0",  # Check from the beginning of the PEL
                    count=CLAIM_COUNT,
                )

                # Compatibility check for redis-py versions
                # Newer versions (>=5.0) return a tuple: (next_id, messages, count_deleted)
                # Older versions (e.g., 4.x) return a list: [next_id, messages] or just messages if redis < 6.2
                # We primarily care about the 'messages' part.
                claimed_messages = None
                if isinstance(claim_result, tuple) and len(claim_result) >= 2:
                    # next_id_bytes, claimed_messages = claim_result # Original structure
                    _next_id, claimed_messages_list = claim_result[
                        :2
                    ]  # Handle tuple structure (>=5.0)
                    # claimed_messages need to be processed like xreadgroup results
                    # Wrap in the stream name structure expected by the processing loop
                    if claimed_messages_list:
                        # Assume decode_responses=True is set, so use string directly
                        claimed_messages = [(REDIS_STREAM, claimed_messages_list)]

                elif isinstance(claim_result, list) and claim_result:
                    # Handle older redis-py versions or direct message list if redis server < 6.2
                    # Check if the first element might be the next_id
                    if isinstance(
                        claim_result[0], (str, bytes)
                    ):  # Likely [next_id, messages] structure
                        if len(claim_result) > 1 and isinstance(claim_result[1], list):
                            _next_id, claimed_messages_list = claim_result[:2]
                            if claimed_messages_list:
                                # Assume decode_responses=True is set
                                claimed_messages = [
                                    (REDIS_STREAM, claimed_messages_list)
                                ]
                    elif isinstance(
                        claim_result[0], tuple
                    ):  # Direct list of messages [[id, data], ...]
                        claimed_messages_list = claim_result
                        # Assume decode_responses=True is set
                        claimed_messages = [(REDIS_STREAM, claimed_messages_list)]

                if claimed_messages:
                    # Process claimed messages immediately
                    # Cast messages to expected type to satisfy type checker
                    typed_messages = cast(
                        List[Tuple[str, List[Tuple[str, Dict[str, str]]]]],
                        claimed_messages,
                    )
                    # Log after casting and before processing
                    num_claimed = len(typed_messages[0][1]) if typed_messages else 0
                    logger.info(
                        f"[{datetime.now(timezone.utc).isoformat()}] Claimed {num_claimed} pending message(s). Processing..."
                    )
                    stream_name_str, stream_messages = typed_messages[0]
                    for (
                        message_id_str,
                        message_data_str_dict,
                    ) in stream_messages:
                        logger.info(
                            f"[Consumer] Processing claimed message {message_id_str}"
                        )
                        process_message(message_id_str, message_data_str_dict)
                    # After processing claimed messages, loop back to check for more potentially
                    # This avoids immediately blocking on XREADGROUP if there were claimed messages
                    continue
                else:  # Added log
                    logger.debug(
                        f"[{datetime.now(timezone.utc).isoformat()}] No pending messages claimed."
                    )  # Added log

        except ResponseError as e_claim:
            # Handle specific errors like NOGROUP gracefully if needed
            if "NOGROUP" in str(e_claim):
                logger.critical(
                    f"Consumer group {REDIS_CONSUMER_GROUP} not found during xautoclaim. Exiting."
                )
                sys.exit(1)
            else:
                logger.error(f"[Consumer] Error during XAUTOCLAIM: {e_claim}")
                # Decide if this is fatal or recoverable
                logger.error(
                    f"[{datetime.now(timezone.utc).isoformat()}] Error during XAUTOCLAIM: {e_claim}"
                )  # Added log
                time.sleep(5)  # Wait before retrying claim
        except ConnectionError as e_claim_conn:
            logger.error(
                f"Redis connection error during XAUTOCLAIM: {e_claim_conn}. Will attempt reconnect in main loop."
            )
            # Let the main ConnectionError handler below deal with reconnection
            logger.error(
                f"[{datetime.now(timezone.utc).isoformat()}] Redis connection error during XAUTOCLAIM: {e_claim_conn}. Will attempt reconnect."
            )  # Added log
            time.sleep(5)  # Avoid tight loop on connection errors during claim
        except Exception as e_claim_other:
            logger.error(
                f"[Consumer] Unexpected error during XAUTOCLAIM/processing claimed messages: {e_claim_other}"
            )
            logger.error(
                f"[{datetime.now(timezone.utc).isoformat()}] Unexpected error during XAUTOCLAIM/processing claimed: {e_claim_other}"
            )  # Added log
            logger.exception("XAUTOCLAIM/Processing Error Traceback:")
            time.sleep(5)  # Wait before retrying

        # --- Read New Messages from Redis Stream ---
        if target_function is None:
            # If code failed to load initially or during reload, wait before retrying
            logger.warning(
                "[Consumer] Target function not loaded, waiting 5s before checking again..."
            )
            time.sleep(5)
            continue  # Skip reading from Redis until code is loaded

        assert isinstance(REDIS_STREAM, str)
        assert isinstance(REDIS_CONSUMER_GROUP, str)

        streams_arg: Dict[str, str] = {REDIS_STREAM: ">"}

        # With decode_responses=True, redis-py expects str types here
        logger.debug(
            f"[{datetime.now(timezone.utc).isoformat()}] Calling xreadgroup (block=5000ms)..."
        )  # Added log
        messages = r.xreadgroup(
            REDIS_CONSUMER_GROUP,
            consumer_name,
            streams_arg,  # type: ignore[arg-type] # Suppress linter warning
            count=1,
            block=5000,  # Use milliseconds for block
        )

        if not messages:
            logger.trace(
                f"[{datetime.now(timezone.utc).isoformat()}] xreadgroup timed out (no new messages)."
            )  # Added log
            # logger.debug("[Consumer] No new messages.") # Reduce verbosity
            continue
        # Removed the else block here

        # If we reached here, messages is not empty.
        # Assert messages is not None to help type checker (already implied by `if not messages`)
        assert messages is not None

        # Cast messages to expected type to satisfy type checker (do it once)
        typed_messages = cast(
            List[Tuple[str, List[Tuple[str, Dict[str, str]]]]], messages
        )
        stream_name_str, stream_messages = typed_messages[0]
        num_msgs = len(stream_messages)

        # Log reception and count before processing
        logger.info(
            f"[{datetime.now(timezone.utc).isoformat()}] xreadgroup returned {num_msgs} message(s). Processing..."
        )  # Moved and combined log

        # Process the received messages
        # for msg_id_bytes, msg_data_bytes_dict in stream_messages: # Original structure
        for (
            message_id_str,
            message_data_str_dict,
        ) in stream_messages:  # Structure with decode_responses=True
            # message_id_str = msg_id_bytes.decode('utf-8') # No longer needed
            # Decode keys/values in the message data dict
            # message_data_str_dict = { k.decode('utf-8'): v.decode('utf-8')
            #                          for k, v in msg_data_bytes_dict.items() } # No longer needed
            # print(f"Processing message {message_id_str}") # Reduce verbosity
            # print(f"Message data: {message_data_str_dict}") # Reduce verbosity
            process_message(message_id_str, message_data_str_dict)

except ConnectionError as e:
    logger.error(f"Redis connection error: {e}. Reconnecting in 5s...")
    time.sleep(5)
    # Attempt to reconnect explicitly
    try:
        logger.info("Attempting Redis reconnection...")
        # Close existing potentially broken connection? `r.close()` if available
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        logger.info("Reconnected to Redis.")
    except Exception as recon_e:
        logger.error(f"Failed to reconnect to Redis: {recon_e}")
        # Keep waiting

except ResponseError as e:
    logger.error(f"Redis command error: {e}")
    # Should we exit or retry?
    if "NOGROUP" in str(e):
        logger.critical("Consumer group seems to have disappeared. Exiting.")
        sys.exit(1)
    time.sleep(1)

except Exception as e:
    logger.error(f"Unexpected error in main loop: {e}")
    logger.exception("Main Loop Error Traceback:")
    time.sleep(1)

finally:
    logger.info("Consumer loop exited.")
    # Any other cleanup needed?
