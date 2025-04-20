#!/usr/bin/env python3
import importlib
import json
import os
import socket
import sys
import time
import traceback
import types
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

import redis
import socks
from redis import ConnectionError, ResponseError

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

    print(f"[Code Loader] Attempting to load/reload module: '{module_path}'")
    os.environ[_NEBU_INSIDE_CONSUMER_ENV_VAR] = "1"  # Set guard *before* import/reload
    print(f"[Code Loader] Set environment variable {_NEBU_INSIDE_CONSUMER_ENV_VAR}=1")

    try:
        current_mtime = os.path.getmtime(entrypoint_abs_path)

        # Execute included object sources FIRST (if any)
        if included_object_sources:
            print("[Code Loader] Executing @include object sources...")
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
                    print(
                        f"[Code Loader] Successfully executed included object {i} base source"
                    )
                    for j, arg_source in enumerate(args_sources):
                        try:
                            exec(arg_source, exec_namespace)
                            print(
                                f"[Code Loader] Successfully executed included object {i} arg {j} source"
                            )
                        except Exception as e_arg:
                            print(
                                f"Error executing included object {i} arg {j} source: {e_arg}"
                            )
                            traceback.print_exc()  # Log specific error but continue? Or fail reload?
                except Exception as e_base:
                    print(f"Error executing included object {i} base source: {e_base}")
                    traceback.print_exc()  # Log specific error but continue? Or fail reload?
            print("[Code Loader] Finished executing included object sources.")

        # Check if module is already loaded and needs reload
        if module_path in sys.modules:
            print(
                f"[Code Loader] Module '{module_path}' already imported. Reloading..."
            )
            # Pass the exec_namespace as globals? Usually reload works within its own context.
            # If included objects *modify* the module's global scope upon exec,
            # reload might not pick that up easily. Might need a fresh import instead.
            # Let's try reload first.
            loaded_module = importlib.reload(sys.modules[module_path])
            print(f"[Code Loader] Successfully reloaded module: {module_path}")
        else:
            # Import the main module
            loaded_module = importlib.import_module(module_path)
            print(
                f"[Code Loader] Successfully imported module for the first time: {module_path}"
            )

        # Get the target function from the loaded/reloaded module
        loaded_target_func = getattr(loaded_module, function_name)
        print(
            f"[Code Loader] Successfully loaded function '{function_name}' from module '{module_path}'"
        )

        # Get the init function if specified
        if init_func_name:
            loaded_init_func = getattr(loaded_module, init_func_name)
            print(
                f"[Code Loader] Successfully loaded init function '{init_func_name}' from module '{module_path}'"
            )
            # Execute init_func
            print(f"[Code Loader] Executing init_func: {init_func_name}...")
            loaded_init_func()  # Call the function
            print(f"[Code Loader] Successfully executed init_func: {init_func_name}")

        print("[Code Loader] Code load/reload successful.")
        return (
            loaded_target_func,
            loaded_init_func,
            loaded_module,
            exec_namespace,
            current_mtime,
        )

    except FileNotFoundError:
        print(
            f"[Code Loader] Error: Entrypoint file not found at '{entrypoint_abs_path}'. Cannot load/reload."
        )
        return None, None, None, {}, 0.0  # Indicate failure
    except ImportError as e:
        print(f"[Code Loader] Error importing/reloading module '{module_path}': {e}")
        traceback.print_exc()
        return None, None, None, {}, 0.0  # Indicate failure
    except AttributeError as e:
        print(
            f"[Code Loader] Error accessing function '{function_name}' or '{init_func_name}' in module '{module_path}': {e}"
        )
        traceback.print_exc()
        return None, None, None, {}, 0.0  # Indicate failure
    except Exception as e:
        print(f"[Code Loader] Unexpected error during code load/reload: {e}")
        traceback.print_exc()
        return None, None, None, {}, 0.0  # Indicate failure
    finally:
        # Unset the guard environment variable
        os.environ.pop(_NEBU_INSIDE_CONSUMER_ENV_VAR, None)
        print(
            f"[Code Loader] Unset environment variable {_NEBU_INSIDE_CONSUMER_ENV_VAR}"
        )


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
        print(
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
                print(
                    f"[Consumer] Found entrypoint absolute path via PYTHONPATH: {entrypoint_abs_path}"
                )
                break
        if not found_path:
            print(
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
        print(
            f"FATAL: Entrypoint '{_entrypoint_rel_path}' resolves to ambiguous top-level __init__. Please use a named file or package."
        )
        sys.exit(1)
    if not _module_path:
        print(
            f"FATAL: Could not derive a valid module path from entrypoint '{_entrypoint_rel_path}'"
        )
        sys.exit(1)

    print(
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
        print("FATAL: Initial load of user code failed. Exiting.")
        sys.exit(1)
    print(
        f"[Consumer] Initial code load successful. Last modified time: {last_load_mtime}"
    )


except Exception as e:
    print(f"FATAL: Error during initial environment setup or code load: {e}")
    traceback.print_exc()
    sys.exit(1)

# Get Redis connection parameters from environment
REDIS_URL = os.environ.get("REDIS_URL", "")
REDIS_CONSUMER_GROUP = os.environ.get("REDIS_CONSUMER_GROUP")
REDIS_STREAM = os.environ.get("REDIS_STREAM")

if not all([REDIS_URL, REDIS_CONSUMER_GROUP, REDIS_STREAM]):
    print("Missing required Redis environment variables")
    sys.exit(1)

# Configure SOCKS proxy before connecting to Redis
# Use the proxy settings provided by tailscaled
socks.set_default_proxy(socks.SOCKS5, "localhost", 1055)
socket.socket = socks.socksocket
print("Configured SOCKS5 proxy for socket connections via localhost:1055")

# Connect to Redis
try:
    # Parse the Redis URL to handle potential credentials or specific DBs if needed
    # Although from_url should work now with the patched socket
    r = redis.from_url(
        REDIS_URL, decode_responses=True
    )  # Added decode_responses for convenience
    r.ping()  # Test connection
    redis_info = REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL
    print(f"Connected to Redis via SOCKS proxy at {redis_info}")
except Exception as e:
    print(f"Failed to connect to Redis via SOCKS proxy: {e}")
    traceback.print_exc()
    sys.exit(1)

# Create consumer group if it doesn't exist
try:
    # Assert types before use
    assert isinstance(REDIS_STREAM, str)
    assert isinstance(REDIS_CONSUMER_GROUP, str)
    r.xgroup_create(REDIS_STREAM, REDIS_CONSUMER_GROUP, id="0", mkstream=True)
    print(f"Created consumer group {REDIS_CONSUMER_GROUP} for stream {REDIS_STREAM}")
except ResponseError as e:
    if "BUSYGROUP" in str(e):
        print(f"Consumer group {REDIS_CONSUMER_GROUP} already exists")
    else:
        print(f"Error creating consumer group: {e}")
        traceback.print_exc()


# Function to process messages
def process_message(message_id: str, message_data: Dict[str, str]) -> None:
    # Access the globally managed user code elements
    global target_function, imported_module, local_namespace

    # Check if target_function is loaded (might be None if reload failed)
    if target_function is None or imported_module is None:
        print(
            f"Error processing message {message_id}: User code (target_function or module) is not loaded. Skipping."
        )
        # Decide how to handle this - skip and ack? Send error?
        # Sending error for now, but not acking yet.
        # This requires the main loop to handle potential ack failure later if needed.
        _send_error_response(
            message_id,
            "User code is not loaded (likely due to a failed reload)",
            traceback.format_exc(),
            None,
            None,
        )  # Pass None for user_id if unavailable here
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

        # print(f"Raw payload: {raw_payload}") # Reduce verbosity

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

        # --- Health Check Logic (Keep as is) ---
        if kind == "HealthCheck":
            print(f"Received HealthCheck message {message_id}")
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
                print(f"Sent health check response to {return_stream}")

            # Assert types again closer to usage for type checker clarity
            assert isinstance(REDIS_STREAM, str)
            assert isinstance(REDIS_CONSUMER_GROUP, str)
            r.xack(REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id)
            print(f"Acknowledged HealthCheck message {message_id}")
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

        # print(f"Content: {content}") # Reduce verbosity

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
                            print(
                                f"Warning: Content type class '{content_type_name}' not found in imported module or includes."
                            )
                        else:
                            print(f"Found content model class: {content_model_class}")
                    except AttributeError:
                        print(
                            f"Warning: Content type class '{content_type_name}' not found in imported module."
                        )
                    except Exception as e:
                        print(
                            f"Warning: Error resolving content type class '{content_type_name}': {e}"
                        )

                if content_model_class:
                    try:
                        content_model = content_model_class.model_validate(content)
                        print(f"Validated content model: {content_model}")
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
                        )
                    except Exception as e:
                        print(
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
                            print(
                                f"Warning: Input type class '{param_type_name}' not found. Passing raw content."
                            )
                        input_obj = content
                    else:
                        print(f"Found input model class: {input_type_class}")
                        input_obj = input_type_class.model_validate(content)
                        print(f"Validated input model: {input_obj}")
                except AttributeError:
                    print(
                        f"Warning: Input type class '{param_type_name}' not found in imported module."
                    )
                    input_obj = content
                except Exception as e:
                    print(
                        f"Error resolving/validating input type '{param_type_name}': {e}. Passing raw content."
                    )
                    input_obj = content

        except NameError as e:
            print(
                f"Error: Required class (e.g., Message or parameter type) not found. Import failed? {e}"
            )
            # Can't proceed without types, re-raise or handle error response
            raise RuntimeError(f"Required class not found: {e}") from e
        except Exception as e:
            print(f"Error constructing input object: {e}")
            raise  # Re-raise unexpected errors during input construction

        # print(f"Input object: {input_obj}") # Reduce verbosity

        # Execute the function
        print(f"Executing function...")
        result = target_function(input_obj)
        print(f"Result: {result}")  # Reduce verbosity

        # Convert result to dict if it's a Pydantic model
        if hasattr(result, "model_dump"):  # Use model_dump for Pydantic v2+
            result_content = result.model_dump(mode="json")  # Serialize properly
        elif hasattr(result, "dict"):  # Fallback for older Pydantic
            result_content = result.dict()
        else:
            result_content = result  # Assume JSON-serializable

        # Prepare the response
        response = {
            "kind": "StreamResponseMessage",
            "id": message_id,
            "content": result_content,
            "status": "success",
            "created_at": datetime.now().isoformat(),
            "user_id": user_id,  # Pass user_id back
        }

        # print(f"Response: {response}") # Reduce verbosity

        # Send the result to the return stream
        if return_stream:
            assert isinstance(return_stream, str)
            r.xadd(return_stream, {"data": json.dumps(response)})
            print(f"Processed message {message_id}, result sent to {return_stream}")

        # Acknowledge the message
        assert isinstance(REDIS_STREAM, str)
        assert isinstance(REDIS_CONSUMER_GROUP, str)
        r.xack(REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id)

    except Exception as e:
        print(f"Error processing message {message_id}: {e}")
        traceback.print_exc()
        _send_error_response(
            message_id, str(e), traceback.format_exc(), return_stream, user_id
        )

        # Acknowledge the message even if processing failed
        try:
            assert isinstance(REDIS_STREAM, str)
            assert isinstance(REDIS_CONSUMER_GROUP, str)
            r.xack(REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id)
            print(f"Acknowledged failed message {message_id}")
        except Exception as e_ack:
            print(
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
        print(f"Sent error response for message {message_id} to {error_destination}")
    except Exception as e_redis:
        print(
            f"CRITICAL: Failed to send error response for {message_id} to Redis: {e_redis}"
        )
        traceback.print_exc()


# Main loop
print(f"Starting consumer for stream {REDIS_STREAM} in group {REDIS_CONSUMER_GROUP}")
consumer_name = f"consumer-{os.getpid()}-{socket.gethostname()}"  # More unique name

try:
    while True:
        print("reading from stream...")
        try:
            # --- Check for Code Updates ---
            if entrypoint_abs_path:  # Should always be set after init
                try:
                    current_mtime = os.path.getmtime(entrypoint_abs_path)
                    if current_mtime > last_load_mtime:
                        print(
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
                            print(
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
                            print(
                                "[Consumer] Code reload failed. Continuing with previously loaded code."
                            )
                            # Optionally: Send an alert/log prominently that reload failed

                except FileNotFoundError:
                    print(
                        f"[Consumer] Error: Entrypoint file '{entrypoint_abs_path}' not found during check. Cannot reload."
                    )
                    # Mark as non-runnable? Or just log?
                    target_function = None  # Stop processing until file reappears?
                    imported_module = None
                    last_load_mtime = 0  # Reset mtime to force check next time
                except Exception as e_reload_check:
                    print(f"[Consumer] Error checking/reloading code: {e_reload_check}")
                    traceback.print_exc()
            else:
                print(
                    "[Consumer] Warning: Entrypoint absolute path not set, cannot check for code updates."
                )

            # --- Read from Redis Stream ---
            if target_function is None:
                # If code failed to load initially or during reload, wait before retrying
                print(
                    "[Consumer] Target function not loaded, waiting 5s before checking again..."
                )
                time.sleep(5)
                continue  # Skip reading from Redis until code is loaded

            assert isinstance(REDIS_STREAM, str)
            assert isinstance(REDIS_CONSUMER_GROUP, str)

            streams_arg: Dict[str, str] = {REDIS_STREAM: ">"}

            # With decode_responses=True, redis-py expects str types here
            messages = r.xreadgroup(
                REDIS_CONSUMER_GROUP,
                consumer_name,
                streams_arg,  # type: ignore[arg-type]
                count=1,
                block=5000,  # Use milliseconds for block
            )

            if not messages:
                # print("[Consumer] No new messages.") # Reduce verbosity
                continue

            # Assert messages is not None to help type checker (already implied by `if not messages`)
            assert messages is not None

            # Cast messages to expected type to satisfy type checker
            typed_messages = cast(
                List[Tuple[str, List[Tuple[str, Dict[str, str]]]]], messages
            )
            stream_name_str, stream_messages = typed_messages[0]

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
            print(f"Redis connection error: {e}. Reconnecting in 5s...")
            time.sleep(5)
            # Attempt to reconnect explicitly
            try:
                print("Attempting Redis reconnection...")
                # Close existing potentially broken connection? `r.close()` if available
                r = redis.from_url(REDIS_URL, decode_responses=True)
                r.ping()
                print("Reconnected to Redis.")
            except Exception as recon_e:
                print(f"Failed to reconnect to Redis: {recon_e}")
                # Keep waiting

        except ResponseError as e:
            print(f"Redis command error: {e}")
            # Should we exit or retry?
            if "NOGROUP" in str(e):
                print("Consumer group seems to have disappeared. Exiting.")
                sys.exit(1)
            time.sleep(1)

        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
            traceback.print_exc()
            time.sleep(1)

finally:
    print("Consumer loop exited.")
    # Any other cleanup needed?
