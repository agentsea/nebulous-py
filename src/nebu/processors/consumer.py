#!/usr/bin/env python3
import importlib
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, TypeVar, Any, Optional, List, Tuple

import redis
import socks
from redis import ConnectionError, ResponseError

# Define TypeVar for generic models
T = TypeVar("T")

# Environment variable name used as a guard in the decorator
_NEBU_INSIDE_CONSUMER_ENV_VAR = "_NEBU_INSIDE_CONSUMER_EXEC"

# --- Get Environment Variables ---
try:
    # Core function info
    function_name = os.environ.get("FUNCTION_NAME")
    entrypoint_rel_path = os.environ.get("NEBU_ENTRYPOINT_MODULE_PATH")

    # Type info (still used for message processing logic)
    is_stream_message = os.environ.get("IS_STREAM_MESSAGE") == "True"
    param_type_str = os.environ.get("PARAM_TYPE_STR") # For potential validation/logging
    return_type_str = os.environ.get("RETURN_TYPE_STR") # For potential validation/logging
    content_type_name = os.environ.get("CONTENT_TYPE_NAME") # For Message<T> construction

    # Init func info
    init_func_name = os.environ.get("INIT_FUNC_NAME")

    # Included object sources (keep for now)
    included_object_sources = []
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
            included_object_sources.append((obj_source, args))
            i += 1
        else:
            break

    if not function_name or not entrypoint_rel_path:
        print("FUNCTION_NAME or NEBU_ENTRYPOINT_MODULE_PATH environment variables not set")
        sys.exit(1)

    # Convert entrypoint file path to module path
    # e.g., subdir/my_func.py -> subdir.my_func
    # e.g., my_func.py -> my_func
    # e.g., subdir/__init__.py -> subdir
    module_path = entrypoint_rel_path.replace(os.sep, '.')
    if module_path.endswith('.py'):
        module_path = module_path[:-3]
    if module_path.endswith('.__init__'):
        module_path = module_path[:-len('.__init__')]
    elif module_path == '__init__': # Top-level __init__.py
        # This case is ambiguous. What should it import? The package itself?
        # For now, assume it means the top-level package, represented by an empty string
        # which importlib might not handle well. Let's require named files or packages.
        print(f"Error: Entrypoint '{entrypoint_rel_path}' resolves to ambiguous top-level __init__. Please use a named file or package.")
        sys.exit(1)
    # If module_path becomes empty, it means entrypoint was likely just '__init__.py'
    if not module_path:
        print(f"Error: Could not derive a valid module path from entrypoint '{entrypoint_rel_path}'")
        sys.exit(1)

    print(f"[Consumer] Attempting to import entrypoint module: '{module_path}' from PYTHONPATH: {os.environ.get('PYTHONPATH')}")

    # --- Dynamically Import and Load Function --- (Replaces old exec logic)
    target_function = None
    init_function = None
    local_namespace: Dict[str, Any] = {} # Keep namespace for included objects for now

    # Set the guard environment variable *before* importing user code
    os.environ[_NEBU_INSIDE_CONSUMER_ENV_VAR] = "1"
    print(f"[Consumer] Set environment variable {_NEBU_INSIDE_CONSUMER_ENV_VAR}=1")

    try:
        # Execute included object sources FIRST (if any)
        # These might define things needed during the import of the main module
        if included_object_sources:
            print("[Consumer] Executing @include object sources...")
            # Include necessary imports for the exec context
            exec("from pydantic import BaseModel, Field", local_namespace)
            exec("from typing import Optional, List, Dict, Any, Generic, TypeVar", local_namespace)
            exec("T_exec = TypeVar('T_exec')", local_namespace)
            exec("from nebu.processors.models import *", local_namespace)
            # exec("from nebu.processors.processor import *", local_namespace) # Likely not needed
            # exec("from nebu.processors.decorate import processor", local_namespace) # Avoid re-running decorator
            # exec("from nebu.chatx.openai import *", local_namespace) # Add if needed by included objects

            for i, (obj_source, args_sources) in enumerate(included_object_sources):
                try:
                    exec(obj_source, local_namespace)
                    print(f"[Consumer] Successfully executed included object {i} base source")
                    for j, arg_source in enumerate(args_sources):
                        try:
                            exec(arg_source, local_namespace)
                            print(f"[Consumer] Successfully executed included object {i} arg {j} source")
                        except Exception as e_arg:
                            print(f"Error executing included object {i} arg {j} source: {e_arg}")
                            traceback.print_exc()
                except Exception as e_base:
                    print(f"Error executing included object {i} base source: {e_base}")
                    traceback.print_exc()
            print("[Consumer] Finished executing included object sources.")

        # Import the main module where the decorated function resides
        imported_module = importlib.import_module(module_path)
        print(f"Successfully imported module: {module_path}")

        # Get the target function from the imported module
        target_function = getattr(imported_module, function_name)
        print(f"Successfully loaded function '{function_name}' from module '{module_path}'")

        # Get the init function if specified
        if init_func_name:
            try:
                init_function = getattr(imported_module, init_func_name)
                print(f"Successfully loaded init function '{init_func_name}' from module '{module_path}'")
                # Execute init_func now
                print(f"Executing init_func: {init_func_name}...")
                init_function() # Call the function
                print(f"Successfully executed init_func: {init_func_name}")
            except AttributeError:
                print(f"Error: Init function '{init_func_name}' not found in module '{module_path}'")
                # Decide if this is fatal. Exit for now.
                sys.exit(1)
            except Exception as e:
                print(f"Error executing init_func '{init_func_name}': {e}")
                traceback.print_exc()
                print("Exiting due to init_func failure.")
                sys.exit(1)

    except ImportError as e:
        print(f"Error importing module '{module_path}': {e}")
        print("Please ensure the module exists within the synced code directory and all its dependencies are installed.")
        traceback.print_exc()
        sys.exit(1)
    except AttributeError as e:
        print(f"Error getting function '{function_name}' from module '{module_path}': {e}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Error during dynamic import or function loading: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Unset the guard environment variable
        os.environ.pop(_NEBU_INSIDE_CONSUMER_ENV_VAR, None)
        print(f"[Consumer] Unset environment variable {_NEBU_INSIDE_CONSUMER_ENV_VAR}")

    # Ensure target_function is loaded
    if target_function is None:
        print("Error: Target function was not loaded successfully.")
        sys.exit(1)

except Exception as e:
    print(f"Error during initial environment setup or import: {e}")
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
    return_stream = None
    user_id = None
    try:
        payload_str = message_data.get("data")
        if not payload_str or not isinstance(payload_str, str):
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
        created_at_str = raw_payload.get("created_at") # Get as string or None
        # Attempt to parse created_at, fallback to now()
        try:
            created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(timezone.utc)
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
            return # Exit early for health checks
        # --- End Health Check Logic ---

        # Parse content if it's a string (e.g., double-encoded JSON)
        if isinstance(content_raw, str):
            try:
                content = json.loads(content_raw)
            except json.JSONDecodeError:
                content = content_raw # Keep as string if not valid JSON
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
            from nebu.processors.models import Message # Import needed message class

            if is_stream_message:
                message_class = Message # Use imported class
                content_model_class = None
                if content_type_name:
                    try:
                        # Assume content_type_name refers to a class available in the global scope
                        # (either from imported module or included objects)
                        content_model_class = getattr(imported_module, content_type_name, None)
                        if content_model_class is None:
                            # Check in local_namespace from included objects as fallback?
                            content_model_class = local_namespace.get(content_type_name)
                        if content_model_class is None:
                            print(f"Warning: Content type class '{content_type_name}' not found in imported module or includes.")
                        else:
                            print(f"Found content model class: {content_model_class}")
                    except AttributeError:
                        print(f"Warning: Content type class '{content_type_name}' not found in imported module.")
                    except Exception as e:
                        print(f"Warning: Error resolving content type class '{content_type_name}': {e}")

                if content_model_class:
                    try:
                        content_model = content_model_class.model_validate(content)
                        print(f"Validated content model: {content_model}")
                        input_obj = message_class(
                            kind=kind, id=msg_id, content=content_model, created_at=created_at,
                            return_stream=return_stream, user_id=user_id, orgs=orgs, handle=handle, adapter=adapter
                        )
                    except Exception as e:
                        print(f"Error validating/creating content model '{content_type_name}': {e}. Falling back.")
                        # Fallback to raw content in Message
                        input_obj = message_class(
                            kind=kind, id=msg_id, content=content, created_at=created_at,
                            return_stream=return_stream, user_id=user_id, orgs=orgs, handle=handle, adapter=adapter
                        )
                else:
                    # No content type name or class found, use raw content
                    input_obj = message_class(
                        kind=kind, id=msg_id, content=content, created_at=created_at,
                        return_stream=return_stream, user_id=user_id, orgs=orgs, handle=handle, adapter=adapter
                    )
            else: # Not a stream message, use the function's parameter type
                param_type_name = param_type_str # Assume param_type_str holds the class name
                # Attempt to resolve the parameter type class
                try:
                    input_type_class = getattr(imported_module, param_type_name, None)
                    if input_type_class is None:
                        input_type_class = local_namespace.get(param_type_name)
                    if input_type_class is None:
                        print(f"Warning: Input type class '{param_type_name}' not found. Passing raw content.")
                        input_obj = content
                    else:
                        print(f"Found input model class: {input_type_class}")
                        input_obj = input_type_class.model_validate(content)
                        print(f"Validated input model: {input_obj}")
                except AttributeError:
                    print(f"Warning: Input type class '{param_type_name}' not found in imported module.")
                    input_obj = content
                except Exception as e:
                    print(f"Error resolving/validating input type '{param_type_name}': {e}. Passing raw content.")
                    input_obj = content

        except NameError as e:
            print(f"Error: Required class (e.g., Message or parameter type) not found. Import failed? {e}")
            # Can't proceed without types, re-raise or handle error response
            raise RuntimeError(f"Required class not found: {e}") from e
        except Exception as e:
            print(f"Error constructing input object: {e}")
            raise # Re-raise unexpected errors during input construction

        # print(f"Input object: {input_obj}") # Reduce verbosity

        # Execute the function
        print(f"Executing function...")
        result = target_function(input_obj)
        print(f"Result: {result}") # Reduce verbosity

        # Convert result to dict if it's a Pydantic model
        if hasattr(result, "model_dump"): # Use model_dump for Pydantic v2+
            result_content = result.model_dump(mode='json') # Serialize properly
        elif hasattr(result, "dict"): # Fallback for older Pydantic
            result_content = result.dict()
        else:
            result_content = result # Assume JSON-serializable

        # Prepare the response
        response = {
            "kind": "StreamResponseMessage",
            "id": message_id,
            "content": result_content,
            "status": "success",
            "created_at": datetime.now().isoformat(),
            "user_id": user_id, # Pass user_id back
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

        error_response = {
            "kind": "StreamResponseMessage",
            "id": message_id,
            "content": {
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            "status": "error",
            "created_at": datetime.now().isoformat(),
            "user_id": user_id,
        }

        # Send error response
        error_destination = f"{REDIS_STREAM}.errors" # Default error stream
        if return_stream: # Prefer return_stream if available
            error_destination = return_stream

        try:
            assert isinstance(error_destination, str)
            r.xadd(error_destination, {"data": json.dumps(error_response)})
            print(f"Sent error response for message {message_id} to {error_destination}")
        except Exception as e_redis:
            print(f"CRITICAL: Failed to send error response for {message_id} to Redis: {e_redis}")

        # Acknowledge the message even if processing failed
        try:
            assert isinstance(REDIS_STREAM, str)
            assert isinstance(REDIS_CONSUMER_GROUP, str)
            r.xack(REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id)
            print(f"Acknowledged failed message {message_id}")
        except Exception as e_ack:
            print(f"CRITICAL: Failed to acknowledge failed message {message_id}: {e_ack}")


# Main loop
print(f"Starting consumer for stream {REDIS_STREAM} in group {REDIS_CONSUMER_GROUP}")
consumer_name = f"consumer-{os.getpid()}-{socket.gethostname()}" # More unique name

try:
    while True:
        try:
            assert isinstance(REDIS_STREAM, str)
            assert isinstance(REDIS_CONSUMER_GROUP, str)

            streams = {REDIS_STREAM: ">"}
            # print("Reading from stream...") # Reduce verbosity
            # Type checker fix: Provide explicit types for streams dictionary
            messages: Optional[List[Tuple[bytes, List[Tuple[bytes, Dict[bytes, bytes]]]]]]\
            messages = r.xreadgroup(
                REDIS_CONSUMER_GROUP.encode('utf-8'),
                consumer_name.encode('utf-8'),
                {REDIS_STREAM.encode('utf-8'): b'>'}, # Encode keys and stream ID
                count=1,
                block=5000
            )

            if not messages:
                continue

            # Process response (decode necessary parts)
            # Structure: [[b'stream_name', [(b'msg_id', {b'key': b'val'})]]]
            stream_name_bytes, stream_messages = messages[0]
            for msg_id_bytes, msg_data_bytes_dict in stream_messages:
                message_id_str = msg_id_bytes.decode('utf-8')
                # Decode keys/values in the message data dict
                message_data_str_dict = { k.decode('utf-8'): v.decode('utf-8')
                                         for k, v in msg_data_bytes_dict.items() }
                # print(f"Processing message {message_id_str}") # Reduce verbosity
                # print(f"Message data: {message_data_str_dict}") # Reduce verbosity
                process_message(message_id_str, message_data_str_dict)

        except ConnectionError as e:
            print(f"Redis connection error: {e}. Reconnecting in 5s...")
            time.sleep(5)
            # Attempt to reconnect or rely on redis-py's auto-reconnect?
            # For safety, let's explicitly try to reconnect here if needed
            try:
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
