#!/usr/bin/env python3
import asyncio
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

from nebulous.errors import RetriableError
from nebulous.logging import logger

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

# Global health check subprocess
health_subprocess: Optional[subprocess.Popen] = None

REDIS_CONSUMER_GROUP = os.environ.get("REDIS_CONSUMER_GROUP")
REDIS_STREAM = os.environ.get("REDIS_STREAM")
NEBU_EXECUTION_MODE = os.environ.get("NEBU_EXECUTION_MODE", "inline").lower()
execution_mode = NEBU_EXECUTION_MODE

# Define health check stream and group names
REDIS_HEALTH_STREAM = f"{REDIS_STREAM}.health" if REDIS_STREAM else None
REDIS_HEALTH_CONSUMER_GROUP = (
    f"{REDIS_CONSUMER_GROUP}-health" if REDIS_CONSUMER_GROUP else None
)

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
        # Retry logic for getmtime with progressive backoff
        max_retries = 5
        initial_delay = 0.3  # Start with 100ms
        for attempt in range(max_retries):
            logger.info(f"[Code Loader] Attempt {attempt + 1}/{max_retries}")
            try:
                logger.info(f"[Code Loader] Getting mtime for '{entrypoint_abs_path}'")
                current_mtime = os.path.getmtime(entrypoint_abs_path)
                logger.info(
                    f"[Code Loader] mtime for '{entrypoint_abs_path}' is {current_mtime}"
                )
                break  # Success
            except FileNotFoundError:
                if attempt < max_retries - 1:
                    delay = initial_delay * (2**attempt)
                    logger.warning(
                        f"[Code Loader] getmtime failed for '{entrypoint_abs_path}' (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"[Code Loader] getmtime failed for '{entrypoint_abs_path}' after {max_retries} attempts."
                    )
                    dir_path = os.path.dirname(entrypoint_abs_path)
                    if dir_path and os.path.isdir(dir_path):
                        logger.error(f"Listing contents of directory '{dir_path}':")
                        try:
                            for item in os.listdir(dir_path):
                                logger.error(f"  - {item}")
                        except OSError as e:
                            logger.error(f"    Error listing directory: {e}")
                    elif dir_path:
                        logger.error(f"Directory '{dir_path}' does not exist.")
                    else:
                        logger.error(
                            f"Could not determine directory from path '{entrypoint_abs_path}'"
                        )
                    raise  # Re-raise the final FileNotFoundError
            except Exception as e:
                logger.error(
                    f"[Code Loader] Error getting mtime for '{entrypoint_abs_path}': {e}"
                )
                logger.exception(
                    f"[Code Loader] Traceback for error getting mtime for '{entrypoint_abs_path}':"
                )
                raise  # Re-raise the final Exception

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
            exec("from nebulous.processors.models import *", exec_namespace)
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

    except FileNotFoundError as e:
        """Handle *any* FileNotFoundError raised during loading/reloading.

        Historically we assumed that a FileNotFoundError inside this function
        could only be caused by the entry-point itself going missing.  In
        practice deep imports (e.g. the model loader trying to open HF weight
        shards) can also raise the same exception.  That mis-classification
        hides the real root-cause from the logs and makes debugging harder.

        We now log the *actual* missing filename, the full traceback and make
        it explicit when the missing file is NOT the entrypoint.  This gives
        us a much clearer signal when the failure is happening in a nested
        import or during model/asset loading.
        """

        missing_file = getattr(e, "filename", "<unknown>")
        logger.error(
            f"[Code Loader] FileNotFoundError encountered while loading user code: {e} (missing_file='{missing_file}')"
        )

        # Dump full traceback so we can see exactly where the exception was
        # raised (helps identify nested import failures)
        logger.exception("[Code Loader] FileNotFoundError Traceback:")

        # If the missing file is *not* the entrypoint, call that out
        if missing_file and os.path.abspath(missing_file) != os.path.abspath(
            entrypoint_abs_path
        ):
            logger.error(
                f"[Code Loader] NOTE: The missing file is NOT the entrypoint. Entrypoint: '{entrypoint_abs_path}', missing: '{missing_file}'."
            )

        # For consistency still list the directory that *should* contain the
        # entrypoint – this is often useful even for nested failures because it
        # verifies the working directory / mount points are intact.
        dir_path = os.path.dirname(entrypoint_abs_path)
        if dir_path and os.path.isdir(dir_path):
            logger.error(f"Listing contents of directory '{dir_path}':")
            try:
                for item in os.listdir(dir_path):
                    logger.error(f"  - {item}")
            except OSError as e_list:
                logger.error(f"    Error listing directory: {e_list}")
        elif dir_path:
            logger.error(f"Directory '{dir_path}' does not exist.")
        else:
            logger.error(
                f"Could not determine directory from path '{entrypoint_abs_path}'"
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
    is_async_function = os.environ.get("IS_ASYNC_FUNCTION") == "True"
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
    # This might need adjustment based on deployment specifics
    entrypoint_abs_path = ""  # Initialize as empty
    found_path = False

    # Retry loop to wait for the entrypoint file to be synced
    max_wait_seconds = 15
    start_time = time.time()
    logger.info(f"Searching for entrypoint file '{_entrypoint_rel_path}'...")

    while time.time() - start_time < max_wait_seconds:
        python_path = os.environ.get("PYTHONPATH", "").split(os.pathsep)
        for p_path in python_path:
            # Construct path relative to the PYTHONPATH entry
            potential_path = os.path.join(p_path, _entrypoint_rel_path)
            if os.path.exists(potential_path):
                entrypoint_abs_path = potential_path
                found_path = True
                logger.info(
                    f"[Consumer] Found entrypoint absolute path via PYTHONPATH: {entrypoint_abs_path}"
                )
                break  # Exit the inner for loop
        if found_path:
            break  # Exit the while loop

        logger.debug(
            f"Entrypoint file '{_entrypoint_rel_path}' not found yet. Retrying in 1 second..."
        )
        time.sleep(1)

    if not found_path:
        logger.critical(
            f"FATAL: Could not find entrypoint file '{_entrypoint_rel_path}' in PYTHONPATH after waiting {max_wait_seconds} seconds."
        )
        python_path = os.environ.get("PYTHONPATH", "").split(os.pathsep)
        logger.critical("Searched in the following PYTHONPATH directories:")
        for p_path in python_path:
            if os.path.isdir(p_path):
                logger.critical(f"- Contents of '{p_path}':")
                try:
                    # To avoid log spam, list top-level contents only
                    contents = os.listdir(p_path)
                    for item in contents:
                        logger.critical(f"  - {item}")
                except OSError as e:
                    logger.critical(f"    Could not list directory contents: {e}")
            else:
                logger.critical(f"- '{p_path}' (is not a directory or does not exist)")

        # Attempting abspath anyway for the error message in load function
        entrypoint_abs_path = os.path.abspath(_entrypoint_rel_path)
        sys.exit(1)

    # Convert entrypoint file path to module path
    _module_path = _entrypoint_rel_path.replace(os.sep, ".")
    if _module_path.endswith(".py"):
        _module_path = _module_path[:-3]
    if _module_path.endswith(".__init__"):
        _module_path = _module_path[: -len(".__init__")]
    elif _module_path.startswith("__init__"):
        _module_path = _module_path[len("__init__.") :]
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

# Global Redis connection for the main consumer
r: redis.Redis  # Initialized by connect_redis, which sys.exits on failure


# --- Connect to Redis (Main Consumer) ---
def connect_redis(redis_url: str) -> redis.Redis:
    """Connects to Redis and returns the connection object."""
    try:
        # Parse the Redis URL to handle potential credentials or specific DBs if needed
        # Although from_url should work now with the patched socket
        logger.info(
            f"Attempting to connect to Redis at {redis_url.split('@')[-1] if '@' in redis_url else redis_url}"
        )
        conn = redis.from_url(
            redis_url, decode_responses=True
        )  # Added decode_responses for convenience
        conn.ping()  # Test connection
        redis_info = redis_url.split("@")[-1] if "@" in redis_url else redis_url
        logger.info(f"Connected to Redis via SOCKS proxy at {redis_info}")
        return conn
    except Exception as e:
        logger.critical(f"Failed to connect to Redis via SOCKS proxy: {e}")
        logger.exception("Redis Connection Error Traceback:")
        sys.exit(1)


r = connect_redis(REDIS_URL)


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


# --- Health Check Subprocess Management ---
def start_health_check_subprocess() -> Optional[subprocess.Popen]:
    """Start the health check consumer subprocess."""
    global REDIS_HEALTH_STREAM, REDIS_HEALTH_CONSUMER_GROUP

    print("[DEBUG] start_health_check_subprocess called")
    print(f"[DEBUG] REDIS_URL: {REDIS_URL}")
    print(f"[DEBUG] REDIS_HEALTH_STREAM: {REDIS_HEALTH_STREAM}")
    print(f"[DEBUG] REDIS_HEALTH_CONSUMER_GROUP: {REDIS_HEALTH_CONSUMER_GROUP}")

    if not all([REDIS_URL, REDIS_HEALTH_STREAM, REDIS_HEALTH_CONSUMER_GROUP]):
        print("[DEBUG] Health check not configured - missing required variables")
        logger.warning(
            "[Consumer] Health check stream not configured. Health consumer subprocess not started."
        )
        return None

    try:
        # Type assertions to ensure variables are strings before using them
        assert isinstance(REDIS_HEALTH_STREAM, str)
        assert isinstance(REDIS_HEALTH_CONSUMER_GROUP, str)

        print(
            f"[DEBUG] Starting health check subprocess for stream: {REDIS_HEALTH_STREAM}"
        )
        print(f"[DEBUG] Health consumer group: {REDIS_HEALTH_CONSUMER_GROUP}")

        # Prepare environment variables for the subprocess
        health_env = os.environ.copy()
        health_env["REDIS_HEALTH_STREAM"] = REDIS_HEALTH_STREAM
        health_env["REDIS_HEALTH_CONSUMER_GROUP"] = REDIS_HEALTH_CONSUMER_GROUP

        # Start the health check worker subprocess
        health_cmd = [
            sys.executable,
            "-u",  # Force unbuffered stdout/stderr
            "-m",
            "nebulous.processors.consumer_health_worker",
        ]

        print(f"[DEBUG] Health subprocess command: {' '.join(health_cmd)}")

        process = subprocess.Popen(
            health_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            encoding="utf-8",
            env=health_env,
            bufsize=1,  # Line buffered
        )

        print(
            f"[DEBUG] Health check subprocess started successfully with PID {process.pid}"
        )
        logger.info(
            f"[Consumer] Health check subprocess started with PID {process.pid}"
        )
        return process

    except Exception as e:
        print(f"[DEBUG] Failed to start health check subprocess: {e}")
        print(f"[DEBUG] Exception type: {type(e)}")
        logger.error(f"[Consumer] Failed to start health check subprocess: {e}")
        logger.exception("Health Subprocess Start Error Traceback:")
        return None


def monitor_health_subprocess(process: subprocess.Popen) -> None:
    """Monitor the health check subprocess and log its output."""
    print(f"[DEBUG] monitor_health_subprocess started for PID {process.pid}")
    try:
        # Read output from the subprocess
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                print(f"[DEBUG] [HealthSubprocess] {line.strip()}")
                logger.info(f"[HealthSubprocess] {line.strip()}")
        process.stdout.close() if process.stdout else None
        print(f"[DEBUG] monitor_health_subprocess finished for PID {process.pid}")
    except Exception as e:
        print(f"[DEBUG] Error monitoring health subprocess: {e}")
        logger.error(f"[Consumer] Error monitoring health subprocess: {e}")


def check_health_subprocess() -> bool:
    """Check if the health subprocess is still running and restart if needed."""
    global health_subprocess

    # print(f"[DEBUG] check_health_subprocess called")

    if health_subprocess is None:
        # print(f"[DEBUG] health_subprocess is None")
        return False

    # Cat the health subprocess log file
    # try:
    #     log_dir = os.path.join(os.getcwd(), "logs")
    #     log_file = os.path.join(log_dir, f"health_consumer_{health_subprocess.pid}.log")

    #     if os.path.exists(log_file):
    #         print(
    #             f"[DEBUG] === HEALTH SUBPROCESS LOG (PID {health_subprocess.pid}) ==="
    #         )
    #         try:
    #             with open(log_file, "r") as f:
    #                 log_contents = f.read()
    #                 if log_contents.strip():
    #                     print(log_contents)
    #                 else:
    #                     print("[DEBUG] Log file is empty")
    #         except Exception as e:
    #             print(f"[DEBUG] Error reading log file {log_file}: {e}")
    #         print(f"[DEBUG] === END HEALTH SUBPROCESS LOG ===")
    #     else:
    #         print(f"[DEBUG] Health subprocess log file not found: {log_file}")
    # except Exception as e:
    #     print(f"[DEBUG] Error accessing health subprocess log: {e}")

    # Check if process is still running
    poll_result = health_subprocess.poll()
    # print(f"[DEBUG] health_subprocess.poll() returned: {poll_result}")

    if poll_result is None:
        print(f"[DEBUG] Health subprocess still running (PID {health_subprocess.pid})")
        return True  # Still running

    # Process has exited
    exit_code = health_subprocess.returncode
    # print(f"[DEBUG] Health subprocess exited with code {exit_code}")
    logger.warning(
        f"[Consumer] Health subprocess exited with code {exit_code}. Restarting..."
    )

    # Start a new health subprocess
    # print(f"[DEBUG] Attempting to restart health subprocess...")
    health_subprocess = start_health_check_subprocess()

    if health_subprocess:
        # print(f"[DEBUG] Health subprocess restarted successfully")
        # Start monitoring thread for the new subprocess
        monitor_thread = threading.Thread(
            target=monitor_health_subprocess, args=(health_subprocess,), daemon=True
        )
        monitor_thread.start()
        # print(f"[DEBUG] Monitor thread started for health subprocess")
        logger.info(
            "[Consumer] Health subprocess restarted and monitoring thread started."
        )
        return True
    else:
        # print(f"[DEBUG] Failed to restart health subprocess")
        logger.error("[Consumer] Failed to restart health subprocess.")
        return False


# Function to process messages
def process_message(message_id: str, message_data: Dict[str, str]) -> None:
    # Access the globally managed user code elements
    global target_function, imported_module, local_namespace
    global execution_mode, r, REDIS_STREAM, REDIS_CONSUMER_GROUP

    print(f">>> Processing message {message_id}")

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
                "nebulous.processors.consumer_process_worker",
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
                "FATAL: Worker script 'nebulous.processors.consumer_process_worker' not found. Check PYTHONPATH."
            )
            python_path = os.environ.get("PYTHONPATH", "").split(os.pathsep)
            logger.critical("Current PYTHONPATH directories:")
            for p_path in python_path:
                if os.path.isdir(p_path):
                    logger.critical(f"- Contents of '{p_path}':")
                    try:
                        contents = os.listdir(p_path)
                        for item in contents:
                            logger.critical(f"  - {item}")
                    except OSError as e:
                        logger.critical(f"    Could not list directory contents: {e}")
                else:
                    logger.critical(
                        f"- '{p_path}' (is not a directory or does not exist)"
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
        print(f"[DEBUG] Raw message_data: {message_data}")
        print(f"[DEBUG] Payload string: {payload_str}")

        if (
            not payload_str
        ):  # Covers None and empty string, isinstance check is redundant
            raise ValueError(
                f"Missing or invalid 'data' field (expected non-empty string): {message_data}"
            )
        try:
            raw_payload = json.loads(payload_str)
            print(f"[DEBUG] Parsed raw_payload: {json.dumps(raw_payload, indent=2)}")
        except json.JSONDecodeError as json_err:
            raise ValueError(f"Failed to parse JSON payload: {json_err}") from json_err
        if not isinstance(raw_payload, dict):
            raise TypeError(
                f"Expected parsed payload to be a dictionary, but got {type(raw_payload)}"
            )

        logger.debug(f">> Raw payload: {raw_payload}")

        # --- Extract fields from the *inner* message content for HealthCheck and regular processing ---
        # The actual message content is inside raw_payload["content"]
        inner_content_data = raw_payload.get("content", {})
        print(
            f"[DEBUG] Extracted inner_content_data: {json.dumps(inner_content_data, indent=2) if isinstance(inner_content_data, dict) else inner_content_data}"
        )

        # Add debug logging for content structure analysis
        logger.debug(f">> inner_content_data type: {type(inner_content_data)}")
        logger.debug(
            f">> inner_content_data keys (if dict): {list(inner_content_data.keys()) if isinstance(inner_content_data, dict) else 'N/A'}"
        )

        if not isinstance(inner_content_data, dict):
            # If content is not a dict (e.g. already a primitive from a non-Message processor)
            # we can't reliably get 'kind' or other fields from it.
            # This case is more relevant for non-StreamMessage processors.
            # For HealthChecks, we expect a structured 'content'.
            print("[DEBUG] inner_content_data is not a dict, using defaults")
            logger.warning(
                f"Received non-dict inner_content_data: {inner_content_data}. HealthCheck might be missed if applicable."
            )
            inner_kind = ""  # Default to empty string
            inner_msg_id = ""  # Default to empty string
            actual_content_to_process = inner_content_data  # Use it as is
            inner_created_at_str = None
        else:
            # Check if this looks like a nested message structure (has kind/id/content fields)
            # vs direct content data
            has_message_structure = (
                "kind" in inner_content_data
                and "id" in inner_content_data
                and "content" in inner_content_data
            )

            print(f"[DEBUG] has_message_structure: {has_message_structure}")
            print(f"[DEBUG] inner_content_data keys: {list(inner_content_data.keys())}")
            logger.debug(f">> has_message_structure: {has_message_structure}")

            if has_message_structure:
                # Nested message structure: extract from inner message
                inner_kind = inner_content_data.get("kind", "")
                inner_msg_id = inner_content_data.get("id", "")
                actual_content_to_process = inner_content_data.get("content", {})
                inner_created_at_str = inner_content_data.get("created_at")
                print("[DEBUG] Using nested structure:")
                print(f"[DEBUG]   inner_kind: '{inner_kind}'")
                print(f"[DEBUG]   inner_msg_id: '{inner_msg_id}'")
                print(
                    f"[DEBUG]   actual_content_to_process: {actual_content_to_process}"
                )
                print(f"[DEBUG]   inner_created_at_str: {inner_created_at_str}")
                logger.debug(
                    f">> Using nested structure - inner_kind: {inner_kind}, inner_msg_id: {inner_msg_id}"
                )
                logger.debug(
                    f">> actual_content_to_process keys: {list(actual_content_to_process.keys())}"
                )
            else:
                # Direct content structure: the content data is directly in inner_content_data
                inner_kind = raw_payload.get("kind", "")  # Get kind from outer payload
                inner_msg_id = raw_payload.get("id", "")  # Get id from outer payload
                actual_content_to_process = (
                    inner_content_data  # Use inner_content_data directly
                )
                inner_created_at_str = raw_payload.get(
                    "created_at"
                )  # Get created_at from outer payload
                print("[DEBUG] Using direct structure:")
                print(f"[DEBUG]   inner_kind: '{inner_kind}' (from outer payload)")
                print(f"[DEBUG]   inner_msg_id: '{inner_msg_id}' (from outer payload)")
                print(
                    f"[DEBUG]   actual_content_to_process: {actual_content_to_process}"
                )
                print(f"[DEBUG]   inner_created_at_str: {inner_created_at_str}")
                logger.debug(
                    f">> Using direct structure - inner_kind: {inner_kind}, inner_msg_id: {inner_msg_id}"
                )
                logger.debug(
                    f">> actual_content_to_process keys: {list(actual_content_to_process.keys())}"
                )

        # Attempt to parse inner_created_at, fallback to now()
        try:
            inner_created_at = (
                datetime.fromisoformat(inner_created_at_str)
                if inner_created_at_str and isinstance(inner_created_at_str, str)
                else datetime.now(timezone.utc)
            )
        except ValueError:
            inner_created_at = datetime.now(timezone.utc)

        # These are from the outer envelope, might be useful for routing/meta
        return_stream = raw_payload.get("return_stream")
        user_id = raw_payload.get("user_id")
        orgs = raw_payload.get("orgs")  # from outer
        handle = raw_payload.get("handle")  # from outer
        adapter = raw_payload.get("adapter")  # from outer
        api_key = raw_payload.get("api_key")  # from outer

        print("[DEBUG] Extracted outer envelope data:")
        print(f"[DEBUG]   return_stream: {return_stream}")
        print(f"[DEBUG]   user_id: {user_id}")
        print(f"[DEBUG]   orgs: {orgs}")
        print(f"[DEBUG]   got orgs as orgs: {orgs}")
        print(f"[DEBUG]   handle: {handle}")
        print(f"[DEBUG]   adapter: {adapter}")
        print(f"[DEBUG]   api_key length: {len(api_key) if api_key else 0}")

        logger.debug(f">> Extracted API key length: {len(api_key) if api_key else 0}")

        # --- Health Check Logic ---
        # Use inner_kind for health check
        print(f"[DEBUG] Checking if message is HealthCheck. inner_kind: '{inner_kind}'")
        print(
            f"[DEBUG] Message kind comparison: '{inner_kind}' == 'HealthCheckRequest' -> {inner_kind == 'HealthCheckRequest'}"
        )

        if inner_kind == "HealthCheckRequest":
            print("[DEBUG] *** HEALTH CHECK REQUEST MESSAGE DETECTED ***")
            print(f"[DEBUG] Message ID: {message_id}")
            print(f"[DEBUG] Inner message ID: {inner_msg_id}")
            print(f"[DEBUG] Return stream: {return_stream}")
            print(f"[DEBUG] User ID: {user_id}")
            print(f"[DEBUG] Content: {actual_content_to_process}")

            logger.info(
                f"Received HealthCheckRequest message {message_id} (inner_id: {inner_msg_id})"
            )

            # Forward to health stream for health worker subprocess to process
            if REDIS_HEALTH_STREAM:
                print(
                    f"[DEBUG] Forwarding health check to health stream: {REDIS_HEALTH_STREAM}"
                )
                try:
                    # Forward the entire message data to the health stream
                    health_message_data = {
                        "data": json.dumps(
                            {
                                "kind": inner_kind,
                                "id": inner_msg_id,
                                "content": actual_content_to_process,
                                "created_at": inner_created_at.isoformat(),
                                "return_stream": return_stream,
                                "user_id": user_id,
                                "orgs": orgs,
                                "handle": handle,
                                "adapter": adapter,
                                "api_key": api_key,
                                "original_message_id": message_id,  # Include original message ID for tracking
                            }
                        )
                    }

                    print(
                        f"[DEBUG] Health message data to forward: {json.dumps(health_message_data, indent=2)}"
                    )

                    r.xadd(REDIS_HEALTH_STREAM, health_message_data)  # type: ignore[arg-type]
                    print(
                        f"[DEBUG] Health check forwarded successfully to {REDIS_HEALTH_STREAM}"
                    )
                    logger.info(
                        f"Forwarded HealthCheckRequest {message_id} to health stream {REDIS_HEALTH_STREAM}"
                    )

                except Exception as e:
                    print(
                        f"[DEBUG] ERROR forwarding health check to health stream: {e}"
                    )
                    logger.error(f"Error forwarding health check to health stream: {e}")
                    # Fall back to sending error response directly
                    _send_error_response(
                        message_id,
                        f"Failed to forward health check: {e}",
                        traceback.format_exc(),
                        return_stream,
                        user_id,
                    )
            else:
                print(
                    "[DEBUG] No health stream configured, cannot forward health check"
                )
                logger.warning(
                    "No health stream configured for health check forwarding"
                )
                # Send error response since we can't process health checks
                _send_error_response(
                    message_id,
                    "Health check stream not configured",
                    "REDIS_HEALTH_STREAM environment variable not set",
                    return_stream,
                    user_id,
                )

            # Acknowledge the original message since we've forwarded it
            print(f"[DEBUG] Acknowledging original health check message {message_id}")
            try:
                assert isinstance(REDIS_STREAM, str)
                assert isinstance(REDIS_CONSUMER_GROUP, str)
                r.xack(REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id)
                print(
                    f"[DEBUG] Health check message {message_id} acknowledged successfully"
                )
                logger.info(f"Acknowledged HealthCheckRequest message {message_id}")
            except Exception as e:
                print(f"[DEBUG] ERROR acknowledging health check message: {e}")
                logger.error(f"Error acknowledging health check message: {e}")

            print("[DEBUG] *** HEALTH CHECK FORWARDING COMPLETE ***")
            return  # Exit early for health checks
        # --- End Health Check Logic ---

        # For non-HealthCheck messages, the content to be processed by user function
        # is `actual_content_to_process`
        # The `kind` and `id` for the Message object should be from `inner_content_data`

        # Parse actual_content_to_process if it's a string (e.g., double-encoded JSON)
        # This might be redundant if actual_content_to_process is already a dict from inner_content_data.get("content")
        if isinstance(actual_content_to_process, str):
            try:
                content_for_validation = json.loads(actual_content_to_process)
                logger.debug(
                    f">> Parsed JSON string content_for_validation keys: {list(content_for_validation.keys()) if isinstance(content_for_validation, dict) else 'N/A'}"
                )
            except json.JSONDecodeError:
                content_for_validation = (
                    actual_content_to_process  # Keep as string if not valid JSON
                )
                logger.debug(
                    f">> Failed to parse JSON, keeping as string: {type(actual_content_to_process)}"
                )
        else:
            content_for_validation = actual_content_to_process
            logger.debug(
                f">> Using actual_content_to_process directly: {type(content_for_validation)}"
            )
            logger.debug(
                f">> content_for_validation keys: {list(content_for_validation.keys())}"
            )

        # --- Construct Input Object using Imported Types ---
        input_obj: Any = None
        input_type_class = None

        try:
            # Try to get the actual model classes (they should be available via import)
            # Need to handle potential NameErrors if imports failed silently
            # Note: This assumes models are defined in the imported module scope
            # Or imported by the imported module.
            from nebulous.processors.models import (
                Message,  # Import needed message class
            )

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
                        logger.debug(
                            f">> Attempting to validate content with {content_model_class.__name__}"
                        )
                        logger.debug(
                            f">> Content being validated: {json.dumps(content_for_validation, indent=2) if isinstance(content_for_validation, dict) else str(content_for_validation)}"
                        )
                        content_model = content_model_class.model_validate(
                            content_for_validation
                        )
                        logger.debug(">> Successfully validated content model")
                        # print(f"Validated content model: {content_model}")
                        input_obj = message_class(
                            kind=inner_kind,
                            id=inner_msg_id,
                            content=content_model,
                            created_at=int(inner_created_at.timestamp()),
                            return_stream=return_stream,
                            user_id=user_id,
                            orgs=orgs,
                            handle=handle,
                            adapter=adapter,
                            api_key=api_key,
                        )
                        logger.debug(
                            ">> Successfully created Message object with validated content"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error validating/creating content model '{content_type_name}': {e}. Falling back."
                        )
                        logger.debug(
                            f">> Content validation failed for: {json.dumps(content_for_validation, indent=2) if isinstance(content_for_validation, dict) else str(content_for_validation)}"
                        )
                        # Fallback to raw content in Message
                        input_obj = message_class(
                            kind=inner_kind,
                            id=inner_msg_id,
                            content=cast(Any, content_for_validation),
                            created_at=int(inner_created_at.timestamp()),
                            return_stream=return_stream,
                            user_id=user_id,
                            orgs=orgs,
                            handle=handle,
                            adapter=adapter,
                            api_key=api_key,
                        )
                        logger.debug(
                            ">> Created Message object with raw content fallback"
                        )
                else:
                    # No content type name or class found, use raw content
                    logger.debug(">> No content model class found, using raw content")
                    input_obj = message_class(
                        kind=inner_kind,
                        id=inner_msg_id,
                        content=cast(Any, content_for_validation),
                        created_at=int(inner_created_at.timestamp()),
                        return_stream=return_stream,
                        user_id=user_id,
                        orgs=orgs,
                        handle=handle,
                        adapter=adapter,
                        api_key=api_key,
                    )
                    logger.debug(
                        ">> Created Message object with raw content (no content model class)"
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
                        input_obj = content_for_validation
                    else:
                        logger.debug(f"Found input model class: {input_type_class}")
                        input_obj = input_type_class.model_validate(
                            content_for_validation
                        )
                        # Safe logging that avoids __repr__ issues with BaseModel objects
                        try:
                            if hasattr(input_obj, "model_dump"):
                                logger.debug(
                                    f"Validated input model (BaseModel): {input_obj.model_dump()}"
                                )
                            else:
                                logger.debug(f"Validated input model: {input_obj}")
                        except Exception as log_e:
                            logger.debug(
                                f"Validated input model: <object of type {type(input_obj).__name__}> (repr failed: {log_e})"
                            )
                except AttributeError:
                    logger.warning(
                        f"Warning: Input type class '{param_type_name}' not found in imported module."
                    )
                    input_obj = content_for_validation
                except Exception as e:
                    logger.error(
                        f"Error resolving/validating input type '{param_type_name}': {e}. Passing raw content."
                    )
                    input_obj = content_for_validation

        except NameError as e:
            logger.error(
                f"Error: Required class (e.g., Message or parameter type) not found. Import failed? {e}"
            )
            # Can't proceed without types, re-raise or handle error response
            raise RuntimeError(f"Required class not found: {e}") from e
        except Exception as e:
            logger.error(f"Error constructing input object: {e}")
            raise  # Re-raise unexpected errors during input construction

        # Safe logging that avoids __repr__ issues with BaseModel objects
        try:
            if hasattr(input_obj, "model_dump"):
                try:
                    # Use model_dump_json for safer serialization that handles nested objects
                    logger.debug(
                        f"Input object (BaseModel): {input_obj.model_dump_json()}"
                    )
                except Exception as dump_e:
                    logger.debug(
                        f"Input object: <BaseModel object of type {type(input_obj).__name__}> (model_dump failed: {dump_e})"
                    )
            else:
                logger.debug(f"Input object: {input_obj}")
        except Exception as log_e:
            logger.debug(
                f"Input object: <object of type {type(input_obj).__name__}> (repr failed: {log_e})"
            )

        # Execute the function
        logger.info("Executing function...")

        # Add warning about potential print statement issues in user code
        logger.debug(
            ">> About to execute user function - note: print statements in user code may fail if the Message object has validation issues"
        )

        if is_async_function:
            result = asyncio.run(target_function(input_obj))
        else:
            result = target_function(input_obj)

        # Debug: Check what type of result we got
        logger.info(f"[Consumer] Function result type: {type(result)}")
        logger.info(f"[Consumer] Result value: {result}")
        logger.info(
            f"[Consumer] Is GeneratorType: {isinstance(result, types.GeneratorType)}"
        )
        if hasattr(types, "AsyncGeneratorType"):
            logger.info(
                f"[Consumer] Is AsyncGeneratorType: {isinstance(result, types.AsyncGeneratorType)}"
            )

        # --- Streaming support for generator results ---
        is_generator = isinstance(result, types.GeneratorType)
        is_async_generator = hasattr(types, "AsyncGeneratorType") and isinstance(
            result, types.AsyncGeneratorType
        )

        if is_generator or is_async_generator:
            logger.info(
                f"[Consumer] Detected {'async ' if is_async_generator else ''}generator result – streaming chunks as they arrive."
            )

            chunk_index = 0
            try:
                # Convert async generator to list first if needed
                if is_async_generator:

                    async def collect_async_chunks():
                        chunks = []
                        async for chunk in result:  # type: ignore[misc]
                            chunks.append(chunk)
                        return chunks

                    chunks = asyncio.run(collect_async_chunks())
                else:
                    chunks = list(result)  # type: ignore[misc]

                # Process all chunks
                for chunk in chunks:  # type: ignore[misc]
                    try:
                        # Serialize chunk similar to single result handling
                        if hasattr(chunk, "model_dump"):  # type: ignore[misc]
                            chunk_content = chunk.model_dump(mode="json")  # type: ignore[misc]
                        else:
                            # Ensure JSON serializable
                            try:
                                json.dumps(chunk)
                                chunk_content = chunk
                            except TypeError:
                                logger.warning(
                                    "[Consumer] Skipping non-serializable chunk from generator."
                                )
                                continue  # Skip this chunk

                        if return_stream:
                            assert isinstance(return_stream, str)
                            r.xadd(
                                return_stream,
                                {
                                    "data": json.dumps(
                                        {
                                            "kind": "StreamChunkMessage",
                                            "id": f"{message_id}:{chunk_index}",
                                            "content": chunk_content,
                                            "status": "stream",
                                            "created_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "user_id": user_id,
                                        }
                                    )
                                },
                            )
                            chunk_index += 1
                    except Exception as chunk_err:
                        logger.error(
                            f"[Consumer] Error while processing generator chunk: {chunk_err}"
                        )

            finally:
                # Close generator if needed
                try:
                    result.close()  # type: ignore[attr-defined]
                except Exception:
                    pass

            # After streaming all chunks, send final success envelope (empty content)
            if return_stream:
                try:
                    r.xadd(
                        return_stream,
                        {
                            "data": json.dumps(
                                {
                                    "kind": "StreamResponseMessage",
                                    "id": message_id,
                                    "content": None,
                                    "status": "success",
                                    "created_at": datetime.now(
                                        timezone.utc
                                    ).isoformat(),
                                    "user_id": user_id,
                                }
                            )
                        },
                    )
                except Exception as final_err:
                    logger.error(
                        f"[Consumer] Failed to send final stream completion message: {final_err}"
                    )

            # Acknowledge original message and exit early – already streamed
            assert isinstance(REDIS_STREAM, str)
            assert isinstance(REDIS_CONSUMER_GROUP, str)
            r.xack(REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id)
            return

        # --- Non-generator result handling (existing logic) ---

        result_content = None  # Default to None
        if result is not None:
            try:
                if hasattr(result, "model_dump"):  # type: ignore[misc]
                    result_content = result.model_dump(mode="json")  # type: ignore[misc]
                else:
                    try:
                        json.dumps(result)
                        result_content = result
                    except TypeError as e:
                        logger.warning(
                            f"[Consumer] Warning: Result is not JSON serializable: {e}. Discarding result."
                        )
                        result_content = None
            except Exception as e:
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
async def _async_send_error_response(
    message_id: str,
    error_msg: str,
    tb: str,
    return_stream: Optional[str],
    user_id: Optional[str],
):
    """Sends a standardized error response to Redis asynchronously."""
    global r, REDIS_STREAM

    error_response = {
        "kind": "StreamResponseMessage",
        "id": message_id,
        "content": {
            "error": error_msg,
            "traceback": tb,
        },
        "status": "error",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
    }

    error_destination = f"{REDIS_STREAM}.errors"
    if return_stream:
        error_destination = return_stream

    try:
        assert isinstance(error_destination, str)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, r.xadd, error_destination, {"data": json.dumps(error_response)}
        )  # type: ignore
        logger.info(
            f"Sent error response for message {message_id} to {error_destination}"
        )
    except Exception as e_redis:
        logger.critical(
            f"CRITICAL: Failed to send error response for {message_id} to Redis: {e_redis}"
        )
        logger.exception("Redis Error Response Send Error Traceback:")


# --- Helper to Send Error Response ---
def _send_error_response(
    message_id: str,
    error_msg: str,
    tb: str,
    return_stream: Optional[str],
    user_id: Optional[str],
):
    """Sends a standardized error response to Redis."""
    global r, REDIS_STREAM

    error_response = {
        "kind": "StreamResponseMessage",
        "id": message_id,
        "content": {
            "error": error_msg,
            "traceback": tb,
        },
        "status": "error",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
    }

    error_destination = f"{REDIS_STREAM}.errors"
    if return_stream:
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


async def async_process_message(message_id: str, message_data: Dict[str, str]) -> None:
    """Processes a single message asynchronously, meant to be run concurrently."""
    # This function mirrors the logic of the inline processing path in `process_message`
    # but uses async/await and executes blocking I/O in an executor.
    global target_function, imported_module, local_namespace
    global r, REDIS_STREAM, REDIS_CONSUMER_GROUP

    if target_function is None or imported_module is None:
        logger.error(
            f"Error processing message {message_id}: User code is not loaded. Skipping."
        )
        await _async_send_error_response(
            message_id,
            "User code is not loaded (likely due to a failed reload)",
            traceback.format_exc(),
            None,
            None,
        )
        try:
            assert isinstance(REDIS_STREAM, str)
            assert isinstance(REDIS_CONSUMER_GROUP, str)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, r.xack, REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id
            )  # type: ignore
            logger.warning(
                f"Acknowledged message {message_id} due to code load failure."
            )
        except Exception as e_ack:
            logger.critical(
                f"CRITICAL: Failed to acknowledge message {message_id} after code load failure: {e_ack}"
            )
        return

    return_stream = None
    user_id = None
    try:
        # Most of the parsing and object creation logic is synchronous and can be reused.
        # The following is a recreation of the logic inside the original process_message try/except block.
        payload_str = message_data.get("data")
        if not payload_str:
            raise ValueError(f"Missing 'data' field: {message_data}")
        raw_payload = json.loads(payload_str)
        if not isinstance(raw_payload, dict):
            raise TypeError(f"Expected dict payload, got {type(raw_payload)}")

        inner_content_data = raw_payload.get("content", {})
        has_message_structure = (
            isinstance(inner_content_data, dict)
            and "kind" in inner_content_data
            and "id" in inner_content_data
            and "content" in inner_content_data
        )

        if has_message_structure:
            inner_kind = inner_content_data.get("kind", "")
            inner_msg_id = inner_content_data.get("id", "")
            actual_content_to_process = inner_content_data.get("content", {})
            inner_created_at_str = inner_content_data.get("created_at")
        else:
            inner_kind = raw_payload.get("kind", "")
            inner_msg_id = raw_payload.get("id", "")
            actual_content_to_process = inner_content_data
            inner_created_at_str = raw_payload.get("created_at")

        try:
            inner_created_at = (
                datetime.fromisoformat(inner_created_at_str)
                if inner_created_at_str and isinstance(inner_created_at_str, str)
                else datetime.now(timezone.utc)
            )
        except ValueError:
            inner_created_at = datetime.now(timezone.utc)

        return_stream = raw_payload.get("return_stream")
        user_id = raw_payload.get("user_id")
        orgs = raw_payload.get("orgs")
        handle = raw_payload.get("handle")
        adapter = raw_payload.get("adapter")
        api_key = raw_payload.get("api_key")

        if inner_kind == "HealthCheckRequest":
            # Health checks are forwarded and not processed by the async function directly.
            # This logic is now async to avoid blocking the event loop.
            if REDIS_HEALTH_STREAM:
                health_message_data = {
                    "data": json.dumps(
                        {
                            "kind": inner_kind,
                            "id": inner_msg_id,
                            "content": actual_content_to_process,
                            "created_at": inner_created_at.isoformat(),
                            "return_stream": return_stream,
                            "user_id": user_id,
                            "orgs": orgs,
                            "handle": handle,
                            "adapter": adapter,
                            "api_key": api_key,
                            "original_message_id": message_id,
                        }
                    )
                }
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, r.xadd, REDIS_HEALTH_STREAM, health_message_data
                )  # type: ignore
                logger.info(
                    f"Forwarded HealthCheckRequest {message_id} to health stream {REDIS_HEALTH_STREAM}"
                )
            else:
                logger.warning(
                    "No health stream configured for health check forwarding"
                )
                await _async_send_error_response(
                    message_id,
                    "Health check stream not configured",
                    "REDIS_HEALTH_STREAM environment variable not set",
                    return_stream,
                    user_id,
                )
            # Acknowledge and exit
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, r.xack, REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id
            )  # type: ignore
            logger.info(f"Acknowledged HealthCheckRequest message {message_id}")
            return

        content_for_validation = actual_content_to_process
        if isinstance(actual_content_to_process, str):
            try:
                content_for_validation = json.loads(actual_content_to_process)
            except json.JSONDecodeError:
                pass

        # Input object construction logic (same as in process_message)
        from nebulous.processors.models import Message

        input_obj: Any = None
        if is_stream_message:
            content_model_class = None
            if content_type_name:
                content_model_class = getattr(
                    imported_module, content_type_name, None
                ) or local_namespace.get(content_type_name)
            if content_model_class:
                content_model = content_model_class.model_validate(
                    content_for_validation
                )
                input_obj = Message(
                    kind=inner_kind,
                    id=inner_msg_id,
                    content=content_model,
                    created_at=int(inner_created_at.timestamp()),
                    return_stream=return_stream,
                    user_id=user_id,
                    orgs=orgs,
                    handle=handle,
                    adapter=adapter,
                    api_key=api_key,
                )
            else:
                input_obj = Message(
                    kind=inner_kind,
                    id=inner_msg_id,
                    content=cast(Any, content_for_validation),
                    created_at=int(inner_created_at.timestamp()),
                    return_stream=return_stream,
                    user_id=user_id,
                    orgs=orgs,
                    handle=handle,
                    adapter=adapter,
                    api_key=api_key,
                )
        else:
            param_type_name = param_type_str
            input_type_class = (
                getattr(imported_module, param_type_name, None)
                if param_type_name
                else None
            ) or (local_namespace.get(param_type_name) if param_type_name else None)
            if input_type_class:
                input_obj = input_type_class.model_validate(content_for_validation)
            else:
                input_obj = content_for_validation

        # Execute the function asynchronously
        logger.info(f"Executing async function for message {message_id}...")
        result = await target_function(input_obj)

        result_content = None
        if result is not None:
            if hasattr(result, "model_dump"):  # type: ignore[misc]
                result_content = result.model_dump(mode="json")  # type: ignore[misc]
            else:
                try:
                    json.dumps(result)
                    result_content = result
                except TypeError:
                    logger.warning("Result is not JSON serializable, discarding.")
                    result_content = None

        response = {
            "kind": "StreamResponseMessage",
            "id": message_id,
            "content": result_content,
            "status": "success",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
        }

        loop = asyncio.get_running_loop()
        if return_stream:
            assert isinstance(return_stream, str)
            await loop.run_in_executor(
                None, r.xadd, return_stream, {"data": json.dumps(response)}
            )  # type: ignore
            logger.info(
                f"Processed message {message_id}, result sent to {return_stream}"
            )

        assert isinstance(REDIS_STREAM, str)
        assert isinstance(REDIS_CONSUMER_GROUP, str)
        await loop.run_in_executor(
            None, r.xack, REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id
        )  # type: ignore

    except RetriableError as e:
        logger.warning(f"Retriable error processing message {message_id}: {e}")
        await _async_send_error_response(
            message_id, str(e), traceback.format_exc(), return_stream, user_id
        )
        logger.info(f"Message {message_id} will be retried later.")

    except Exception as e:
        logger.error(f"Error processing message {message_id}: {e}")
        logger.exception("Message Processing Error Traceback:")
        await _async_send_error_response(
            message_id, str(e), traceback.format_exc(), return_stream, user_id
        )

        try:
            assert isinstance(REDIS_STREAM, str)
            assert isinstance(REDIS_CONSUMER_GROUP, str)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, r.xack, REDIS_STREAM, REDIS_CONSUMER_GROUP, message_id
            )  # type: ignore
            logger.info(f"Acknowledged failed message {message_id}")
        except Exception as e_ack:
            logger.critical(
                f"CRITICAL: Failed to acknowledge failed message {message_id}: {e_ack}"
            )


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

# Start the health check consumer subprocess
if REDIS_HEALTH_STREAM and REDIS_HEALTH_CONSUMER_GROUP:
    print("[DEBUG] === HEALTH SUBPROCESS INITIALIZATION ===")
    print(f"[DEBUG] REDIS_HEALTH_STREAM is set: {REDIS_HEALTH_STREAM}")
    print(f"[DEBUG] REDIS_HEALTH_CONSUMER_GROUP is set: {REDIS_HEALTH_CONSUMER_GROUP}")

    health_subprocess = start_health_check_subprocess()
    if health_subprocess:
        print(
            "[DEBUG] Health subprocess started successfully, starting monitor thread..."
        )
        # Start monitoring thread for subprocess output
        monitor_thread = threading.Thread(
            target=monitor_health_subprocess, args=(health_subprocess,), daemon=True
        )
        monitor_thread.start()
        print("[DEBUG] Monitor thread started for health subprocess")
        logger.info(
            f"[Consumer] Health check subprocess for {REDIS_HEALTH_STREAM} started and monitoring thread started."
        )
    else:
        print("[DEBUG] Health subprocess failed to start")
        logger.error("[Consumer] Failed to start health check subprocess.")
else:
    print("[DEBUG] === HEALTH SUBPROCESS NOT CONFIGURED ===")
    print(f"[DEBUG] REDIS_HEALTH_STREAM: {REDIS_HEALTH_STREAM}")
    print(f"[DEBUG] REDIS_HEALTH_CONSUMER_GROUP: {REDIS_HEALTH_CONSUMER_GROUP}")
    logger.warning(
        "[Consumer] Health check stream not configured. Health consumer subprocess not started."
    )

try:
    while True:
        logger.debug(
            f"[{datetime.now(timezone.utc).isoformat()}] --- Top of main loop ---"
        )  # Added log
        print("[DEBUG] === MAIN LOOP ITERATION START ===")

        # --- Check Health Subprocess Status ---
        if health_subprocess:
            print("[DEBUG] Checking health subprocess status...")
            health_status = check_health_subprocess()
            print(f"[DEBUG] Health subprocess status check result: {health_status}")
        else:
            print("[DEBUG] No health subprocess to check")

        # --- Check for Code Updates ---
        if not disable_hot_reload:
            logger.debug(
                f"[{datetime.now(timezone.utc).isoformat()}] Checking for code updates..."
            )  # Added log
            if entrypoint_abs_path:  # Should always be set after init
                try:
                    # Retry logic for getmtime with progressive backoff
                    current_mtime = 0
                    max_retries = 5
                    initial_delay = 0.1  # Start with 100ms
                    for attempt in range(max_retries):
                        try:
                            current_mtime = os.path.getmtime(entrypoint_abs_path)
                            break  # Success
                        except FileNotFoundError:
                            if attempt < max_retries - 1:
                                delay = initial_delay * (2**attempt)
                                logger.warning(
                                    f"[Consumer] getmtime check failed for '{entrypoint_abs_path}' (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.2f}s..."
                                )
                                time.sleep(delay)
                            else:
                                logger.error(
                                    f"[Consumer] getmtime check failed for '{entrypoint_abs_path}' after {max_retries} attempts. Skipping reload."
                                )
                                raise  # Re-raise the final FileNotFoundError

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
                    # Add directory listing for debugging
                    dir_path = os.path.dirname(entrypoint_abs_path)
                    if dir_path and os.path.isdir(dir_path):
                        logger.error(f"Listing contents of directory '{dir_path}':")
                        try:
                            for item in os.listdir(dir_path):
                                logger.error(f"  - {item}")
                        except OSError as e:
                            logger.error(f"    Error listing directory: {e}")
                    elif dir_path:
                        logger.error(f"Directory '{dir_path}' does not exist.")
                    else:
                        logger.error(
                            f"Could not determine directory from path '{entrypoint_abs_path}'"
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
        print("[DEBUG] About to call xreadgroup...")
        print(f"[DEBUG] Stream: {REDIS_STREAM}")
        print(f"[DEBUG] Consumer group: {REDIS_CONSUMER_GROUP}")
        print(f"[DEBUG] Consumer name: {consumer_name}")

        messages = r.xreadgroup(
            REDIS_CONSUMER_GROUP,
            consumer_name,
            streams_arg,  # type: ignore[arg-type] # Suppress linter warning
            count=10,
            block=5000,  # Use milliseconds for block
        )

        print(f"[DEBUG] xreadgroup returned: {messages}")
        print(f"[DEBUG] Messages type: {type(messages)}")

        if not messages:
            logger.trace(
                f"[{datetime.now(timezone.utc).isoformat()}] xreadgroup timed out (no new messages)."
            )  # Added log
            print("[DEBUG] No messages received (timeout or empty)")
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

        print(f"[DEBUG] Processing {num_msgs} message(s) from stream {stream_name_str}")

        # Log reception and count before processing
        logger.info(
            f"[{datetime.now(timezone.utc).isoformat()}] xreadgroup returned {num_msgs} message(s). Processing..."
        )  # Moved and combined log

        # Process the received messages
        if is_async_function:
            # Concurrent processing for async functions
            print(f"[DEBUG] Processing {num_msgs} messages concurrently...")

            async def process_batch():
                tasks = []
                for message_id_str, message_data_str_dict in stream_messages:
                    print(f"[DEBUG] Creating task for message {message_id_str}")
                    task = asyncio.create_task(
                        async_process_message(message_id_str, message_data_str_dict)
                    )
                    tasks.append(task)
                await asyncio.gather(*tasks)
                print("[DEBUG] Finished processing batch.")

            asyncio.run(process_batch())
        else:
            # Sequential processing for regular functions
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
                print(f"[DEBUG] === PROCESSING MESSAGE {message_id_str} ===")
                print(
                    f"[DEBUG] Message data keys: {list(message_data_str_dict.keys())}"
                )

                process_message(message_id_str, message_data_str_dict)

                print(f"[DEBUG] === FINISHED PROCESSING MESSAGE {message_id_str} ===")

except ConnectionError as e:
    logger.error(f"Redis connection error: {e}. Reconnecting in 5s...")
    time.sleep(5)
    # Attempt to reconnect explicitly
    try:
        logger.info("Attempting Redis reconnection...")
        # Close existing potentially broken connection?
        if r:  # Check if r was initialized
            try:
                r.close()
            except Exception:
                pass  # Ignore errors during close
        r = connect_redis(REDIS_URL)  # connect_redis will sys.exit on failure
        logger.info("Reconnected to Redis.")
    except Exception as recon_e:  # Should not be reached if connect_redis exits
        logger.error(f"Failed to reconnect to Redis: {recon_e}")

except ResponseError as e:
    logger.error(f"Redis command error: {e}")
    # Should we exit or retry?
    if r and "NOGROUP" in str(e):  # Check if r is initialized
        logger.critical("Consumer group seems to have disappeared. Exiting.")
        sys.exit(1)
    time.sleep(1)

except Exception as e:
    logger.error(f"Unexpected error in main loop: {e}")
    logger.exception("Main Loop Error Traceback:")
    time.sleep(1)

finally:
    logger.info("Consumer loop exited.")
    # Cleanup health subprocess
    if health_subprocess and health_subprocess.poll() is None:
        logger.info("[Consumer] Terminating health check subprocess...")
        health_subprocess.terminate()
        try:
            health_subprocess.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(
                "[Consumer] Health subprocess did not terminate gracefully, killing it."
            )
            health_subprocess.kill()
        logger.info("[Consumer] Health subprocess cleanup complete.")
