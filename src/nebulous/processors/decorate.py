import ast  # For parsing notebook code
import inspect
import os
import re
import textwrap
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)
from urllib.parse import urlparse

import dill
import requests
from botocore.exceptions import ClientError
from pydantic import BaseModel

from nebulous.auth import get_user_profile
from nebulous.config import GlobalConfig
from nebulous.containers.models import (
    V1AuthzConfig,
    V1ContainerHealthCheck,
    V1ContainerRequest,
    V1ContainerResources,
    V1EnvVar,
    V1Meter,
    V1PortRequest,
    V1SSHKey,
    V1VolumeDriver,
    V1VolumePath,
)
from nebulous.data import Bucket
from nebulous.logging import logger
from nebulous.meta import V1ResourceMetaRequest
from nebulous.processors.models import (
    Message,
    V1Scale,
)
from nebulous.processors.processor import Processor

from .default import DEFAULT_MAX_REPLICAS, DEFAULT_MIN_REPLICAS, DEFAULT_SCALE

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R", bound=BaseModel)

# Attribute name for explicitly stored source
_NEBU_EXPLICIT_SOURCE_ATTR = "_nebu_explicit_source"
# Environment variable to prevent decorator recursion inside consumer
_NEBU_INSIDE_CONSUMER_ENV_VAR = "_NEBU_INSIDE_CONSUMER_EXEC"

# Define target directory in container
CONTAINER_CODE_DIR = "/app"
# Define S3 prefix for code storage (under the base URI from token endpoint)
S3_CODE_PREFIX = "nebulous-code"
NEBU_API_BASE_URL = GlobalConfig.get_server_url()

# --- Jupyter Helper Functions ---


def is_jupyter_notebook():
    """
    Determine if the current code is running inside a Jupyter notebook.
    Returns bool: True if running inside a Jupyter notebook, False otherwise.
    """
    try:
        # Use importlib to avoid runtime dependency if not needed
        import importlib.util

        if importlib.util.find_spec("IPython") is None:
            return False
        # Fix: Import get_ipython directly
        from IPython.core.getipython import get_ipython  # Now safe to import

        ip = get_ipython()  # Use the imported function
        if ip is None:  # type: ignore
            # logger.debug("is_jupyter_notebook: No IPython instance found.")
            return False
        class_name = str(ip.__class__)
        # logger.debug(f"is_jupyter_notebook: IPython class name: {class_name}")
        if "ZMQInteractiveShell" in class_name:
            # logger.debug("is_jupyter_notebook: Jupyter detected (ZMQInteractiveShell).")
            return True
        # logger.debug("is_jupyter_notebook: Not Jupyter (IPython instance found, but not ZMQInteractiveShell).")
        return False
    except Exception as e:
        logger.debug(
            f"is_jupyter_notebook: Exception occurred: {e}"
        )  # Keep as debug for less noise
        return False


def get_notebook_executed_code():
    """
    Returns all executed code from the current notebook session.
    Returns str or None: All executed code as a string, or None if not possible.
    """
    logger.debug("Attempting to get notebook execution history...")
    try:
        # Fix: Import get_ipython directly
        from IPython.core.getipython import get_ipython

        ip = get_ipython()  # Use the imported function
        if ip is None or not hasattr(ip, "history_manager"):
            logger.debug(
                "get_notebook_executed_code: No IPython instance or history_manager."
            )
            return None
        history_manager = ip.history_manager
        # Limiting history range for debugging? Maybe get_tail(N)? For now, get all.
        # history = history_manager.get_range(start=1) # type: ignore
        history = list(history_manager.get_range(start=1))  # type: ignore # Convert to list to get length
        logger.debug(
            f"get_notebook_executed_code: Retrieved {len(history)} history entries."
        )
        source_code = ""
        separator = "\n#<NEBU_CELL_SEP>#\n"
        for _, _, content in history:  # Use _ for unused session, lineno
            if isinstance(content, str) and content.strip():
                source_code += content + separator
        logger.debug(
            f"get_notebook_executed_code: Total history source length: {len(source_code)}"
        )
        return source_code
    except Exception as e:
        logger.error(f"get_notebook_executed_code: Error getting history: {e}")
        return None


def extract_definition_source_from_string(
    source_string: str, def_name: str, def_type: type = ast.FunctionDef
) -> Optional[str]:
    """
    Attempts to extract the source code of a function or class from a larger string
    (like notebook history). Finds the *last* complete definition.
    Uses AST parsing for robustness.
    def_type can be ast.FunctionDef or ast.ClassDef.
    """
    logger.debug(
        f"Extracting '{def_name}' ({def_type.__name__}) from history string (len: {len(source_string)})..."
    )
    if not source_string or not def_name:
        logger.debug("extract: Empty source string or def_name.")
        return None

    cells = source_string.split("#<NEBU_CELL_SEP>#")
    logger.debug(f"extract: Split history into {len(cells)} potential cells.")
    last_found_source = None

    for i, cell in enumerate(reversed(cells)):
        cell_num = len(cells) - 1 - i
        cell = cell.strip()
        if not cell:
            continue
        # logger.debug(f"extract: Analyzing cell #{cell_num}...") # Can be very verbose
        try:
            tree = ast.parse(cell)
            found_in_cell = False
            for node in ast.walk(tree):
                if (
                    isinstance(
                        node, def_type
                    )  # Check if it's the right type (FuncDef or ClassDef)
                    and getattr(node, "name", None) == def_name  # Safely check name
                ):
                    logger.debug(
                        f"extract: Found node for '{def_name}' in cell #{cell_num}."
                    )
                    try:
                        # Use ast.get_source_segment for accurate extraction (Python 3.8+)
                        func_source = ast.get_source_segment(cell, node)
                        if func_source:
                            logger.debug(
                                f"extract: Successfully extracted source using get_source_segment for '{def_name}'."
                            )
                            last_found_source = func_source
                            found_in_cell = True
                            break  # Stop searching this cell
                    except AttributeError:  # Fallback for Python < 3.8
                        logger.debug(
                            f"extract: get_source_segment failed (likely Py < 3.8), using fallback for '{def_name}'."
                        )
                        start_lineno = getattr(node, "lineno", 1) - 1
                        end_lineno = getattr(node, "end_lineno", start_lineno + 1)

                        if hasattr(node, "decorator_list") and node.decorator_list:  # type: ignore
                            # Ensure it's a node type that *can* have decorators
                            # FunctionDef and ClassDef have decorator_list
                            first_decorator_start_line = (
                                getattr(
                                    node.decorator_list[0],  # type: ignore
                                    "lineno",
                                    start_lineno + 1,
                                )
                                - 1
                            )  # type: ignore
                            start_lineno = min(start_lineno, first_decorator_start_line)

                        lines = cell.splitlines()
                        if 0 <= start_lineno < len(lines) and end_lineno <= len(lines):
                            extracted_lines = lines[start_lineno:end_lineno]
                            if extracted_lines and (
                                extracted_lines[0].strip().startswith("@")
                                or extracted_lines[0]
                                .strip()
                                .startswith(("def ", "class "))
                            ):
                                last_found_source = "\n".join(extracted_lines)
                                logger.debug(
                                    f"extract: Extracted source via fallback for '{def_name}'."
                                )
                                found_in_cell = True
                                break
                        else:
                            logger.warning(
                                f"extract: Line numbers out of bounds for {def_name} in cell (fallback)."
                            )

            if found_in_cell:
                logger.debug(
                    f"extract: Found and returning source for '{def_name}' from cell #{cell_num}."
                )
                return last_found_source  # Found last definition, return immediately

        except (SyntaxError, ValueError):
            # logger.debug(f"extract: Skipping cell #{cell_num} due to parse error: {e}") # Can be verbose
            continue
        except Exception as e:
            logger.warning(
                f"extract: AST processing error for {def_name} in cell #{cell_num}: {e}"
            )
            continue

    if not last_found_source:
        logger.debug(
            f"extract: Definition '{def_name}' of type {def_type.__name__} not found in history search."
        )
    return last_found_source


# --- End Jupyter Helper Functions ---


def include(obj: Any) -> Any:
    """
    Decorator to explicitly capture the source code of a function or class,
    intended for use in environments where inspect/dill might fail (e.g., Jupyter).
    NOTE: This source is currently added to environment variables. Consider if
    large included objects should also use S3.
    """
    try:
        # Still use dill for @include as it might capture things not in the main file dir
        source = dill.source.getsource(obj)
        dedented_source = textwrap.dedent(source)
        setattr(obj, _NEBU_EXPLICIT_SOURCE_ATTR, dedented_source)
        logger.debug(
            f"@include: Successfully captured source for: {getattr(obj, '__name__', str(obj))}"
        )
    except Exception as e:
        # Don't fail the definition, just warn
        logger.warning(
            f"@include could not capture source for {getattr(obj, '__name__', str(obj))}: {e}. Automatic source retrieval will be attempted later."
        )
    return obj


def get_model_source(
    model_class: Any, notebook_code: Optional[str] = None
) -> Optional[str]:
    """
    Get the source code of a model class.
    Checks explicit source, then notebook history (if provided), then dill.
    """
    model_name_str = getattr(model_class, "__name__", str(model_class))
    logger.debug(f"get_model_source: Getting source for: {model_name_str}")
    # 1. Check explicit source
    explicit_source = getattr(model_class, _NEBU_EXPLICIT_SOURCE_ATTR, None)
    if explicit_source:
        logger.debug(
            f"get_model_source: Using explicit source (@include) for: {model_name_str}"
        )
        return explicit_source

    # 2. Check notebook history
    if notebook_code and hasattr(model_class, "__name__"):
        logger.debug(
            f"get_model_source: Attempting notebook history extraction for: {model_class.__name__}"
        )
        extracted_source = extract_definition_source_from_string(
            notebook_code, model_class.__name__, ast.ClassDef
        )
        if extracted_source:
            logger.debug(
                f"get_model_source: Using notebook history source for: {model_class.__name__}"
            )
            return extracted_source
        else:
            logger.debug(
                f"get_model_source: Notebook history extraction failed for: {model_class.__name__}. Proceeding to dill."
            )

    # 3. Fallback to dill
    try:
        logger.debug(
            f"get_model_source: Attempting dill fallback for: {model_name_str}"
        )
        source = dill.source.getsource(model_class)
        logger.debug(f"get_model_source: Using dill source for: {model_name_str}")
        return textwrap.dedent(source)
    except (IOError, TypeError, OSError) as e:
        logger.debug(
            f"get_model_source: Failed dill fallback for: {model_name_str}: {e}"
        )
        return None


# Reintroduce get_type_source to handle generics
def get_type_source(
    type_obj: Any, notebook_code: Optional[str] = None
) -> Optional[Any]:
    """Get the source code for a type, including generic parameters."""
    type_obj_str = str(type_obj)
    logger.debug(f"get_type_source: Getting source for type: {type_obj_str}")
    origin = get_origin(type_obj)
    args = get_args(type_obj)

    if origin is not None:
        # Use updated get_model_source for origin
        logger.debug(
            f"get_type_source: Detected generic type. Origin: {origin}, Args: {args}"
        )
        origin_source = get_model_source(origin, notebook_code)
        args_sources = []

        # Recursively get sources for all type arguments
        for arg in args:
            logger.debug(
                f"get_type_source: Recursively getting source for generic arg #{arg}"
            )
            arg_source = get_type_source(arg, notebook_code)
            if arg_source:
                args_sources.append(arg_source)

        # Return tuple only if origin source or some arg sources were found
        if origin_source or args_sources:
            logger.debug(
                f"get_type_source: Returning tuple source for generic: {type_obj_str}"
            )
            return (origin_source, args_sources)

    # Fallback if not a class or recognizable generic alias
    # Try get_model_source as a last resort for unknown types
    fallback_source = get_model_source(type_obj, notebook_code)
    if fallback_source:
        logger.debug(
            f"get_type_source: Using fallback get_model_source for: {type_obj_str}"
        )
        return fallback_source

    logger.debug(f"get_type_source: Failed to get source for: {type_obj_str}")
    return None


def processor(
    image: str,
    setup_script: Optional[str] = None,
    scale: V1Scale = DEFAULT_SCALE,
    min_replicas: int = DEFAULT_MIN_REPLICAS,
    max_replicas: int = DEFAULT_MAX_REPLICAS,
    platform: Optional[str] = None,
    accelerators: Optional[List[str]] = None,
    namespace: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    env: Optional[List[V1EnvVar]] = None,
    volumes: Optional[List[V1VolumePath]] = None,
    resources: Optional[V1ContainerResources] = None,
    meters: Optional[List[V1Meter]] = None,
    authz: Optional[V1AuthzConfig] = None,
    python_cmd: str = "python",
    no_delete: bool = False,
    include: Optional[List[Any]] = None,
    init_func: Optional[Callable[[], None]] = None,
    queue: Optional[str] = None,
    timeout: Optional[str] = None,
    ssh_keys: Optional[List[V1SSHKey]] = None,
    ports: Optional[List[V1PortRequest]] = None,
    proxy_port: Optional[int] = None,
    health_check: Optional[V1ContainerHealthCheck] = None,
    execution_mode: str = "inline",
    stream: bool = False,
    config: Optional[GlobalConfig] = None,
    hot_reload: bool = True,
    debug: bool = False,
    name: Optional[str] = None,
    wait_for_healthy: bool = False,
):
    """
    Decorator that creates a processor from a function.

    This decorator transforms a function into a distributed processor that can be
    deployed and executed remotely. The function code is uploaded to S3 and a
    containerized processor is created.

    Args:
        image: Container image to use for the processor
        setup_script: Optional script to run during container setup
        scale: Scaling configuration for the processor
        min_replicas: Minimum number of processor replicas
        max_replicas: Maximum number of processor replicas
        platform: Target platform for container deployment
        accelerators: List of accelerators (e.g., GPUs) to request
        namespace: Kubernetes namespace for the processor
        labels: Additional labels to apply to the processor
        env: Environment variables to set in the container
        volumes: Volume mounts for the container
        resources: Resource requests and limits
        meters: Billing/metering configuration
        authz: Authorization configuration
        python_cmd: Python command to use (default: "python")
        no_delete: Whether to prevent deletion of the processor
        include: List of objects to include in the processor environment
        init_func: Optional initialization function to run once
        queue: Message queue configuration
        timeout: Processor timeout configuration
        ssh_keys: SSH keys for secure access
        ports: Port configurations for the container
        proxy_port: Proxy port configuration
        health_check: Container health check configuration
        execution_mode: Execution mode ("inline" or "subprocess")
        stream: Boolean flag to enable generator semantics
        config: Global configuration override
        hot_reload: Enable/disable hot code reloading (default: True)
        debug: Enable debug mode
        name: Override processor name (defaults to function name)
        wait_for_healthy: If True, wait for the processor to become healthy
                         before the decorator returns (default: False)

    Returns:
        Processor: A Processor instance wrapping the decorated function

    Example:
        ```python
        @processor(
            image="python:3.11",
            setup_script="pip install numpy",
            wait_for_healthy=True
        )
        def my_processor(data: Message[InputModel]) -> OutputModel:
            return OutputModel(result=data.content.value * 2)
        ```

    Note:
        When wait_for_healthy=True, the decorator will send health check messages
        to the processor and wait for successful responses before returning. This
        ensures the processor is ready to accept requests but may add startup time.
    """

    def decorator(
        func: Callable[[Any], Any],
    ) -> Processor:
        # --- Prevent Recursion Guard ---
        if os.environ.get(_NEBU_INSIDE_CONSUMER_ENV_VAR) == "1":
            logger.debug(
                f"Decorator Guard triggered for '{func.__name__}'. Returning original function."
            )
            return func  # type: ignore
        # --- End Guard ---

        is_async = inspect.iscoroutinefunction(func)
        logger.debug(f"Decorator: Function '{func.__name__}' is async: {is_async}")

        logger.debug(
            f"Decorator Init: @processor decorating function '{func.__name__}'"
        )
        all_env = env or []
        processor_name = name or func.__name__
        actual_function_name = func.__name__
        all_volumes = volumes or []  # Initialize volumes list

        # Use a local variable for config resolution
        effective_config = config

        # --- Get Decorated Function File Path and Directory ---
        logger.debug("Decorator: Getting source file path for decorated function...")
        func_file_path: Optional[str] = None
        func_dir: Optional[str] = None
        rel_func_path: Optional[str] = None  # Relative path within func_dir
        try:
            func_file_path = inspect.getfile(func)
            # Resolve symlinks to get the actual directory containing the file
            func_file_path = os.path.realpath(func_file_path)
            func_dir = os.path.dirname(func_file_path)
            # Calculate relative path based on the resolved directory
            rel_func_path = os.path.relpath(func_file_path, func_dir)
            logger.debug(f"Decorator: Found real file path: {func_file_path}")
            logger.debug(f"Decorator: Found function directory: {func_dir}")
            logger.debug(f"Decorator: Relative function path: {rel_func_path}")
        except (TypeError, OSError) as e:
            # TypeError can happen if func is not a module, class, method, function, traceback, frame, or code object
            raise ValueError(
                f"Could not get file path for function '{processor_name}'. Ensure it's defined in a file and is a standard function/method."
            ) from e
        except Exception as e:
            raise ValueError(
                f"Unexpected error getting file path for '{processor_name}': {e}"
            ) from e

        # --- Fetch S3 Token and Upload Code ---
        s3_destination_uri: Optional[str] = None
        if not func_dir or not rel_func_path:
            # This case should be caught by the exceptions above, but double-check
            raise ValueError(
                "Could not determine function directory or relative path for S3 upload."
            )
        # --- Get API Key ---
        logger.debug("Decorator: Loading Nebu configuration...")
        try:
            if not effective_config:
                effective_config = GlobalConfig.read()
            current_server = effective_config.get_current_server_config()
            if not current_server or not current_server.api_key:
                raise ValueError("Nebu server configuration or API key not found.")
            api_key = current_server.api_key
            logger.debug("Decorator: Nebu API key loaded successfully.")

            # # Add additional environment variables from current configuration
            # all_env.append(V1EnvVar(key="AGENTSEA_API_KEY", value=api_key))

            # # Get server URL and auth server URL from current configuration
            # server_url = (
            #     current_server.server
            #     if current_server
            #     else GlobalConfig.get_server_url()
            # )
            # auth_server_url = current_server.auth_server if current_server else None

            # if server_url:
            #     all_env.append(V1EnvVar(key="NEBULOUS_SERVER", value=server_url))

            # if auth_server_url:
            #     all_env.append(
            #         V1EnvVar(key="AGENTSEA_AUTH_SERVER", value=auth_server_url)
            #     )

            # orign_server = get_orign_server()
            # if orign_server:
            #     all_env.append(V1EnvVar(key="ORIGN_SERVER", value=orign_server))
            # else:
            #     logger.debug("Decorator: No Orign server found. Not setting...")

        except Exception as e:
            logger.error(f"Failed to load Nebu configuration or API key: {e}")
            raise RuntimeError(
                f"Failed to load Nebu configuration or API key: {e}"
            ) from e
        # --- End Get API Key ---

        # --- Determine Namespace ---
        effective_namespace = namespace  # Start with the provided namespace
        if effective_namespace is None:
            logger.debug("Decorator: Namespace not provided, fetching user profile...")
            try:
                user_profile = get_user_profile(api_key)
                if user_profile.handle:
                    effective_namespace = user_profile.handle
                    logger.debug(
                        f"Decorator: Using user handle '{effective_namespace}' as namespace."
                    )
                else:
                    raise ValueError("User profile does not contain a handle.")
            except Exception as e:
                logger.error(
                    f"Failed to get user profile or handle for default namespace: {e}"
                )
                raise RuntimeError(
                    f"Failed to get user profile or handle for default namespace: {e}"
                ) from e
        # --- End Determine Namespace ---

        # Use processor_name instead of name
        S3_TOKEN_ENDPOINT = f"{NEBU_API_BASE_URL}/v1/auth/temp-s3-tokens/{effective_namespace}/{processor_name}"
        logger.debug(f"Decorator: Fetching S3 token from: {S3_TOKEN_ENDPOINT}")
        try:
            headers = {"Authorization": f"Bearer {api_key}"}  # Add headers here

            # Try GET request instead of POST for this endpoint
            response = requests.get(S3_TOKEN_ENDPOINT, headers=headers, timeout=10)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            s3_token_data = response.json()

            logger.debug(f"Decorator: S3 token data: {s3_token_data}")

            aws_access_key_id = s3_token_data.get("access_key_id")
            aws_secret_access_key = s3_token_data.get("secret_access_key")
            aws_session_token = s3_token_data.get(
                "session_token"
            )  # May be None for non-STS keys
            s3_base_uri = s3_token_data.get("s3_base_uri")

            if not all([aws_access_key_id, aws_secret_access_key, s3_base_uri]):
                raise ValueError(
                    "Missing required fields (access_key_id, secret_access_key, s3_base_uri) in S3 token response."
                )

            # Construct unique S3 path: s3://<base_bucket>/<base_prefix>/<code_prefix>/<namespace>/<processor_name>/
            unique_suffix = f"{effective_namespace}/{processor_name}"
            parsed_base = urlparse(s3_base_uri)
            if not parsed_base.scheme == "s3" or not parsed_base.netloc:
                raise ValueError(f"Invalid s3_base_uri received: {s3_base_uri}")

            base_path = parsed_base.path.strip("/")
            s3_dest_components = [S3_CODE_PREFIX, unique_suffix]
            if base_path:
                # Handle potential multiple path segments in base_path
                path_components = [comp for comp in base_path.split("/") if comp]
                s3_dest_components = path_components + s3_dest_components

            # Filter out empty strings that might result from split
            s3_destination_key_components = [
                comp for comp in s3_dest_components if comp
            ]
            s3_destination_key = (
                "/".join(s3_destination_key_components) + "/"
            )  # Ensure trailing slash for prefix
            s3_destination_uri = f"s3://{parsed_base.netloc}/{s3_destination_key}"

            logger.debug(
                f"Decorator: Uploading code from '{func_dir}' to '{s3_destination_uri}'"
            )

            # Instantiate Bucket with temporary credentials
            s3_bucket = Bucket(
                verbose=True,  # Make verbosity configurable later if needed
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
            )

            # Use sync to upload directory contents recursively
            # Ensure source directory exists before syncing
            if not os.path.isdir(func_dir):
                raise ValueError(
                    f"Source path for upload is not a directory: {func_dir}"
                )

            s3_bucket.sync(
                source=func_dir,
                destination=s3_destination_uri,
                delete=True,
                dry_run=False,
                excludes=["__pycache__", "*.pyc"],  # Add excludes here
            )
            logger.debug("Decorator: S3 code upload completed.")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch S3 token from {S3_TOKEN_ENDPOINT}: {e}")
            raise RuntimeError(
                f"Failed to fetch S3 token from {S3_TOKEN_ENDPOINT}: {e}"
            ) from e
        except ClientError as e:
            logger.error(f"Failed to upload code to S3 {s3_destination_uri}: {e}")
            # Attempt to provide more context from the error if possible
            error_code = e.response.get("Error", {}).get("Code")
            error_msg = e.response.get("Error", {}).get("Message")
            logger.error(f"      S3 Error Code: {error_code}, Message: {error_msg}")
            raise RuntimeError(
                f"Failed to upload code to {s3_destination_uri}: {e}"
            ) from e
        except ValueError as e:  # Catch ValueErrors from validation
            logger.error(f"Configuration or response data error: {e}")
            raise RuntimeError(f"Configuration or response data error: {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected error during S3 token fetch or upload: {e}")
            # Consider logging traceback here for better debugging
            import traceback

            traceback.print_exc()  # Keep this explicit traceback for now in case logging isn't configured yet
            raise RuntimeError(f"Unexpected error during S3 setup: {e}") from e

        # --- Process Manually Included Objects (Keep for now, add source via env) ---
        included_sources: Dict[Any, Any] = {}
        notebook_code_for_include = None  # Get notebook code only if needed for include
        if include:
            # Determine if we are in Jupyter only if needed for include fallback
            # logger.debug("Decorator: Processing manually included objects...")
            is_jupyter_env = is_jupyter_notebook()
            if is_jupyter_env:
                notebook_code_for_include = get_notebook_executed_code()

            for i, obj in enumerate(include):
                obj_name_str = getattr(obj, "__name__", str(obj))
                # logger.debug(f"Decorator: Getting source for manually included object: {obj_name_str}")
                # Pass notebook code only if available and needed by get_model_source
                obj_source = get_model_source(
                    obj, notebook_code_for_include if is_jupyter_env else None
                )
                if obj_source:
                    included_sources[obj] = obj_source
                    # Decide how to pass included source - keep using Env Vars for now
                    env_key_base = f"INCLUDED_OBJECT_{i}"
                    # Correct check for string type - Linter might complain but it's safer
                    if isinstance(obj_source, str):  # type: ignore
                        all_env.append(
                            V1EnvVar(key=f"{env_key_base}_SOURCE", value=obj_source)
                        )
                        # logger.debug(f"Decorator: Added string source to env for included obj: {obj_name_str}")
                    elif isinstance(obj_source, tuple):
                        # Handle tuple source (origin, args) - assumes get_model_source/get_type_source logic
                        # Ensure obj_source is indeed a tuple before unpacking
                        if len(obj_source) == 2:
                            # Now safe to unpack
                            origin_src, arg_srcs = obj_source  # type: ignore
                            if origin_src and isinstance(origin_src, str):
                                all_env.append(
                                    V1EnvVar(
                                        key=f"{env_key_base}_SOURCE", value=origin_src
                                    )
                                )
                            # Handle arg_srcs - ensure it's iterable (list)
                            # Linter complains about "Never" not iterable, check type explicitly
                            if isinstance(arg_srcs, list):
                                for j, arg_src in enumerate(arg_srcs):
                                    # Ensure arg_src is string before adding
                                    if isinstance(arg_src, str):
                                        all_env.append(
                                            V1EnvVar(
                                                key=f"{env_key_base}_ARG_{j}_SOURCE",
                                                value=arg_src,
                                            )
                                        )
                            else:
                                logger.warning(
                                    f"Decorator: Expected arg_srcs to be a list, got {type(arg_srcs)}"
                                )
                        else:
                            # Handle unexpected type or structure for obj_source if necessary
                            logger.warning(
                                f"Decorator: Unexpected obj_source structure: {obj_source}"
                            )
                    else:
                        logger.warning(
                            f"Unknown source type for included object {obj_name_str}: {type(obj_source)}"
                        )
                else:
                    logger.warning(
                        f"Could not retrieve source for manually included object: {obj_name_str}. It might not be available in the consumer."
                    )
        # --- End Manually Included Objects ---

        # --- Validate Function Signature and Types (Keep as is) ---
        logger.debug(
            f"Decorator: Validating signature and type hints for {processor_name}..."
        )
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if len(params) != 1:
            raise TypeError(
                f"{processor_name} must take exactly one parameter"
            )  # Stricter check

        try:
            # Attempt to resolve type hints
            type_hints = get_type_hints(func, globalns=func.__globals__, localns=None)
            logger.debug(f"Decorator: Resolved type hints: {type_hints}")
        except NameError as e:
            # Specific handling for NameError (common in notebooks/dynamic environments)
            logger.warning(
                f"Could not fully resolve type hints for {processor_name} due to NameError: {e}. Type validation might be incomplete."
            )
            # Try to get raw annotations as fallback?
            type_hints = getattr(func, "__annotations__", {})
            logger.debug(f"Decorator: Using raw annotations as fallback: {type_hints}")
        except Exception as e:
            logger.error(f"Decorator: Error getting type hints: {e}")
            # Potentially re-raise or handle based on severity
            raise TypeError(
                f"Could not evaluate type hints for {processor_name}: {e}. Ensure all type dependencies are defined or imported."
            ) from e

        param_name = params[0].name
        if param_name not in type_hints:
            # Allow missing param type hint if using raw annotations? Maybe not.
            raise TypeError(
                f"{processor_name} parameter '{param_name}' must have a type hint"
            )
        param_type = type_hints.get(
            param_name
        )  # Use .get for safety with raw annotations fallback
        param_type_str_repr = str(param_type)  # Use string representation
        logger.debug(
            f"Decorator: Parameter '{param_name}' type hint: {param_type_str_repr}"
        )

        return_type = type_hints.get("return")
        return_type_str_repr = str(return_type)
        logger.debug(f"Decorator: Return type hint: {return_type_str_repr}")

        # --- Determine Input Type (StreamMessage, ContentType) ---
        # This logic remains mostly the same, using the resolved types
        logger.debug(
            f"Decorator: Determining input type structure for param type hint: {param_type_str_repr}"
        )
        origin = get_origin(param_type) if param_type else None
        args = get_args(param_type) if param_type else tuple()
        logger.debug(
            f"Decorator: For param_type '{param_type_str_repr}': origin = {origin!s}, args = {args!s}"
        )  # More detailed log
        print(
            f"Decorator: For param_type '{param_type_str_repr}': origin = {origin!s}, args = {args!s}"
        )  # More detailed log

        is_stream_message = False
        content_type = None
        content_type_name_from_regex = None  # Store regex result here

        # Use Message class directly for comparison
        message_cls = Message  # Get the class object

        # Check 1: Standard introspection
        if origin is message_cls or (
            isinstance(origin, type) and origin is message_cls
        ):
            logger.debug(
                "Decorator: Input type identified as Message via get_origin/isinstance."
            )
            is_stream_message = True
            if args:
                content_type = args[0]
                logger.debug(
                    f"Decorator: Content type extracted via get_args: {content_type}"
                )
            else:
                logger.debug(
                    "Decorator: Message detected, but no generic arguments found via get_args. Attempting regex fallback on string repr."
                )
                # --- Regex Fallback Start ---
                match = re.search(r"Message\[([\w\.]+)\]", param_type_str_repr)
                if match:
                    content_type_name_from_regex = match.group(1)
                    logger.debug(
                        f"Decorator: Extracted content type name via regex: {content_type_name_from_regex}"
                    )
                else:
                    logger.debug(
                        "Decorator: Regex fallback failed to extract content type name."
                    )
                # --- Regex Fallback End ---
        # Check 2a: Regex fallback if get_origin failed but string matches pattern
        elif origin is None and param_type is not None:
            logger.debug(
                "Decorator: get_origin failed. Attempting regex fallback on string representation."
            )
            match = re.search(r"Message\[([\w\.]+)\]", param_type_str_repr)
            if match:
                logger.debug(
                    "Decorator: Regex fallback successful after get_origin failed."
                )
                is_stream_message = True
                content_type_name_from_regex = match.group(1)
                # We don't have the actual content_type object here, only the name
                content_type = None
                logger.debug(
                    f"Decorator: Extracted content type name via regex: {content_type_name_from_regex}"
                )
            else:
                logger.debug(
                    "Decorator: Regex fallback also failed. Treating as non-Message type."
                )
                is_stream_message = False
                content_type = None
        # Check 2: Direct type check (Handles cases where get_origin might fail but type is correct)
        elif isinstance(param_type, type) and param_type is message_cls:
            # This case likely won't have generic args accessible easily if get_origin failed
            logger.debug(
                "Decorator: Input type identified as direct Message type. Attempting regex fallback."
            )
            is_stream_message = True
            # --- Regex Fallback Start ---
            match = re.search(r"Message\[([\w\.]+)\]", param_type_str_repr)
            if match:
                content_type_name_from_regex = match.group(1)
                logger.debug(
                    f"Decorator: Extracted content type name via regex: {content_type_name_from_regex}"
                )
            else:
                logger.debug(
                    "Decorator: Regex fallback failed to extract content type name."
                )
            # --- Regex Fallback End ---
        # Check 3: Removed old placeholder elif branch

        else:  # Handle cases where param_type might be None or origin is something else
            logger.debug(
                f"Decorator: Input parameter '{param_name}' type ({param_type_str_repr}) identified as non-Message type."
            )

        logger.debug(
            f"Decorator: Final Input Type Determination: is_stream_message={is_stream_message}, content_type={content_type}"
        )
        # --- End Input Type Determination ---

        # --- Validate Types are BaseModel ---
        logger.debug(
            "Decorator: Validating parameter and return types are BaseModel subclasses..."
        )

        # Define check_basemodel locally or ensure it's available
        def check_basemodel(type_to_check: Optional[Any], desc: str):
            # logger.debug(f"Decorator check_basemodel: Checking {desc} - Type: {type_to_check}") # Verbose
            if type_to_check is None or type_to_check is Any:
                logger.debug(
                    f"Decorator check_basemodel: Skipping check for {desc} (type is None or Any)."
                )
                return
            # Handle Optional[T] by getting the inner type
            actual_type = type_to_check
            type_origin = get_origin(type_to_check)
            if (
                type_origin is Optional or str(type_origin) == "typing.Union"
            ):  # Handle Optional and Union for None
                type_args = get_args(type_to_check)
                # Find the first non-None type argument
                non_none_args = [arg for arg in type_args if arg is not type(None)]
                if len(non_none_args) == 1:
                    actual_type = non_none_args[0]
                    # logger.debug(f"Decorator check_basemodel: Unwrapped Optional/Union to {actual_type} for {desc}")
                else:
                    # Handle complex Unions later if needed, skip check for now
                    logger.debug(
                        f"Decorator check_basemodel: Skipping check for complex Union {desc}: {type_to_check}"
                    )
                    return

            # Check the actual type
            effective_type = (
                get_origin(actual_type) or actual_type
            )  # Handle generics like List[Model]
            # logger.debug(f"Decorator check_basemodel: Effective type for {desc}: {effective_type}") # Verbose
            if isinstance(effective_type, type) and not issubclass(
                effective_type, BaseModel
            ):
                # Allow non-BaseModel basic types (str, int, bool, float, dict, list)
                allowed_non_model_types = (
                    str,
                    int,
                    float,
                    bool,
                    dict,
                    list,
                    type(None),
                )
                if effective_type not in allowed_non_model_types:
                    logger.error(
                        f"Decorator check_basemodel: Error - {desc} effective type ({effective_type.__name__}) is not BaseModel or standard type."
                    )
                    raise TypeError(
                        f"{desc} effective type ({effective_type.__name__}) must be BaseModel subclass or standard type (str, int, etc.)"
                    )
                else:
                    logger.debug(
                        f"Decorator check_basemodel: OK - {desc} is standard type {effective_type.__name__}."
                    )

            elif not isinstance(effective_type, type):
                # Allow TypeVars or other constructs for now? Or enforce BaseModel? Enforce for now.
                logger.warning(
                    f"Decorator check_basemodel: Warning - {desc} effective type '{effective_type}' is not a class. Cannot verify BaseModel subclass."
                )
                # Revisit this if TypeVars bound to BaseModel are needed.
            else:
                logger.debug(
                    f"Decorator check_basemodel: OK - {desc} effective type ({effective_type.__name__}) is a BaseModel subclass."
                )

        effective_param_type = (
            content_type
            if is_stream_message and content_type
            else param_type
            if not is_stream_message
            else None  # If just Message without content type, param is Message itself (not BaseModel)
        )
        # Check param only if it's not the base Message class
        if effective_param_type is not message_cls:
            check_basemodel(effective_param_type, f"Parameter '{param_name}'")
        check_basemodel(return_type, "Return value")
        logger.debug("Decorator: Type validation complete.")
        # --- End Type Validation ---

        # --- Validate Execution Mode ---
        if execution_mode not in ["inline", "subprocess"]:
            raise ValueError(
                f"Invalid execution_mode: '{execution_mode}'. Must be 'inline' or 'subprocess'."
            )
        logger.debug(f"Decorator: Using execution mode: {execution_mode}")
        # --- End Execution Mode Validation ---

        # --- Populate Environment Variables ---
        logger.debug("Decorator: Populating environment variables...")
        # Keep: FUNCTION_NAME, PARAM_TYPE_STR, RETURN_TYPE_STR, IS_STREAM_MESSAGE, CONTENT_TYPE_NAME, MODULE_NAME
        # Add: NEBU_ENTRYPOINT_MODULE_PATH
        # Add: Included object sources (if any)
        # Add: INIT_FUNC_NAME (if provided)

        # Calculate module_path based on relative file path
        calculated_module_path = None
        if rel_func_path:
            # Convert OS-specific path to module path (e.g., subdir/file.py -> subdir.file)
            base, ext = os.path.splitext(rel_func_path)
            if ext == ".py":
                module_path_parts = base.split(os.sep)
                if module_path_parts[-1] == "__init__":
                    module_path_parts.pop()  # Remove __init__
                # Filter out potential empty strings if path started with / or had //
                module_path_parts = [part for part in module_path_parts if part]
                calculated_module_path = ".".join(module_path_parts)
            else:
                # Not a python file? Should not happen based on inspect.getfile
                logger.warning(
                    f"Decorator: Function source file is not a .py file: {rel_func_path}"
                )
                # Set calculated_module_path to None explicitly to trigger fallback later
                calculated_module_path = None
        else:
            # Should have errored earlier if rel_func_path is None
            logger.warning(
                "Decorator: Could not determine relative function path. Falling back to func.__module__."
            )
            # Set calculated_module_path to None explicitly to trigger fallback later
            calculated_module_path = None

        # Assign final module_path using fallback if calculation failed or wasn't applicable
        if calculated_module_path is not None:
            module_path = calculated_module_path
            logger.debug(f"Decorator: Using calculated module path: {module_path}")
        else:
            module_path = func.__module__  # Fallback
            logger.debug(f"Decorator: Falling back to func.__module__: {module_path}")

        # Basic info needed by consumer to find and run the function
        all_env.append(V1EnvVar(key="FUNCTION_NAME", value=actual_function_name))
        all_env.append(V1EnvVar(key="IS_ASYNC_FUNCTION", value=str(is_async)))

        if rel_func_path:
            # For now, just pass the relative file path, consumer will handle conversion
            all_env.append(
                V1EnvVar(key="NEBU_ENTRYPOINT_MODULE_PATH", value=rel_func_path)
            )
            logger.debug(
                f"Decorator: Set NEBU_ENTRYPOINT_MODULE_PATH to: {rel_func_path}"
            )
        # No else needed, handled by fallback calculation above

        if init_func:
            init_func_name = init_func.__name__  # Get name here
            # Validate signature (must take no arguments) - moved validation earlier conceptually
            before_sig = inspect.signature(init_func)
            if len(before_sig.parameters) != 0:
                raise TypeError(
                    f"init_func '{init_func_name}' must take zero parameters"
                )
            all_env.append(V1EnvVar(key="INIT_FUNC_NAME", value=init_func_name))
            logger.debug(f"Decorator: Set INIT_FUNC_NAME to: {init_func_name}")

        # Type info (still useful for deserialization/validation in consumer)
        # Adjust type strings to replace '__main__' with the calculated module path
        param_type_str_repr = str(param_type)
        if module_path != "__main__" and "__main__." in param_type_str_repr:
            # Be careful with replacement - replace only module prefix
            # Example: "<class '__main__.MyModel'>" -> "<class 'mymodule.MyModel'>"
            # Example: "typing.Optional[__main__.MyModel]" -> "typing.Optional[mymodule.MyModel]"
            param_type_str_repr = param_type_str_repr.replace(
                "__main__.", f"{module_path}."
            )
            logger.debug(
                f"Decorator: Adjusted param type string: {param_type_str_repr}"
            )

        all_env.append(V1EnvVar(key="PARAM_TYPE_STR", value=param_type_str_repr))

        return_type_str_repr = str(return_type)
        if module_path != "__main__" and "__main__." in return_type_str_repr:
            return_type_str_repr = return_type_str_repr.replace(
                "__main__.", f"{module_path}."
            )
            logger.debug(
                f"Decorator: Adjusted return type string: {return_type_str_repr}"
            )

        all_env.append(V1EnvVar(key="RETURN_TYPE_STR", value=return_type_str_repr))
        all_env.append(V1EnvVar(key="IS_STREAM_MESSAGE", value=str(is_stream_message)))

        # Determine and set CONTENT_TYPE_NAME using object or regex fallback
        content_type_name_to_set = None
        if content_type and isinstance(content_type, type):
            content_type_name_to_set = content_type.__name__
            logger.debug(
                f"Decorator: Using content type name from resolved type object: {content_type_name_to_set}"
            )
        elif content_type_name_from_regex:
            content_type_name_to_set = content_type_name_from_regex
            logger.debug(
                f"Decorator: Using content type name from regex fallback: {content_type_name_to_set}"
            )
        else:
            # Only warn if it was supposed to be a Message type
            if is_stream_message:
                logger.warning(
                    f"Could not determine CONTENT_TYPE_NAME for Message parameter {param_name} ({param_type_str_repr}). Consumer might use raw content."
                )

        if content_type_name_to_set:
            all_env.append(
                V1EnvVar(key="CONTENT_TYPE_NAME", value=content_type_name_to_set)
            )

        # Use the calculated module_path for MODULE_NAME
        all_env.append(
            V1EnvVar(
                key="MODULE_NAME", value=module_path
            )  # module_path is guaranteed to be a string here (calculated or fallback)
        )
        logger.debug(f"Decorator: Set MODULE_NAME to: {module_path}")

        # Add Execution Mode
        all_env.append(V1EnvVar(key="NEBU_EXECUTION_MODE", value=execution_mode))
        logger.debug(f"Decorator: Set NEBU_EXECUTION_MODE to: {execution_mode}")

        # Add processor_name as a separate environment variable if needed for logging/identification in consumer
        all_env.append(V1EnvVar(key="PROCESSOR_RESOURCE_NAME", value=processor_name))
        logger.debug(f"Decorator: Set PROCESSOR_RESOURCE_NAME to: {processor_name}")

        # Add Hot Reload Configuration
        if not hot_reload:
            all_env.append(V1EnvVar(key="NEBU_DISABLE_HOT_RELOAD", value="1"))
            logger.debug(
                "Decorator: Set NEBU_DISABLE_HOT_RELOAD to: 1 (Hot reload disabled)"
            )
        else:
            # Ensure it's explicitly '0' or unset if enabled (consumer defaults to enabled if var missing)
            # Setting to "0" might be clearer than removing it if it was added by other means.
            # Check if it exists and update, otherwise add "0"
            existing_hot_reload_var = next(
                (var for var in all_env if var.key == "NEBU_DISABLE_HOT_RELOAD"), None
            )
            if existing_hot_reload_var:
                existing_hot_reload_var.value = "0"
            else:
                # Not strictly needed as consumer defaults to enabled, but explicit is good.
                all_env.append(V1EnvVar(key="NEBU_DISABLE_HOT_RELOAD", value="0"))
            logger.debug(
                "Decorator: Hot reload enabled (NEBU_DISABLE_HOT_RELOAD=0 or unset)"
            )

        # Add PYTHONPATH
        pythonpath_value = CONTAINER_CODE_DIR
        existing_pythonpath = next(
            (var for var in all_env if var.key == "PYTHONPATH"), None
        )
        if existing_pythonpath:
            if existing_pythonpath.value:
                # Prepend our code dir, ensuring no duplicates and handling separators
                paths = [p for p in existing_pythonpath.value.split(":") if p]
                if pythonpath_value not in paths:
                    paths.insert(0, pythonpath_value)
                existing_pythonpath.value = ":".join(paths)
            else:
                existing_pythonpath.value = pythonpath_value
        else:
            all_env.append(V1EnvVar(key="PYTHONPATH", value=pythonpath_value))
        logger.debug(f"Decorator: Ensured PYTHONPATH includes: {pythonpath_value}")

        logger.debug("Decorator: Finished populating environment variables.")
        # --- End Environment Variables ---

        # --- Add S3 Sync Volume ---
        if s3_destination_uri:
            logger.debug(
                f"Decorator: Adding volume to sync S3 code from {s3_destination_uri} to {CONTAINER_CODE_DIR}"
            )
            s3_sync_volume = V1VolumePath(
                source=s3_destination_uri,
                dest=CONTAINER_CODE_DIR,
                driver=V1VolumeDriver.RCLONE_SYNC,  # Use SYNC for one-way download
                # Add flags if needed, e.g., --checksum, --fast-list?
            )
            # Check if an identical volume already exists
            if not any(
                v.source == s3_sync_volume.source and v.dest == s3_sync_volume.dest
                for v in all_volumes
            ):
                all_volumes.append(s3_sync_volume)
            else:
                logger.debug(
                    f"Decorator: Volume for {s3_destination_uri} to {CONTAINER_CODE_DIR} already exists."
                )
        else:
            # Should have errored earlier if S3 upload failed
            raise RuntimeError(
                "Internal Error: S3 destination URI not set, cannot add sync volume."
            )
        # --- End S3 Sync Volume ---

        # --- Final Setup ---
        logger.debug("Decorator: Preparing final Processor object...")

        # Determine ResolvedInputType for Processor Generic
        ResolvedInputType: type[BaseModel] = (
            BaseModel  # Default to BaseModel to satisfy generic bound
        )
        if is_stream_message:
            if (
                content_type
                and isinstance(content_type, type)
                and issubclass(content_type, BaseModel)
            ):
                ResolvedInputType = content_type
            else:
                logger.warning(
                    f"Decorator: Message type hint found, but ContentType '{content_type!s}' is not a valid Pydantic Model. Defaulting InputType to BaseModel."
                )
                # ResolvedInputType remains BaseModel (default)
        elif (
            param_type
            and isinstance(param_type, type)
            and issubclass(param_type, BaseModel)
        ):
            ResolvedInputType = param_type  # Function takes the data model directly
        else:
            logger.warning(
                f"Decorator: Parameter type '{param_type!s}' is not a valid Pydantic Model. Defaulting InputType to BaseModel."
            )
            # ResolvedInputType remains BaseModel (default)

        ResolvedOutputType: type[BaseModel] = BaseModel  # Default to BaseModel
        if (
            return_type
            and isinstance(return_type, type)
            and issubclass(return_type, BaseModel)
        ):
            ResolvedOutputType = return_type
        elif return_type is not None:  # It was something, but not a BaseModel subclass
            logger.warning(
                f"Decorator: Return type '{return_type!s}' is not a valid Pydantic Model. Defaulting OutputType to BaseModel."
            )
        # Else (return_type is None), ResolvedOutputType remains BaseModel

        logger.debug(
            f"Decorator: Resolved Generic Types for Processor: InputType={ResolvedInputType.__name__}, OutputType={ResolvedOutputType.__name__}"
        )

        metadata = V1ResourceMetaRequest(
            name=processor_name, namespace=effective_namespace, labels=labels
        )
        # Base command now just runs the consumer module, relies on PYTHONPATH finding code
        consumer_module = "nebulous.processors.consumer"
        if "accelerate launch" in python_cmd:
            consumer_execution_command = f"{python_cmd.strip()} -m {consumer_module}"
        else:
            # Standard python execution
            consumer_execution_command = f"{python_cmd} -u -m {consumer_module}"

        # Setup commands: Base dependencies needed by consumer.py itself or the framework
        # Install required dependencies for the consumer to run properly
        base_deps_install = (
            "pip install orign redis PySocks pydantic dill boto3 requests"
        )
        setup_commands_list = [base_deps_install]

        if setup_script:
            logger.debug("Decorator: Adding user setup script to setup commands.")
            setup_commands_list.append(setup_script.strip())

        if debug:
            all_env.append(V1EnvVar(key="PYTHON_LOG", value="DEBUG"))

        # Combine setup commands and the final execution command
        all_commands = setup_commands_list + [consumer_execution_command]
        # Use newline separator for clarity in logs and script execution
        final_command = "\n".join(all_commands)

        logger.debug(
            f"Decorator: Final container command:\\n-------\\n{final_command}\\n-------"
        )

        container_request = V1ContainerRequest(
            image=image,
            command=final_command,
            env=all_env,
            volumes=all_volumes,  # Use updated volumes list
            accelerators=accelerators,
            resources=resources,
            meters=meters,
            restart="Always",  # Consider making this configurable? Defaulting to Always
            authz=authz,
            platform=platform,
            metadata=metadata,
            # Pass through optional parameters from the main decorator function
            queue=queue,
            timeout=timeout,
            ssh_keys=ssh_keys,
            ports=ports,
            proxy_port=proxy_port,
            health_check=health_check,
        )
        logger.debug("Decorator: Final Container Request Env Vars (Summary):")
        for env_var in all_env:
            # Avoid printing potentially large included source code
            value_str = env_var.value or ""
            if "SOURCE" in env_var.key and len(value_str) > 100:
                logger.debug(
                    f"  {env_var.key}: <source code present, length={len(value_str)}>"
                )
            else:
                logger.debug(f"  {env_var.key}: {value_str}")

        # Create the generically typed Processor instance
        _processor_instance = Processor[ResolvedInputType, ResolvedOutputType](
            name=processor_name,
            namespace=effective_namespace,
            labels=labels,
            container=container_request,
            input_model_cls=ResolvedInputType,
            output_model_cls=ResolvedOutputType,
            common_schema=None,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            scale_config=scale,
            config=effective_config,
            api_key=api_key,
            no_delete=no_delete,
            wait_for_healthy=wait_for_healthy,
        )
        # Type hint for the variable. The instance itself IS correctly typed with specific models.
        processor_instance: Processor[BaseModel, BaseModel] = _processor_instance

        logger.debug(
            f"Decorator: Processor instance '{processor_name}' created successfully with generic types."
        )
        # Store original func for potential local invocation/testing? Keep for now.
        # TODO: Add original_func to Processor model definition if this is desired
        # Commenting out as Processor model does not have this field
        # try:
        #     # This will fail if Processor hasn't been updated to include this field
        #     processor_instance.original_func = func  # type: ignore
        # except AttributeError:
        #     logger.warning(
        #         "Could not assign original_func to Processor instance. Update Processor model or remove assignment."
        #     )

        # Store the stream flag so the Processor instance knows to operate in generator mode by default
        try:
            setattr(_processor_instance, "_stream_enabled", stream)
        except Exception:
            # If for some reason setattr fails, log but do not break creation
            logger.warning(
                "Failed to set _stream_enabled attribute on Processor instance – streaming may not work as expected."
            )

        return processor_instance

    return decorator
