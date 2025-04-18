import ast  # For parsing notebook code
import inspect
import json  # Add json import
import os  # Add os import
import re  # Import re for fallback check
import tempfile  # Add tempfile import
import textwrap
import uuid  # Add uuid import
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
from urllib.parse import urlparse  # Add urlparse import

import dill  # Add dill import
import requests  # Add requests import
from botocore.exceptions import ClientError  # Import ClientError
from pydantic import BaseModel

from nebu.containers.models import (
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
from nebu.data import Bucket  # Import Bucket
from nebu.meta import V1ResourceMetaRequest
from nebu.processors.models import (
    Message,
    V1Scale,
)
from nebu.processors.processor import Processor

from .default import DEFAULT_MAX_REPLICAS, DEFAULT_MIN_REPLICAS, DEFAULT_SCALE

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R", bound=BaseModel)

# Attribute name for explicitly stored source
_NEBU_EXPLICIT_SOURCE_ATTR = "_nebu_explicit_source"
# Environment variable to prevent decorator recursion inside consumer
_NEBU_INSIDE_CONSUMER_ENV_VAR = "_NEBU_INSIDE_CONSUMER_EXEC"

# Define target directory in container
CONTAINER_CODE_DIR = "/app/src"
# Define S3 prefix for code storage (under the base URI from token endpoint)
S3_CODE_PREFIX = "nebu-code"
# Define the token endpoint URL (replace with actual URL)
# Use environment variable for flexibility, provide a default for local dev
NEBU_API_BASE_URL = os.environ.get("NEBU_API_BASE_URL", "http://localhost:8080")
S3_TOKEN_ENDPOINT = f"{NEBU_API_BASE_URL}/iam/s3-token"

# --- Jupyter Helper Functions ---


def is_jupyter_notebook():
    """
    Determine if the current code is running inside a Jupyter notebook.
    Returns bool: True if running inside a Jupyter notebook, False otherwise.
    """
    # print("[DEBUG Helper] Checking if running in Jupyter...") # Reduce verbosity
    try:
        # Use importlib to avoid runtime dependency if not needed
        import importlib.util

        if importlib.util.find_spec("IPython") is None:
            return False
        import IPython  # Now safe to import

        ip = IPython.get_ipython()
        if ip is None:
            # print("[DEBUG Helper] is_jupyter_notebook: No IPython instance found.")
            return False
        class_name = str(ip.__class__)
        # print(f"[DEBUG Helper] is_jupyter_notebook: IPython class name: {class_name}")
        if "ZMQInteractiveShell" in class_name:
            # print("[DEBUG Helper] is_jupyter_notebook: Jupyter detected (ZMQInteractiveShell).")
            return True
        # print("[DEBUG Helper] is_jupyter_notebook: Not Jupyter (IPython instance found, but not ZMQInteractiveShell).")
        return False
    except Exception as e:
        # print(f"[DEBUG Helper] is_jupyter_notebook: Exception occurred: {e}") # Reduce verbosity
        return False


def get_notebook_executed_code():
    """
    Returns all executed code from the current notebook session.
    Returns str or None: All executed code as a string, or None if not possible.
    """
    print("[DEBUG Helper] Attempting to get notebook execution history...")
    try:
        import IPython

        ip = IPython.get_ipython()
        if ip is None or not hasattr(ip, "history_manager"):
            print(
                "[DEBUG Helper] get_notebook_executed_code: No IPython instance or history_manager."
            )
            return None
        history_manager = ip.history_manager
        # Limiting history range for debugging? Maybe get_tail(N)? For now, get all.
        # history = history_manager.get_range(start=1) # type: ignore
        history = list(history_manager.get_range(start=1))  # type: ignore # Convert to list to get length
        print(
            f"[DEBUG Helper] get_notebook_executed_code: Retrieved {len(history)} history entries."
        )
        source_code = ""
        separator = "\n#<NEBU_CELL_SEP>#\n"
        for _, _, content in history:  # Use _ for unused session, lineno
            if isinstance(content, str) and content.strip():
                source_code += content + separator
        print(
            f"[DEBUG Helper] get_notebook_executed_code: Total history source length: {len(source_code)}"
        )
        return source_code
    except Exception as e:
        print(f"[DEBUG Helper] get_notebook_executed_code: Error getting history: {e}")
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
    print(
        f"[DEBUG Helper] Extracting '{def_name}' ({def_type.__name__}) from history string (len: {len(source_string)})..."
    )
    if not source_string or not def_name:
        print("[DEBUG Helper] extract: Empty source string or def_name.")
        return None

    cells = source_string.split("#<NEBU_CELL_SEP>#")
    print(f"[DEBUG Helper] extract: Split history into {len(cells)} potential cells.")
    last_found_source = None

    for i, cell in enumerate(reversed(cells)):
        cell_num = len(cells) - 1 - i
        cell = cell.strip()
        if not cell:
            continue
        # print(f"[DEBUG Helper] extract: Analyzing cell #{cell_num}...") # Can be very verbose
        try:
            tree = ast.parse(cell)
            found_in_cell = False
            for node in ast.walk(tree):
                if (
                    isinstance(node, def_type)
                    and hasattr(node, "name")
                    and node.name == def_name
                ):
                    print(
                        f"[DEBUG Helper] extract: Found node for '{def_name}' in cell #{cell_num}."
                    )
                    try:
                        # Use ast.get_source_segment for accurate extraction (Python 3.8+)
                        func_source = ast.get_source_segment(cell, node)
                        if func_source:
                            print(
                                f"[DEBUG Helper] extract: Successfully extracted source using get_source_segment for '{def_name}'."
                            )
                            last_found_source = func_source
                            found_in_cell = True
                            break  # Stop searching this cell
                    except AttributeError:  # Fallback for Python < 3.8
                        print(
                            f"[DEBUG Helper] extract: get_source_segment failed (likely Py < 3.8), using fallback for '{def_name}'."
                        )
                        start_lineno = getattr(node, "lineno", 1) - 1
                        end_lineno = getattr(node, "end_lineno", start_lineno + 1)

                        if hasattr(node, "decorator_list") and node.decorator_list:
                            first_decorator_start_line = (
                                getattr(
                                    node.decorator_list[0], "lineno", start_lineno + 1
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
                                print(
                                    f"[DEBUG Helper] extract: Extracted source via fallback for '{def_name}'."
                                )
                                found_in_cell = True
                                break
                        else:
                            print(
                                f"[DEBUG Helper] extract: Warning: Line numbers out of bounds for {def_name} in cell (fallback)."
                            )

            if found_in_cell:
                print(
                    f"[DEBUG Helper] extract: Found and returning source for '{def_name}' from cell #{cell_num}."
                )
                return last_found_source  # Found last definition, return immediately

        except (SyntaxError, ValueError) as e:
            # print(f"[DEBUG Helper] extract: Skipping cell #{cell_num} due to parse error: {e}") # Can be verbose
            continue
        except Exception as e:
            print(
                f"[DEBUG Helper] extract: Warning: AST processing error for {def_name} in cell #{cell_num}: {e}"
            )
            continue

    if not last_found_source:
        print(
            f"[DEBUG Helper] extract: Definition '{def_name}' of type {def_type.__name__} not found in history search."
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
        print(
            f"[DEBUG @include] Successfully captured source for: {getattr(obj, '__name__', str(obj))}"
        )
    except Exception as e:
        # Don't fail the definition, just warn
        print(
            f"Warning: @include could not capture source for {getattr(obj, '__name__', str(obj))}: {e}. Automatic source retrieval will be attempted later."
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
    print(f"[DEBUG get_model_source] Getting source for: {model_name_str}")
    # 1. Check explicit source
    explicit_source = getattr(model_class, _NEBU_EXPLICIT_SOURCE_ATTR, None)
    if explicit_source:
        print(
            f"[DEBUG get_model_source] Using explicit source (@include) for: {model_name_str}"
        )
        return explicit_source

    # 2. Check notebook history
    if notebook_code and hasattr(model_class, "__name__"):
        print(
            f"[DEBUG get_model_source] Attempting notebook history extraction for: {model_class.__name__}"
        )
        extracted_source = extract_definition_source_from_string(
            notebook_code, model_class.__name__, ast.ClassDef
        )
        if extracted_source:
            print(
                f"[DEBUG get_model_source] Using notebook history source for: {model_class.__name__}"
            )
            return extracted_source
        else:
            print(
                f"[DEBUG get_model_source] Notebook history extraction failed for: {model_class.__name__}. Proceeding to dill."
            )

    # 3. Fallback to dill
    try:
        print(
            f"[DEBUG get_model_source] Attempting dill fallback for: {model_name_str}"
        )
        source = dill.source.getsource(model_class)
        print(f"[DEBUG get_model_source] Using dill source for: {model_name_str}")
        return textwrap.dedent(source)
    except (IOError, TypeError, OSError) as e:
        print(
            f"[DEBUG get_model_source] Failed dill fallback for: {model_name_str}: {e}"
        )
        return None


# Reintroduce get_type_source to handle generics
def get_type_source(
    type_obj: Any, notebook_code: Optional[str] = None
) -> Optional[Any]:
    """Get the source code for a type, including generic parameters."""
    type_obj_str = str(type_obj)
    print(f"[DEBUG get_type_source] Getting source for type: {type_obj_str}")
    origin = get_origin(type_obj)
    args = get_args(type_obj)

    if origin is not None:
        # Use updated get_model_source for origin
        print(
            f"[DEBUG get_type_source] Detected generic type. Origin: {origin}, Args: {args}"
        )
        origin_source = get_model_source(origin, notebook_code)
        args_sources = []

        # Recursively get sources for all type arguments
        for arg in args:
            print(
                f"[DEBUG get_type_source] Recursively getting source for generic arg #{arg}"
            )
            arg_source = get_type_source(arg, notebook_code)
            if arg_source:
                args_sources.append(arg_source)

        # Return tuple only if origin source or some arg sources were found
        if origin_source or args_sources:
            print(
                f"[DEBUG get_type_source] Returning tuple source for generic: {type_obj_str}"
            )
            return (origin_source, args_sources)

    # Fallback if not a class or recognizable generic alias
    # Try get_model_source as a last resort for unknown types
    fallback_source = get_model_source(type_obj, notebook_code)
    if fallback_source:
        print(
            f"[DEBUG get_type_source] Using fallback get_model_source for: {type_obj_str}"
        )
        return fallback_source

    print(f"[DEBUG get_type_source] Failed to get source for: {type_obj_str}")
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
):
    def decorator(
        func: Callable[[Any], Any],
    ) -> Processor | Callable[[Any], Any]:
        # --- Prevent Recursion Guard ---
        if os.environ.get(_NEBU_INSIDE_CONSUMER_ENV_VAR) == "1":
            print(
                f"[DEBUG Decorator] Guard triggered for '{func.__name__}'. Returning original function."
            )
            return func
        # --- End Guard ---

        print(
            f"[DEBUG Decorator Init] @processor decorating function '{func.__name__}'"
        )
        all_env = env or []
        processor_name = func.__name__
        all_volumes = volumes or []  # Initialize volumes list

        # --- Get Decorated Function File Path and Directory ---
        print("[DEBUG Decorator] Getting source file path for decorated function...")
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
            print(f"[DEBUG Decorator] Found real file path: {func_file_path}")
            print(f"[DEBUG Decorator] Found function directory: {func_dir}")
            print(f"[DEBUG Decorator] Relative function path: {rel_func_path}")
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

        print(f"[DEBUG Decorator] Fetching S3 token from: {S3_TOKEN_ENDPOINT}")
        try:
            response = requests.get(S3_TOKEN_ENDPOINT, timeout=10)  # Add timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            s3_token_data = response.json()

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

            # Construct unique S3 path: s3://<base_bucket>/<base_prefix>/<code_prefix>/<processor_name>-<uuid>/
            unique_suffix = f"{processor_name}-{uuid.uuid4()}"
            parsed_base = urlparse(s3_base_uri)
            if not parsed_base.scheme == "s3" or not parsed_base.netloc:
                raise ValueError(f"Invalid s3_base_uri received: {s3_base_uri}")

            base_path = parsed_base.path.strip("/")
            s3_dest_components = [S3_CODE_PREFIX, unique_suffix]
            if base_path:
                # Handle potential multiple path segments in base_path
                s3_dest_components.insert(0, *base_path.split("/"))

            # Filter out empty strings that might result from split
            s3_destination_key_components = [
                comp for comp in s3_dest_components if comp
            ]
            s3_destination_key = (
                "/".join(s3_destination_key_components) + "/"
            )  # Ensure trailing slash for prefix
            s3_destination_uri = f"s3://{parsed_base.netloc}/{s3_destination_key}"

            print(
                f"[DEBUG Decorator] Uploading code from '{func_dir}' to '{s3_destination_uri}'"
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
                delete=False,
                dry_run=False,
            )
            print("[DEBUG Decorator] S3 code upload completed.")

        except requests.exceptions.RequestException as e:
            print(f"ERROR: Failed to fetch S3 token from {S3_TOKEN_ENDPOINT}: {e}")
            raise RuntimeError(
                f"Failed to fetch S3 token from {S3_TOKEN_ENDPOINT}: {e}"
            ) from e
        except ClientError as e:
            print(f"ERROR: Failed to upload code to S3 {s3_destination_uri}: {e}")
            # Attempt to provide more context from the error if possible
            error_code = e.response.get("Error", {}).get("Code")
            error_msg = e.response.get("Error", {}).get("Message")
            print(f"      S3 Error Code: {error_code}, Message: {error_msg}")
            raise RuntimeError(
                f"Failed to upload code to {s3_destination_uri}: {e}"
            ) from e
        except ValueError as e:  # Catch ValueErrors from validation
            print(f"ERROR: Configuration or response data error: {e}")
            raise RuntimeError(f"Configuration or response data error: {e}") from e
        except Exception as e:
            print(f"ERROR: Unexpected error during S3 token fetch or upload: {e}")
            # Consider logging traceback here for better debugging
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Unexpected error during S3 setup: {e}") from e

        # --- Process Manually Included Objects (Keep for now, add source via env) ---
        # This part remains unchanged for now, using @include and environment variables.
        # Future: Could potentially upload these to S3 as well if they become large.
        included_sources: Dict[Any, Any] = {}
        notebook_code_for_include = None  # Get notebook code only if needed for include
        if include:
            # Determine if we are in Jupyter only if needed for include fallback
            # print("[DEBUG Decorator] Processing manually included objects...")
            is_jupyter_env = is_jupyter_notebook()
            if is_jupyter_env:
                notebook_code_for_include = get_notebook_executed_code()

            for i, obj in enumerate(include):
                obj_name_str = getattr(obj, "__name__", str(obj))
                # print(f"[DEBUG Decorator] Getting source for manually included object: {obj_name_str}")
                # Pass notebook code only if available and needed by get_model_source
                obj_source = get_model_source(
                    obj, notebook_code_for_include if is_jupyter_env else None
                )
                if obj_source:
                    included_sources[obj] = obj_source
                    # Decide how to pass included source - keep using Env Vars for now
                    env_key_base = f"INCLUDED_OBJECT_{i}"
                    if isinstance(obj_source, str):
                        all_env.append(
                            V1EnvVar(key=f"{env_key_base}_SOURCE", value=obj_source)
                        )
                        # print(f"[DEBUG Decorator] Added string source to env for included obj: {obj_name_str}")
                    elif isinstance(obj_source, tuple):
                        # Handle tuple source (origin, args) - assumes get_model_source/get_type_source logic
                        origin_src, arg_srcs = obj_source
                        if origin_src and isinstance(origin_src, str):
                            all_env.append(
                                V1EnvVar(key=f"{env_key_base}_SOURCE", value=origin_src)
                            )
                        for j, arg_src in enumerate(arg_srcs):
                            if isinstance(arg_src, str):
                                all_env.append(
                                    V1EnvVar(
                                        key=f"{env_key_base}_ARG_{j}_SOURCE",
                                        value=arg_src,
                                    )
                                )
                            # Handle nested tuples if necessary, or keep it simple
                        # print(f"[DEBUG Decorator] Added tuple source to env for included obj: {obj_name_str}")
                    else:
                        print(
                            f"Warning: Unknown source type for included object {obj_name_str}: {type(obj_source)}"
                        )
                else:
                    print(
                        f"Warning: Could not retrieve source for manually included object: {obj_name_str}. It might not be available in the consumer."
                    )
        # --- End Manually Included Objects ---

        # --- Validate Function Signature and Types (Keep as is) ---
        print(
            f"[DEBUG Decorator] Validating signature and type hints for {processor_name}..."
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
            print(f"[DEBUG Decorator] Resolved type hints: {type_hints}")
        except NameError as e:
            # Specific handling for NameError (common in notebooks/dynamic environments)
            print(
                f"Warning: Could not fully resolve type hints for {processor_name} due to NameError: {e}. Type validation might be incomplete."
            )
            # Try to get raw annotations as fallback?
            type_hints = getattr(func, "__annotations__", {})
            print(f"[DEBUG Decorator] Using raw annotations as fallback: {type_hints}")
        except Exception as e:
            print(f"[DEBUG Decorator] Error getting type hints: {e}")
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
        print(
            f"[DEBUG Decorator] Parameter '{param_name}' type hint: {param_type_str_repr}"
        )

        if "return" not in type_hints:
            raise TypeError(f"{processor_name} must have a return type hint")
        return_type = type_hints.get("return")
        return_type_str_repr = str(return_type)
        print(f"[DEBUG Decorator] Return type hint: {return_type_str_repr}")

        # --- Determine Input Type (StreamMessage, ContentType) ---
        # This logic remains mostly the same, using the resolved types
        print(
            f"[DEBUG Decorator] Determining input type structure for param type hint: {param_type_str_repr}"
        )
        origin = get_origin(param_type) if param_type else None
        args = get_args(param_type) if param_type else tuple()
        print(f"[DEBUG Decorator] get_origin result: {origin}, get_args result: {args}")
        is_stream_message = False
        content_type = None

        # Use Message class directly for comparison
        message_cls = Message  # Get the class object

        # Check 1: Standard introspection
        if origin is message_cls or (
            isinstance(origin, type) and origin is message_cls
        ):
            print(
                "[DEBUG Decorator] Input type identified as Message via get_origin/isinstance."
            )
            is_stream_message = True
            if args:
                content_type = args[0]
                print(
                    f"[DEBUG Decorator] Content type extracted via get_args: {content_type}"
                )
            else:
                print(
                    "[DEBUG Decorator] Message detected, but no generic arguments found via get_args."
                )
        # Check 2: Direct type check (Handles cases where get_origin might fail but type is correct)
        elif isinstance(param_type, type) and param_type is message_cls:
            print("[DEBUG Decorator] Input type identified as direct Message type.")
            is_stream_message = True
        # Check 3: Regex fallback might be less reliable now, but keep as last resort?
        elif (
            origin is None and param_type is not None
        ):  # Only if origin failed and type exists
            # ... (existing regex fallback logic using param_type_str_repr) ...
            pass  # Keep existing regex logic here if desired

        else:  # Handle cases where param_type might be None or origin is something else
            print(
                f"[DEBUG Decorator] Input parameter '{param_name}' type ({param_type_str_repr}) identified as non-Message type."
            )

        print(
            f"[DEBUG Decorator] Final Input Type Determination: is_stream_message={is_stream_message}, content_type={content_type}"
        )
        # --- End Input Type Determination ---

        # --- Validate Types are BaseModel ---
        print(
            "[DEBUG Decorator] Validating parameter and return types are BaseModel subclasses..."
        )

        # Define check_basemodel locally or ensure it's available
        def check_basemodel(type_to_check: Optional[Any], desc: str):
            # print(f"[DEBUG Decorator] check_basemodel: Checking {desc} - Type: {type_to_check}") # Verbose
            if type_to_check is None or type_to_check is Any:
                print(
                    f"[DEBUG Decorator] check_basemodel: Skipping check for {desc} (type is None or Any)."
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
                    # print(f"[DEBUG Decorator] check_basemodel: Unwrapped Optional/Union to {actual_type} for {desc}")
                else:
                    # Handle complex Unions later if needed, skip check for now
                    print(
                        f"[DEBUG Decorator] check_basemodel: Skipping check for complex Union {desc}: {type_to_check}"
                    )
                    return

            # Check the actual type
            effective_type = (
                get_origin(actual_type) or actual_type
            )  # Handle generics like List[Model]
            # print(f"[DEBUG Decorator] check_basemodel: Effective type for {desc}: {effective_type}") # Verbose
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
                    print(
                        f"[DEBUG Decorator] check_basemodel: Error - {desc} effective type ({effective_type.__name__}) is not BaseModel or standard type."
                    )
                    raise TypeError(
                        f"{desc} effective type ({effective_type.__name__}) must be BaseModel subclass or standard type (str, int, etc.)"
                    )
                else:
                    print(
                        f"[DEBUG Decorator] check_basemodel: OK - {desc} is standard type {effective_type.__name__}."
                    )

            elif not isinstance(effective_type, type):
                # Allow TypeVars or other constructs for now? Or enforce BaseModel? Enforce for now.
                print(
                    f"[DEBUG Decorator] check_basemodel: Warning - {desc} effective type '{effective_type}' is not a class. Cannot verify BaseModel subclass."
                )
                # Revisit this if TypeVars bound to BaseModel are needed.
            else:
                print(
                    f"[DEBUG Decorator] check_basemodel: OK - {desc} effective type ({effective_type.__name__}) is a BaseModel subclass."
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
        print("[DEBUG Decorator] Type validation complete.")
        # --- End Type Validation ---

        # --- Populate Environment Variables ---
        print("[DEBUG Decorator] Populating environment variables...")
        # Keep: FUNCTION_NAME, PARAM_TYPE_STR, RETURN_TYPE_STR, IS_STREAM_MESSAGE, CONTENT_TYPE_NAME, MODULE_NAME
        # Add: NEBU_ENTRYPOINT_MODULE_PATH
        # Add: Included object sources (if any)
        # Add: INIT_FUNC_NAME (if provided)

        # Basic info needed by consumer to find and run the function
        all_env.append(V1EnvVar(key="FUNCTION_NAME", value=processor_name))
        if rel_func_path:
            # Convert OS-specific path to module path (e.g., subdir/file.py -> subdir.file)
            module_path = rel_func_path.replace(os.sep, ".")
            if module_path.endswith(".py"):
                module_path = module_path[:-3]
            # Handle __init__.py -> treat as package name
            if module_path.endswith(".__init__"):
                module_path = module_path[: -len(".__init__")]
            elif module_path == "__init__":  # Top-level __init__.py
                module_path = ""  # Or handle differently? Let's assume it means import '.'? Maybe error?

            # For now, just pass the relative file path, consumer will handle conversion
            all_env.append(
                V1EnvVar(key="NEBU_ENTRYPOINT_MODULE_PATH", value=rel_func_path)
            )
            print(
                f"[DEBUG Decorator] Set NEBU_ENTRYPOINT_MODULE_PATH to: {rel_func_path}"
            )
        else:
            # Should have errored earlier if rel_func_path is None
            raise RuntimeError("Internal error: Relative function path not determined.")

        if init_func:
            init_func_name = init_func.__name__  # Get name here
            # Validate signature (must take no arguments) - moved validation earlier conceptually
            before_sig = inspect.signature(init_func)
            if len(before_sig.parameters) != 0:
                raise TypeError(
                    f"init_func '{init_func_name}' must take zero parameters"
                )
            all_env.append(V1EnvVar(key="INIT_FUNC_NAME", value=init_func_name))
            print(f"[DEBUG Decorator] Set INIT_FUNC_NAME to: {init_func_name}")

        # Type info (still useful for deserialization/validation in consumer)
        all_env.append(V1EnvVar(key="PARAM_TYPE_STR", value=param_type_str_repr))
        all_env.append(
            V1EnvVar(key="RETURN_TYPE_STR", value=return_type_str_repr)
        )  # Use repr
        all_env.append(V1EnvVar(key="IS_STREAM_MESSAGE", value=str(is_stream_message)))
        if content_type and hasattr(content_type, "__name__"):
            # Check if content_type is a class before accessing __name__
            if isinstance(content_type, type):
                all_env.append(
                    V1EnvVar(key="CONTENT_TYPE_NAME", value=content_type.__name__)
                )
            else:
                # Handle unresolved types / typevars if needed
                print(
                    f"Warning: Content type '{content_type}' is not a class, cannot get name."
                )
        # MODULE_NAME might be less reliable now, depends on where func is defined relative to project root
        all_env.append(
            V1EnvVar(key="MODULE_NAME", value=func.__module__)
        )  # Keep for potential debugging/info

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
        print(f"[DEBUG Decorator] Ensured PYTHONPATH includes: {pythonpath_value}")

        print("[DEBUG Decorator] Finished populating environment variables.")
        # --- End Environment Variables ---

        # --- Add S3 Sync Volume ---
        if s3_destination_uri:
            print(
                f"[DEBUG Decorator] Adding volume to sync S3 code from {s3_destination_uri} to {CONTAINER_CODE_DIR}"
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
                print(
                    f"[DEBUG Decorator] Volume for {s3_destination_uri} to {CONTAINER_CODE_DIR} already exists."
                )
        else:
            # Should have errored earlier if S3 upload failed
            raise RuntimeError(
                "Internal Error: S3 destination URI not set, cannot add sync volume."
            )
        # --- End S3 Sync Volume ---

        # --- Final Setup ---
        print("[DEBUG Decorator] Preparing final Processor object...")
        metadata = V1ResourceMetaRequest(
            name=processor_name, namespace=namespace, labels=labels
        )
        # Base command now just runs the consumer module, relies on PYTHONPATH finding code
        consumer_module = "nebu.processors.consumer"
        if "accelerate launch" in python_cmd:
            consumer_execution_command = f"{python_cmd.strip()} -m {consumer_module}"
        else:
            # Standard python execution
            consumer_execution_command = f"{python_cmd} -u -m {consumer_module}"

        # Setup commands: Base dependencies needed by consumer.py itself or the framework
        # Assume nebu package (and thus boto3, requests, redis-py, dill, pydantic)
        # are installed in the base image or via other means.
        # User's setup_script is still valuable for *their* specific dependencies.
        setup_commands_list = []
        if setup_script:
            print("[DEBUG Decorator] Adding user setup script to setup commands.")
            setup_commands_list.append(setup_script.strip())

        # Combine setup commands and the final execution command
        all_commands = setup_commands_list + [consumer_execution_command]
        # Use newline separator for clarity in logs and script execution
        final_command = "\n".join(all_commands)

        print(
            f"[DEBUG Decorator] Final container command:\n-------\n{final_command}\n-------"
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
        print("[DEBUG Decorator] Final Container Request Env Vars (Summary):")
        for env_var in all_env:
            # Avoid printing potentially large included source code
            value_str = env_var.value or ""
            if "SOURCE" in env_var.key and len(value_str) > 100:
                print(
                    f"[DEBUG Decorator]  {env_var.key}: <source code present, length={len(value_str)}>"
                )
            else:
                print(f"[DEBUG Decorator]  {env_var.key}: {value_str}")

        processor_instance = Processor(
            name=processor_name,
            namespace=namespace,
            labels=labels,
            container=container_request,
            schema_=None,  # Schema info might be derived differently now if needed
            common_schema=None,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            scale_config=scale,
            no_delete=no_delete,
        )
        print(
            f"[DEBUG Decorator] Processor instance '{processor_name}' created successfully."
        )
        # Store original func for potential local invocation/testing? Keep for now.
        # TODO: Add original_func to Processor model definition if this is desired
        # setattr(processor_instance, 'original_func', func) # Use setattr if not in model
        try:
            # This will fail if Processor hasn't been updated to include this field
            processor_instance.original_func = func
        except AttributeError:
            print(
                "Warning: Could not assign original_func to Processor instance. Update Processor model or remove assignment."
            )

        return processor_instance

    return decorator
