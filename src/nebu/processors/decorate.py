import ast  # For parsing notebook code
import inspect
import re  # Import re for fallback check
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

import dill  # Add dill import
from pydantic import BaseModel

from nebu.containers.models import (
    V1AuthzConfig,
    V1ContainerRequest,
    V1ContainerResources,
    V1EnvVar,
    V1Meter,
    V1VolumePath,
)
from nebu.meta import V1ResourceMetaRequest
from nebu.processors.models import (
    V1Scale,
    V1StreamMessage,
)
from nebu.processors.processor import Processor

from .default import DEFAULT_MAX_REPLICAS, DEFAULT_MIN_REPLICAS, DEFAULT_SCALE

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R", bound=BaseModel)

# Attribute name for explicitly stored source
_NEBU_EXPLICIT_SOURCE_ATTR = "_nebu_explicit_source"

# --- Jupyter Helper Functions ---


def is_jupyter_notebook():
    """
    Determine if the current code is running inside a Jupyter notebook.
    Returns bool: True if running inside a Jupyter notebook, False otherwise.
    """
    print("[DEBUG Helper] Checking if running in Jupyter...")
    try:
        import IPython

        ip = IPython.get_ipython()
        if ip is None:
            print("[DEBUG Helper] is_jupyter_notebook: No IPython instance found.")
            return False
        class_name = str(ip.__class__)
        print(f"[DEBUG Helper] is_jupyter_notebook: IPython class name: {class_name}")
        if "ZMQInteractiveShell" in class_name:
            print(
                "[DEBUG Helper] is_jupyter_notebook: Jupyter detected (ZMQInteractiveShell)."
            )
            return True
        print(
            "[DEBUG Helper] is_jupyter_notebook: Not Jupyter (IPython instance found, but not ZMQInteractiveShell)."
        )
        return False
    except Exception as e:
        print(f"[DEBUG Helper] is_jupyter_notebook: Exception occurred: {e}")
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
    """
    try:
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
):
    def decorator(
        func: Callable[[Any], Any],
    ) -> Processor:
        # Moved init print here
        print(
            f"[DEBUG Decorator Init] @processor decorating function '{func.__name__}'"
        )
        all_env = env or []
        processor_name = func.__name__

        # --- Determine Environment and Get Notebook Code ---
        print("[DEBUG Decorator] Determining execution environment...")
        in_jupyter = is_jupyter_notebook()
        notebook_code = None
        if in_jupyter:
            print("[DEBUG Decorator] Jupyter environment detected.")
            notebook_code = get_notebook_executed_code()
            if not notebook_code:
                print(
                    "[DEBUG Decorator] Warning: Failed to get Jupyter execution history. Will attempt dill."
                )
            else:
                print(
                    f"[DEBUG Decorator] Retrieved notebook history (length: {len(notebook_code)})."
                )
        else:
            print("[DEBUG Decorator] Non-Jupyter environment detected.")
        # --- End Environment Determination ---

        # --- Process Manually Included Objects ---
        included_sources: Dict[Any, Any] = {}
        if include:
            print(f"[DEBUG Decorator] Processing manually included objects: {include}")
            for i, obj in enumerate(include):
                obj_name_str = getattr(obj, "__name__", str(obj))
                print(
                    f"[DEBUG Decorator] Getting source for manually included object: {obj_name_str}"
                )
                obj_source = get_model_source(obj, notebook_code)
                if obj_source:
                    included_sources[obj] = obj_source
                    env_key = f"INCLUDED_OBJECT_{i}_SOURCE"
                    all_env.append(V1EnvVar(key=env_key, value=obj_source))
                    print(
                        f"[DEBUG Decorator] Added source to env for included obj: {obj_name_str}"
                    )
                else:
                    print(
                        f"Warning: Could not retrieve source for manually included object: {obj_name_str}"
                    )
            print(
                f"[DEBUG Decorator] Finished processing included objects. Sources found: {len(included_sources)}"
            )
        else:
            print("[DEBUG Decorator] No manually included objects specified.")
        # --- End Manually Included Objects ---

        # --- Validate Function Signature and Types ---
        print(
            f"[DEBUG Decorator] Validating signature and type hints for {processor_name}..."
        )
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if len(params) != 1:
            raise TypeError(f"{processor_name} must take one parameter")

        try:
            type_hints = get_type_hints(func, globalns=func.__globals__, localns=None)
            print(f"[DEBUG Decorator] Raw type hints: {type_hints}")
        except Exception as e:
            print(f"[DEBUG Decorator] Error getting type hints: {e}")
            raise TypeError(
                f"Could not evaluate type hints for {processor_name}: {e}"
            ) from e

        param_name = params[0].name
        if param_name not in type_hints:
            raise TypeError(
                f"{processor_name} parameter '{param_name}' must have type hint"
            )
        param_type = type_hints[param_name]
        param_type_str_repr = str(param_type)  # Use string for regex
        print(
            f"[DEBUG Decorator] Parameter '{param_name}' type hint: {param_type_str_repr}"
        )

        if "return" not in type_hints:
            raise TypeError(f"{processor_name} must have return type hint")
        return_type = type_hints["return"]
        print(f"[DEBUG Decorator] Return type hint: {return_type}")

        # --- Determine Input Type (StreamMessage, ContentType) ---
        print(
            f"[DEBUG Decorator] Determining input type structure for param type hint: {param_type_str_repr}"
        )
        origin = get_origin(param_type)
        args = get_args(param_type)
        print(f"[DEBUG Decorator] get_origin result: {origin}, get_args result: {args}")
        is_stream_message = False
        content_type = None

        # Check 1: Standard introspection
        if origin is V1StreamMessage or (
            origin is not None
            and origin.__name__ == V1StreamMessage.__name__
            and origin.__module__ == V1StreamMessage.__module__
        ):
            print(
                "[DEBUG Decorator] Input type identified as V1StreamMessage via get_origin."
            )
            is_stream_message = True
            if args:
                content_type = args[0]
                print(
                    f"[DEBUG Decorator] Content type extracted via get_args: {content_type}"
                )
            else:
                print(
                    "[DEBUG Decorator] V1StreamMessage detected, but no generic arguments found via get_args."
                )
        # Check 2: Direct type check
        elif isinstance(param_type, type) and param_type is V1StreamMessage:
            print(
                "[DEBUG Decorator] Input type identified as direct V1StreamMessage type."
            )
            is_stream_message = True
        # Check 3: Regex fallback on string representation
        elif origin is None:
            print(
                f"[DEBUG Decorator] get_origin failed. Attempting regex fallback on type string: '{param_type_str_repr}'"
            )
            # Use param_type_str_repr in match calls
            generic_match = re.match(
                r"^<class '(?:[a-zA-Z0-9_.]+\.)?V1StreamMessage\[(.+?)\]'>$",
                param_type_str_repr,
            )
            if generic_match:
                print(
                    "[DEBUG Decorator] Regex matched generic V1StreamMessage pattern!"
                )
                is_stream_message = True
                content_type_name_str = generic_match.group(1).strip()
                print(
                    f"[DEBUG Decorator] Captured content type name via regex: '{content_type_name_str}'"
                )
                try:
                    resolved_type = eval(content_type_name_str, func.__globals__)
                    content_type = resolved_type
                    print(
                        f"[DEBUG Decorator] Successfully resolved content type name '{content_type_name_str}' to type: {content_type}"
                    )
                except NameError:
                    print(
                        f"[DEBUG Decorator] Warning: Regex found content type name '{content_type_name_str}', but it's not defined in function's globals. Consumer might fail."
                    )
                    content_type = None
                except Exception as e:
                    print(
                        f"[DEBUG Decorator] Warning: Error evaluating content type name '{content_type_name_str}': {e}"
                    )
                    content_type = None
            else:
                # Use param_type_str_repr in match calls
                simple_match = re.match(
                    r"^<class '(?:[a-zA-Z0-9_.]+\.)?V1StreamMessage'>$",
                    param_type_str_repr,
                )
                if simple_match:
                    print(
                        "[DEBUG Decorator] Regex identified direct V1StreamMessage (no generic) from string."
                    )
                    is_stream_message = True
                else:
                    print(
                        f"[DEBUG Decorator] Regex did not match V1StreamMessage pattern for string '{param_type_str_repr}'. Assuming not StreamMessage."
                    )
        else:
            print(
                f"[DEBUG Decorator] Input parameter '{param_name}' type ({param_type_str_repr}) identified as non-StreamMessage type (origin was not None and not V1StreamMessage)."
            )

        print(
            f"[DEBUG Decorator] Final Input Type Determination: is_stream_message={is_stream_message}, content_type={content_type}"
        )
        # --- End Input Type Determination ---

        # --- Validate Types are BaseModel ---
        print(
            "[DEBUG Decorator] Validating parameter and return types are BaseModel subclasses..."
        )

        def check_basemodel(type_to_check, desc):
            print(
                f"[DEBUG Decorator] check_basemodel: Checking {desc} - Type: {type_to_check}"
            )
            if not type_to_check:
                print(
                    f"[DEBUG Decorator] check_basemodel: Skipping check for {desc} (type is None/empty)."
                )
                return
            actual_type = get_origin(type_to_check) or type_to_check
            print(
                f"[DEBUG Decorator] check_basemodel: Actual type for {desc}: {actual_type}"
            )
            if isinstance(actual_type, type) and not issubclass(actual_type, BaseModel):
                print(
                    f"[DEBUG Decorator] check_basemodel: Error - {desc} effective type ({actual_type.__name__}) is not a BaseModel subclass."
                )
                raise TypeError(
                    f"{desc} effective type ({actual_type.__name__}) must be BaseModel subclass"
                )
            elif not isinstance(actual_type, type):
                print(
                    f"[DEBUG Decorator] check_basemodel: Warning - {desc} effective type '{actual_type}' is not a class. Cannot verify BaseModel subclass."
                )
            else:
                print(
                    f"[DEBUG Decorator] check_basemodel: OK - {desc} effective type ({actual_type.__name__}) is a BaseModel subclass."
                )

        effective_param_type = (
            content_type
            if is_stream_message and content_type
            else param_type
            if not is_stream_message
            else None
        )
        check_basemodel(effective_param_type, f"Parameter '{param_name}'")
        check_basemodel(return_type, "Return value")
        print("[DEBUG Decorator] Type validation complete.")
        # --- End Type Validation ---

        # --- Get Function Source ---
        print(
            f"[DEBUG Decorator] Getting source code for function '{processor_name}'..."
        )
        function_source = None
        explicit_source = getattr(func, _NEBU_EXPLICIT_SOURCE_ATTR, None)

        if explicit_source:
            print(
                f"[DEBUG Decorator] Using explicit source (@include) for function {processor_name}"
            )
            function_source = explicit_source
        elif in_jupyter and notebook_code:
            print(
                f"[DEBUG Decorator] Attempting notebook history extraction for function '{processor_name}'..."
            )
            function_source = extract_definition_source_from_string(
                notebook_code, processor_name, ast.FunctionDef
            )
            if function_source:
                print(
                    f"[DEBUG Decorator] Found function '{processor_name}' source in notebook history."
                )
            else:
                print(
                    f"[DEBUG Decorator] Failed to find function '{processor_name}' in notebook history, falling back to dill."
                )
        if function_source is None:
            print(
                f"[DEBUG Decorator] Using dill fallback for function '{processor_name}'..."
            )
            try:
                raw_function_source = dill.source.getsource(func)
                function_source = textwrap.dedent(raw_function_source)
                print(
                    f"[DEBUG Decorator] Successfully got source via dill for '{processor_name}'."
                )
            except (IOError, TypeError, OSError) as e:
                print(
                    f"[DEBUG Decorator] Dill fallback failed for '{processor_name}': {e}"
                )
                if not (in_jupyter and notebook_code):
                    raise ValueError(
                        f"Could not retrieve source for '{processor_name}' using dill: {e}"
                    ) from e

        if function_source is None:  # Final check after all attempts
            raise ValueError(
                f"Failed to obtain source code for function '{processor_name}' using any method."
            )

        print(f"[DEBUG Decorator] Final function source obtained for '{processor_name}' (len: {len(function_source)}). Source starts:\n-------\
{function_source[:250]}...\n-------")
        # --- End Function Source ---

        # --- Get Model Sources ---
        print("[DEBUG Decorator] Getting model sources...")
        input_model_source = None
        output_model_source = None
        content_type_source = None
        print("[DEBUG Decorator] Getting base V1StreamMessage source...")
        stream_message_source = get_type_source(V1StreamMessage, notebook_code)

        if is_stream_message:
            print(
                f"[DEBUG Decorator] Input is StreamMessage. Content type: {content_type}"
            )
            if content_type:
                print(
                    f"[DEBUG Decorator] Getting source for content_type: {content_type}"
                )
                content_type_source = get_type_source(content_type, notebook_code)
                if content_type_source is None:
                    print(
                        f"Warning: Failed to get source for content_type: {content_type}"
                    )
        else:  # Not a stream message
            print(
                f"[DEBUG Decorator] Input is not StreamMessage. Getting source for param_type: {param_type}"
            )
            input_model_source = get_type_source(param_type, notebook_code)
            if input_model_source is None:
                print(
                    f"Warning: Failed to get source for input param_type: {param_type}"
                )

        print(f"[DEBUG Decorator] Getting source for return_type: {return_type}")
        output_model_source = get_type_source(return_type, notebook_code)
        if output_model_source is None:
            print(f"Warning: Failed to get source for return_type: {return_type}")

        print(
            f"[DEBUG Decorator] Source Result - Content Type: {'Found' if content_type_source else 'Not Found or N/A'}"
        )
        print(
            f"[DEBUG Decorator] Source Result - Input Model (non-stream): {'Found' if input_model_source else 'Not Found or N/A'}"
        )
        print(
            f"[DEBUG Decorator] Source Result - Output Model: {'Found' if output_model_source else 'Not Found'}"
        )
        print(
            f"[DEBUG Decorator] Source Result - Base StreamMessage: {'Found' if stream_message_source else 'Not Found'}"
        )
        # --- End Model Sources ---

        # --- Populate Environment Variables ---
        print("[DEBUG Decorator] Populating environment variables...")
        all_env.append(V1EnvVar(key="FUNCTION_SOURCE", value=function_source))
        all_env.append(V1EnvVar(key="FUNCTION_NAME", value=processor_name))

        def add_source_to_env(key_base: str, source: Any):
            print(f"[DEBUG Decorator] add_source_to_env: Processing key '{key_base}'")
            if not source:
                print(
                    f"[DEBUG Decorator] add_source_to_env: No source for '{key_base}', skipping."
                )
                return

            if isinstance(source, tuple):
                origin_src, arg_srcs = source
                print(
                    f"[DEBUG Decorator] add_source_to_env: '{key_base}' is tuple source. Origin found: {bool(origin_src)}, Num args: {len(arg_srcs)}"
                )
                if origin_src and isinstance(origin_src, str):
                    all_env.append(V1EnvVar(key=f"{key_base}_SOURCE", value=origin_src))
                    print(f"[DEBUG Decorator] Added env var {key_base}_SOURCE (origin)")
                for i, arg_src in enumerate(arg_srcs):
                    if isinstance(arg_src, str):
                        all_env.append(
                            V1EnvVar(key=f"{key_base}_ARG_{i}_SOURCE", value=arg_src)
                        )
                        print(
                            f"[DEBUG Decorator] Added env var {key_base}_ARG_{i}_SOURCE"
                        )
                    elif isinstance(arg_src, tuple):
                        arg_origin_src, _ = arg_src
                        if arg_origin_src and isinstance(arg_origin_src, str):
                            all_env.append(
                                V1EnvVar(
                                    key=f"{key_base}_ARG_{i}_SOURCE",
                                    value=arg_origin_src,
                                )
                            )
                            print(
                                f"[DEBUG Decorator] Added env var {key_base}_ARG_{i}_SOURCE (nested origin)"
                            )
                        else:
                            print(
                                f"[DEBUG Decorator] Skipping complex/non-string nested arg origin for {key_base}_ARG_{i}"
                            )
                    else:
                        print(
                            f"[DEBUG Decorator] Skipping complex/non-string arg source for {key_base}_ARG_{i}"
                        )
            elif isinstance(source, str):
                all_env.append(V1EnvVar(key=f"{key_base}_SOURCE", value=source))
                print(f"[DEBUG Decorator] Added env var {key_base}_SOURCE (string)")
            else:
                print(
                    f"[DEBUG Decorator] Warning: Unknown source type for {key_base}: {type(source)}. Skipping."
                )

        add_source_to_env("INPUT_MODEL", input_model_source)
        add_source_to_env("OUTPUT_MODEL", output_model_source)
        add_source_to_env("CONTENT_TYPE", content_type_source)
        add_source_to_env("STREAM_MESSAGE", stream_message_source)

        print("[DEBUG Decorator] Adding type info env vars...")
        all_env.append(V1EnvVar(key="PARAM_TYPE_STR", value=param_type_str_repr))
        all_env.append(V1EnvVar(key="RETURN_TYPE_STR", value=str(return_type)))
        all_env.append(V1EnvVar(key="IS_STREAM_MESSAGE", value=str(is_stream_message)))
        if content_type and hasattr(content_type, "__name__"):
            all_env.append(
                V1EnvVar(key="CONTENT_TYPE_NAME", value=content_type.__name__)
            )
        all_env.append(V1EnvVar(key="MODULE_NAME", value=func.__module__))
        print("[DEBUG Decorator] Finished populating environment variables.")
        # --- End Environment Variables ---

        # --- Final Setup ---
        print("[DEBUG Decorator] Preparing final Processor object...")
        metadata = V1ResourceMetaRequest(
            name=processor_name, namespace=namespace, labels=labels
        )
        consumer_command = f"{python_cmd} -u -m nebu.processors.consumer"
        setup_commands = [
            f"{python_cmd} -m pip install dill pydantic redis nebu",  # Base deps
        ]
        if setup_script:
            print("[DEBUG Decorator] Adding setup script to command.")
            setup_commands.append(f"\n{setup_script}\n")
        setup_commands.append(consumer_command)
        final_command = "\n".join(setup_commands)
        print(
            f"[DEBUG Decorator] Final container command:\n-------\n{final_command}\n-------"
        )

        container_request = V1ContainerRequest(
            image=image,
            command=final_command,
            env=all_env,
            volumes=volumes,
            accelerators=accelerators,
            resources=resources,
            meters=meters,
            restart="Always",
            authz=authz,
            platform=platform,
            metadata=metadata,
        )
        print("[DEBUG Decorator] Final Container Request Env Vars (Summary):")
        for env_var in all_env:
            if "SOURCE" in env_var.key:
                print(f"[DEBUG Decorator]  {env_var.key}: <source code present>")
            else:
                print(f"[DEBUG Decorator]  {env_var.key}: {env_var.value}")

        processor_instance = Processor(
            name=processor_name,
            stream=processor_name,
            namespace=namespace,
            labels=labels,
            container=container_request,
            schema_=None,
            common_schema=None,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            scale_config=scale,
            no_delete=no_delete,
        )
        print(
            f"[DEBUG Decorator] Processor instance '{processor_name}' created successfully."
        )
        return processor_instance

    return decorator
