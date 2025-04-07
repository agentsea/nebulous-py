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


def get_model_source(model_class: Any) -> Optional[str]:
    """Get the source code of a model class."""
    try:
        source = inspect.getsource(model_class)
        return textwrap.dedent(source)
    except (IOError, TypeError):
        print(f"[DEBUG get_model_source] Failed for: {model_class}")  # Added debug
        return None


def get_type_source(type_obj: Any) -> Optional[Any]:
    """Get the source code for a type, including generic parameters."""
    # If it's a class, get its source
    if isinstance(type_obj, type):
        return get_model_source(type_obj)

    # If it's a GenericAlias (like V1StreamMessage[SomeType])
    # Use get_origin and get_args for robustness
    origin = get_origin(type_obj)
    args = get_args(type_obj)

    if origin is not None:
        origin_source = get_model_source(origin)
        args_sources = []

        # Get sources for all type arguments
        for arg in args:
            arg_source = get_type_source(arg)
            if arg_source:
                args_sources.append(arg_source)

        # Return tuple only if origin source and some arg sources were found
        if origin_source or args_sources:
            return (
                origin_source,
                args_sources,
            )  # Return even if origin_source is None if args_sources exist

    return None  # Fallback if not a class or recognizable generic alias


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
    """
    Decorator that converts a function into a Processor.

    Args:
        image: The container image to use for the processor
        setup_script: Optional setup script to run before starting the processor
        scale: Optional scaling configuration
        min_replicas: Minimum number of replicas to maintain
        max_replicas: Maximum number of replicas to scale to
        platform: Optional compute platform to run on
        accelerators: Optional list of accelerator types
        namespace: Optional namespace for the processor
        labels: Optional labels to apply to the processor
        env: Optional environment variables
        volumes: Optional volume mounts
        resources: Optional resource requirements
        meters: Optional metering configuration
        authz: Optional authorization configuration
        python_cmd: Optional python command to use
        no_delete: Whether to prevent deleting the processor on updates
        include: Optional list of Python objects whose source code should be included
    """

    def decorator(
        func: Callable[[Any], Any],
    ) -> Processor:  # Changed T/R to Any for broader compatibility
        # Prepare environment variables early
        all_env = env or []

        # --- Process Included Objects First ---
        included_sources: Dict[Any, Any] = {}  # Store source keyed by the object itself
        if include:
            print(f"[DEBUG Decorator] Processing included objects: {include}")
            for i, obj in enumerate(include):
                # Directly use get_model_source as include expects types/classes usually
                obj_source = get_model_source(obj)
                if obj_source:
                    print(f"[DEBUG Decorator] Found source for included object: {obj}")
                    included_sources[obj] = obj_source  # Store source by object
                    # Add to env vars immediately (simplifies later logic)
                    env_key = f"INCLUDED_OBJECT_{i}_SOURCE"
                    all_env.append(V1EnvVar(key=env_key, value=obj_source))
                    print(f"[DEBUG Decorator] Added {env_key} for {obj}")

                else:
                    # Optionally raise an error or log a warning if source can't be found
                    print(
                        f"Warning: Could not retrieve source via get_model_source for included object: {obj}. Decorator might fail if this type is needed but cannot be auto-detected."
                    )
            print(
                f"[DEBUG Decorator] Finished processing included objects. Sources found: {len(included_sources)}"
            )
        # --- End Included Objects Processing ---

        # Validate function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if len(params) != 1:
            raise TypeError(f"Function {func.__name__} must take exactly one parameter")

        # Check parameter type hint
        try:
            # Use eval_str=True for forward references if needed, requires Python 3.10+ globals/locals
            type_hints = get_type_hints(
                func, globalns=func.__globals__, localns=None
            )  # Pass globals
        except Exception as e:
            print(
                f"[DEBUG Decorator] Error getting type hints for {func.__name__}: {e}"
            )
            raise TypeError(
                f"Could not evaluate type hints for {func.__name__}. Ensure all types are defined or imported."
            ) from e

        param_name = params[0].name
        if param_name not in type_hints:
            raise TypeError(
                f"Parameter {param_name} in function {func.__name__} must have a type annotation"
            )
        param_type = type_hints[param_name]

        # --- Determine Input Type, Content Type, and is_stream_message ---
        print(f"[DEBUG Decorator] Full type_hints: {type_hints}")
        print(f"[DEBUG Decorator] Detected param_type: {param_type}")
        origin = get_origin(param_type)
        args = get_args(param_type)
        print(f"[DEBUG Decorator] Param type origin (using get_origin): {origin}")
        print(f"[DEBUG Decorator] Param type args (using get_args): {args}")
        if origin:
            print(
                f"[DEBUG Decorator] Origin name: {getattr(origin, '__name__', 'N/A')}, module: {getattr(origin, '__module__', 'N/A')}"
            )
            print(
                f"[DEBUG Decorator] V1StreamMessage name: {V1StreamMessage.__name__}, module: {V1StreamMessage.__module__}"
            )

        is_stream_message = False
        content_type = None

        # Check 1: Standard check using get_origin
        if (
            origin is not None
            and origin.__name__ == V1StreamMessage.__name__
            and origin.__module__ == V1StreamMessage.__module__
        ):
            is_stream_message = True
            print("[DEBUG Decorator] V1StreamMessage detected via origin check.")
            if args:
                content_type = args[0]

        # Check 2: Fallback check using string representation
        elif origin is None:
            type_str = str(param_type)
            match = re.match(
                r"<class 'nebu\.processors\.models\.V1StreamMessage\[(.*)\]\'>",
                type_str,
            )
            if match:
                print(
                    "[DEBUG Decorator] V1StreamMessage detected via string regex check (origin/args failed)."
                )
                content_type_name = match.group(1)
                print(
                    f"[DEBUG Decorator] Manually parsed content_type name: {content_type_name}"
                )
                # Attempt to find the type
                resolved_type = None
                func_globals = func.__globals__
                if content_type_name in func_globals:
                    resolved_type = func_globals[content_type_name]
                    print(
                        f"[DEBUG Decorator] Found content type '{content_type_name}' in function globals."
                    )
                else:
                    func_module = inspect.getmodule(func)
                    if func_module and hasattr(func_module, content_type_name):
                        resolved_type = getattr(func_module, content_type_name)
                        print(
                            f"[DEBUG Decorator] Found content type '{content_type_name}' in function module."
                        )

                if resolved_type:
                    content_type = resolved_type
                    is_stream_message = True  # Set flag *only if* resolved
                else:
                    print(
                        f"[DEBUG Decorator] Fallback failed: Could not find type '{content_type_name}' in globals or module. Use 'include'."
                    )
            # else: Fallback regex did not match

        # Check 3: Handle direct V1StreamMessage
        elif param_type is V1StreamMessage:
            print("[DEBUG Decorator] V1StreamMessage detected via direct type check.")
            is_stream_message = True
            # content_type remains None

        print(f"[DEBUG Decorator] Final is_stream_message: {is_stream_message}")
        print(f"[DEBUG Decorator] Final content_type: {content_type}")
        # --- End Input Type Determination ---

        # --- Validate Parameter Type is BaseModel ---
        type_to_check_for_basemodel = None
        if is_stream_message:
            if content_type:
                type_to_check_for_basemodel = content_type
            # else: Base V1StreamMessage itself is a BaseModel, no need to check further
        else:
            type_to_check_for_basemodel = param_type

        if type_to_check_for_basemodel:
            actual_type_to_check = (
                get_origin(type_to_check_for_basemodel) or type_to_check_for_basemodel
            )
            if not issubclass(actual_type_to_check, BaseModel):
                raise TypeError(
                    f"Parameter '{param_name}' effective type ({actual_type_to_check.__name__}) in function '{func.__name__}' must be a BaseModel subclass"
                )
        # --- End Parameter Validation ---

        # --- Validate Return Type ---
        if "return" not in type_hints:
            raise TypeError(
                f"Function {func.__name__} must have a return type annotation"
            )
        return_type = type_hints["return"]
        actual_return_type = get_origin(return_type) or return_type
        if not issubclass(actual_return_type, BaseModel):
            raise TypeError(
                f"Return value of function {func.__name__} must be a BaseModel subclass"
            )
        # --- End Return Type Validation ---

        # --- Get Function Source ---
        processor_name = func.__name__
        try:
            raw_function_source = inspect.getsource(func)
            # ... (rest of source processing remains the same) ...
            lines = raw_function_source.splitlines()
            func_def_index = -1
            decorator_lines = 0
            in_decorator = False
            for i, line in enumerate(lines):
                stripped_line = line.strip()
                if stripped_line.startswith("@"):
                    in_decorator = True
                    decorator_lines += 1
                    continue  # Skip decorator line
                if in_decorator and stripped_line.endswith(
                    ")"
                ):  # Simple check for end of decorator args
                    in_decorator = False
                    decorator_lines += 1
                    continue
                if in_decorator:
                    decorator_lines += 1
                    continue  # Skip multi-line decorator args

                if stripped_line.startswith("def "):
                    func_def_index = i
                    break

            if func_def_index != -1:
                # Keep lines from the 'def' line onwards
                function_source = "\n".join(lines[func_def_index:])
            else:
                raise ValueError(
                    f"Could not find function definition 'def' in source for {func.__name__}"
                )

            print(
                f"[DEBUG Decorator] Processed function source for {func.__name__}:\n{function_source[:200]}..."
            )

        except (IOError, TypeError) as e:
            print(f"[DEBUG Decorator] Error getting source for {func.__name__}: {e}")
            raise ValueError(
                f"Could not retrieve source code for function {func.__name__}: {e}"
            ) from e
        # --- End Function Source ---

        # --- Get Model Sources (Prioritizing Included) ---
        input_model_source = None
        output_model_source = None
        content_type_source = None
        stream_message_source = get_model_source(V1StreamMessage)  # Still get this

        # Get content_type source (if applicable)
        if is_stream_message and content_type:
            if content_type in included_sources:
                content_type_source = included_sources[content_type]
                print(
                    f"[DEBUG Decorator] Using included source for content_type: {content_type}"
                )
            else:
                print(
                    f"[DEBUG Decorator] Attempting get_type_source for content_type: {content_type}"
                )
                content_type_source = get_type_source(content_type)
                if content_type_source is None:
                    print(
                        f"[DEBUG Decorator] Warning: get_type_source failed for content_type: {content_type}. Consumer might fail if not included."
                    )

            print(
                f"[DEBUG Decorator] Final content_type_source: {str(content_type_source)[:100]}..."
            )

        # Get input_model source (which is V1StreamMessage if is_stream_message)
        if is_stream_message:
            input_model_source = (
                stream_message_source  # Always use base stream message source
            )
        elif (
            param_type in included_sources
        ):  # Check if non-stream-message input type was included
            input_model_source = included_sources[param_type]
            print(
                f"[DEBUG Decorator] Using included source for param_type: {param_type}"
            )
        else:  # Fallback for non-stream-message, non-included input type
            print(
                f"[DEBUG Decorator] Attempting get_type_source for param_type: {param_type}"
            )
            input_model_source = get_type_source(param_type)
            if input_model_source is None:
                print(
                    f"[DEBUG Decorator] Warning: get_type_source failed for param_type: {param_type}. Consumer might fail if not included."
                )
        print(
            f"[DEBUG Decorator] Final input_model_source: {str(input_model_source)[:100]}..."
        )

        # Get output_model source
        if return_type in included_sources:
            output_model_source = included_sources[return_type]
            print(
                f"[DEBUG Decorator] Using included source for return_type: {return_type}"
            )
        else:
            print(
                f"[DEBUG Decorator] Attempting get_type_source for return_type: {return_type}"
            )
            output_model_source = get_type_source(return_type)
            if output_model_source is None:
                print(
                    f"[DEBUG Decorator] Warning: get_type_source failed for return_type: {return_type}. Consumer might fail if not included."
                )
        print(
            f"[DEBUG Decorator] Final output_model_source: {str(output_model_source)[:100]}..."
        )
        # --- End Model Sources ---

        # --- Populate Environment Variables ---
        print("[DEBUG Decorator] Populating environment variables...")
        all_env.append(V1EnvVar(key="FUNCTION_SOURCE", value=function_source))
        all_env.append(V1EnvVar(key="FUNCTION_NAME", value=func.__name__))

        # Add model source codes (handle tuples from get_type_source if necessary, although unlikely with prioritization)
        def add_source_to_env(key_base: str, source: Any):
            if source:
                if isinstance(source, tuple):
                    # This path is less likely now with include prioritization
                    if source[0]:  # Origin source
                        all_env.append(
                            V1EnvVar(key=f"{key_base}_SOURCE", value=source[0])
                        )
                    for i, arg_source in enumerate(source[1]):  # Arg sources
                        all_env.append(
                            V1EnvVar(key=f"{key_base}_ARG_{i}_SOURCE", value=arg_source)
                        )
                else:  # Simple string source
                    all_env.append(V1EnvVar(key=f"{key_base}_SOURCE", value=source))

        add_source_to_env("INPUT_MODEL", input_model_source)
        add_source_to_env("OUTPUT_MODEL", output_model_source)
        add_source_to_env("CONTENT_TYPE", content_type_source)
        add_source_to_env(
            "STREAM_MESSAGE", stream_message_source
        )  # Add base stream message source

        # Type names for consumer validation/parsing
        all_env.append(
            V1EnvVar(
                key="PARAM_TYPE_STR", value=str(param_type)
            )  # Send string representation
        )
        all_env.append(
            V1EnvVar(
                key="RETURN_TYPE_STR", value=str(return_type)
            )  # Send string representation
        )
        all_env.append(V1EnvVar(key="IS_STREAM_MESSAGE", value=str(is_stream_message)))
        if content_type:
            all_env.append(
                V1EnvVar(key="CONTENT_TYPE_NAME", value=content_type.__name__)
            )

        all_env.append(V1EnvVar(key="MODULE_NAME", value=func.__module__))
        # --- End Environment Variables ---

        # --- Final Setup ---
        metadata = V1ResourceMetaRequest(
            name=processor_name, namespace=namespace, labels=labels
        )
        consumer_command = f"{python_cmd} -m nebu.processors.consumer"
        final_command = f"{python_cmd} -m pip install redis nebu\n\n{setup_script or ''}\n\n{consumer_command}"

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
        print("[DEBUG Decorator] Final Container Request Env Vars:")
        for env_var in all_env:
            print(
                f"[DEBUG Decorator]  {env_var.key}: {str(env_var.value)[:70]}..."
            )  # Print key and start of value

        processor_instance = Processor(
            name=processor_name,
            stream=processor_name,  # Default stream name to processor name
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

        return processor_instance

    return decorator
