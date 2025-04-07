import inspect
import textwrap
from typing import Any, Callable, Dict, List, Optional, TypeVar, get_type_hints

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
        return None


def get_type_source(type_obj: Any) -> Optional[Any]:
    """Get the source code for a type, including generic parameters."""
    # If it's a class, get its source
    if isinstance(type_obj, type):
        return get_model_source(type_obj)

    # If it's a GenericAlias (like V1StreamMessage[SomeType])
    if hasattr(type_obj, "__origin__") and hasattr(type_obj, "__args__"):
        origin_source = get_model_source(type_obj.__origin__)
        args_sources = []

        # Get sources for all type arguments
        for arg in type_obj.__args__:
            arg_source = get_type_source(arg)
            if arg_source:
                args_sources.append(arg_source)

        return origin_source, args_sources

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

    def decorator(func: Callable[[T], R]) -> Processor:
        # Validate that the function takes a single parameter that is a BaseModel
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if len(params) != 1:
            raise TypeError(f"Function {func.__name__} must take exactly one parameter")

        # Check parameter type
        type_hints = get_type_hints(func)
        param_name = params[0].name
        if param_name not in type_hints:
            raise TypeError(
                f"Parameter {param_name} in function {func.__name__} must have a type annotation"
            )

        param_type = type_hints[param_name]

        # Check if input type is V1StreamMessage or a subclass
        is_stream_message = False
        content_type = None

        # Handle generic V1StreamMessage
        if (
            hasattr(param_type, "__origin__")
            and param_type.__origin__ == V1StreamMessage
        ):
            is_stream_message = True
            # Extract the content type from V1StreamMessage[ContentType]
            if hasattr(param_type, "__args__") and param_type.__args__:
                content_type = param_type.__args__[0]
        # Handle direct V1StreamMessage
        elif param_type is V1StreamMessage:
            is_stream_message = True

        # Ensure the parameter is a BaseModel
        actual_type = (
            param_type.__origin__ if hasattr(param_type, "__origin__") else param_type  # type: ignore
        )
        if not issubclass(actual_type, BaseModel):
            raise TypeError(
                f"Parameter {param_name} in function {func.__name__} must be a BaseModel"
            )

        # Check return type
        if "return" not in type_hints:
            raise TypeError(
                f"Function {func.__name__} must have a return type annotation"
            )

        return_type = type_hints["return"]
        actual_return_type = (
            return_type.__origin__
            if hasattr(return_type, "__origin__")
            else return_type
        )
        if not issubclass(actual_return_type, BaseModel):
            raise TypeError(
                f"Return value of function {func.__name__} must be a BaseModel"
            )

        # Get function name to use as processor name
        processor_name = func.__name__

        # Prepare environment variables
        all_env = env or []

        # Get the source code of the function
        try:
            raw_function_source = inspect.getsource(func)
            print(
                f"[DEBUG Decorator] Raw source for {func.__name__}:\n{raw_function_source}"
            )

            # Clean up the indentation
            dedented_function_source = textwrap.dedent(raw_function_source)
            print(
                f"[DEBUG Decorator] Dedented source for {func.__name__}:\n{dedented_function_source}"
            )

            # Find the start of the function definition ('def')
            # Skip lines starting with '@' or empty lines until 'def' is found
            lines = dedented_function_source.splitlines()
            func_def_index = -1
            for i, line in enumerate(lines):
                stripped_line = line.strip()
                if stripped_line.startswith("def "):
                    func_def_index = i
                    break
                elif stripped_line.startswith("@") or not stripped_line:
                    # Skip decorator lines and empty lines before 'def'
                    continue
                else:
                    # Found something unexpected before 'def'
                    raise ValueError(
                        f"Unexpected content found before 'def' in source for {func.__name__}. Cannot reliably strip decorators."
                    )

            if func_def_index != -1:
                # Keep lines from the 'def' line onwards
                function_source = "\n".join(
                    lines[func_def_index:]
                )  # Use \n for env var
            else:
                # If 'def' wasn't found (shouldn't happen with valid function source)
                raise ValueError(
                    f"Could not find function definition 'def' in source for {func.__name__}"
                )

            print(
                f"[DEBUG Decorator] Processed function source for {func.__name__}:\n{function_source}"
            )

        except (IOError, TypeError) as e:
            print(f"[DEBUG Decorator] Error getting source for {func.__name__}: {e}")
            raise ValueError(
                f"Could not retrieve source code for function {func.__name__}: {e}"
            ) from e

        # Get source code for the models
        input_model_source = None
        output_model_source = None
        content_type_source = None

        # Get the V1StreamMessage class source
        stream_message_source = get_model_source(V1StreamMessage)

        # Get input model source
        if is_stream_message:
            input_model_source = stream_message_source
            if content_type:
                content_type_source = get_type_source(content_type)
        else:
            input_model_source = get_type_source(param_type)

        # Get output model source
        output_model_source = get_type_source(return_type)

        # Add function source code to environment variables
        print(
            f"[DEBUG Decorator] Setting FUNCTION_SOURCE: {function_source[:100]}..."
        )  # Print first 100 chars
        all_env.append(V1EnvVar(key="FUNCTION_SOURCE", value=function_source))
        print(f"[DEBUG Decorator] Setting FUNCTION_NAME: {func.__name__}")
        all_env.append(V1EnvVar(key="FUNCTION_NAME", value=func.__name__))

        # Add model source codes
        if input_model_source:
            if isinstance(input_model_source, tuple):
                all_env.append(
                    V1EnvVar(key="INPUT_MODEL_SOURCE", value=input_model_source[0])
                )
                # Add generic args sources
                for i, arg_source in enumerate(input_model_source[1]):
                    all_env.append(
                        V1EnvVar(key=f"INPUT_MODEL_ARG_{i}_SOURCE", value=arg_source)
                    )
            else:
                all_env.append(
                    V1EnvVar(key="INPUT_MODEL_SOURCE", value=input_model_source)
                )

        if output_model_source:
            if isinstance(output_model_source, tuple):
                all_env.append(
                    V1EnvVar(key="OUTPUT_MODEL_SOURCE", value=output_model_source[0])
                )
                # Add generic args sources
                for i, arg_source in enumerate(output_model_source[1]):
                    all_env.append(
                        V1EnvVar(key=f"OUTPUT_MODEL_ARG_{i}_SOURCE", value=arg_source)
                    )
            else:
                all_env.append(
                    V1EnvVar(key="OUTPUT_MODEL_SOURCE", value=output_model_source)
                )

        if stream_message_source:
            all_env.append(
                V1EnvVar(key="STREAM_MESSAGE_SOURCE", value=stream_message_source)
            )

        if content_type_source:
            if isinstance(content_type_source, tuple):
                all_env.append(
                    V1EnvVar(key="CONTENT_TYPE_SOURCE", value=content_type_source[0])
                )
                # Add generic args sources for content type
                for i, arg_source in enumerate(content_type_source[1]):
                    all_env.append(
                        V1EnvVar(key=f"CONTENT_TYPE_ARG_{i}_SOURCE", value=arg_source)
                    )
            else:
                all_env.append(
                    V1EnvVar(key="CONTENT_TYPE_SOURCE", value=content_type_source)
                )

        # Add included object sources
        if include:
            for i, obj in enumerate(include):
                obj_source = get_type_source(
                    obj
                )  # Reuse existing function for source retrieval
                if obj_source:
                    if isinstance(obj_source, tuple):
                        # Handle complex types (like generics) if needed, similar to models
                        all_env.append(
                            V1EnvVar(
                                key=f"INCLUDED_OBJECT_{i}_SOURCE", value=obj_source[0]
                            )
                        )
                        for j, arg_source in enumerate(obj_source[1]):
                            all_env.append(
                                V1EnvVar(
                                    key=f"INCLUDED_OBJECT_{i}_ARG_{j}_SOURCE",
                                    value=arg_source,
                                )
                            )
                    else:
                        all_env.append(
                            V1EnvVar(
                                key=f"INCLUDED_OBJECT_{i}_SOURCE", value=obj_source
                            )
                        )
                else:
                    # Optionally raise an error or log a warning if source can't be found
                    print(
                        f"Warning: Could not retrieve source for included object: {obj}"
                    )

        # Add parameter and return type info for runtime validation
        all_env.append(
            V1EnvVar(
                key="PARAM_TYPE_NAME",
                value=param_type.__name__
                if hasattr(param_type, "__name__")
                else str(param_type),
            )
        )
        all_env.append(
            V1EnvVar(
                key="RETURN_TYPE_NAME",
                value=return_type.__name__
                if hasattr(return_type, "__name__")
                else str(return_type),
            )
        )
        all_env.append(V1EnvVar(key="IS_STREAM_MESSAGE", value=str(is_stream_message)))

        if content_type:
            all_env.append(
                V1EnvVar(
                    key="CONTENT_TYPE_NAME",
                    value=content_type.__name__
                    if hasattr(content_type, "__name__")
                    else str(content_type),
                )
            )

        # We still add the module for reference, but we won't rely on importing it
        all_env.append(V1EnvVar(key="MODULE_NAME", value=func.__module__))

        # Prepare metadata
        metadata = V1ResourceMetaRequest(
            name=processor_name, namespace=namespace, labels=labels
        )

        # Create the command to run the consumer directly
        consumer_command = f"{python_cmd} -m nebu.processors.consumer"

        final_command = f"{python_cmd} -m pip install redis nebu\n\n{setup_script}\n\n{consumer_command}"

        # Create the V1ContainerRequest
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
        print("container_request", container_request)

        # Create the processor instance
        processor_instance = Processor(
            name=processor_name,
            stream=processor_name,
            namespace=namespace,
            labels=labels,
            container=container_request,
            schema_=None,  # TODO
            common_schema=None,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            scale_config=scale,
            no_delete=no_delete,
        )

        return processor_instance

    return decorator
