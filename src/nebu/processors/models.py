from typing import Any, Optional

from pydantic import BaseModel, Field

from nebu.containers.models import V1Container
from nebu.meta import V1ResourceMeta, V1ResourceMetaRequest

# If these are in another module, import them as:
# from .containers import V1Container, V1ResourceMeta, V1ResourceMetaRequest
# For demonstration, simply assume they're available in scope:
# class V1Container(BaseModel): ...
# class V1ResourceMeta(BaseModel): ...
# class V1ResourceMetaRequest(BaseModel): ...


class V1ProcessorStatus(BaseModel):
    status: Optional[str] = None
    message: Optional[str] = None
    pressure: Optional[int] = None


class V1ScaleUp(BaseModel):
    above_pressure: Optional[int] = None
    duration: Optional[str] = None


class V1ScaleDown(BaseModel):
    below_pressure: Optional[int] = None
    duration: Optional[str] = None


class V1ScaleZero(BaseModel):
    duration: Optional[str] = None


class V1Scale(BaseModel):
    up: Optional[V1ScaleUp] = None
    down: Optional[V1ScaleDown] = None
    zero: Optional[V1ScaleZero] = None


DEFAULT_PROCESSOR_KIND = "Processor"


class V1Processor(BaseModel):
    kind: str = Field(default=DEFAULT_PROCESSOR_KIND)
    metadata: V1ResourceMeta
    container: Optional["V1Container"] = None
    stream: Optional[str] = None
    schema_: Optional[Any] = None  # Or Dict[str, Any], if you know the schema format
    common_schema: Optional[str] = None
    min_replicas: Optional[int] = None
    max_replicas: Optional[int] = None
    scale: Optional[V1Scale] = None
    status: Optional[V1ProcessorStatus] = None


class V1ProcessorRequest(BaseModel):
    kind: str = Field(default=DEFAULT_PROCESSOR_KIND)
    metadata: V1ResourceMetaRequest
    container: Optional["V1Container"] = None
    stream: Optional[str] = None
    schema_: Optional[Any] = None
    common_schema: Optional[str] = None
    min_replicas: Optional[int] = None
    max_replicas: Optional[int] = None
    scale: Optional[V1Scale] = None
