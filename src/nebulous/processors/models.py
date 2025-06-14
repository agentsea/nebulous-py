from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel

# Assuming these are imported from other modules
from nebulous.containers.models import V1ContainerRequest
from nebulous.meta import V1ResourceMeta, V1ResourceMetaRequest, V1ResourceReference

# Type variable for content that must be a BaseModel
T = TypeVar("T", bound=BaseModel)


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


class V1Processor(BaseModel):
    kind: str = "Processor"
    metadata: V1ResourceMeta
    container: Optional[V1ContainerRequest] = None
    stream: str
    schema_: Optional[Any] = None
    common_schema: Optional[str] = None
    min_replicas: Optional[int] = None
    max_replicas: Optional[int] = None
    scale: Optional[V1Scale] = None
    status: Optional[V1ProcessorStatus] = None

    def to_resource_reference(self) -> V1ResourceReference:
        return V1ResourceReference(
            kind=self.kind,
            name=self.metadata.name,
            namespace=self.metadata.namespace,
        )


class V1ProcessorRequest(BaseModel):
    kind: str = "Processor"
    metadata: V1ResourceMetaRequest
    container: Optional[V1ContainerRequest] = None
    schema_: Optional[Any] = None
    common_schema: Optional[str] = None
    min_replicas: Optional[int] = None
    max_replicas: Optional[int] = None
    scale: Optional[V1Scale] = None


class V1Processors(BaseModel):
    processors: List[V1Processor] = []


class V1ProcessorScaleRequest(BaseModel):
    replicas: Optional[int] = None
    min_replicas: Optional[int] = None


class V1UpdateProcessor(BaseModel):
    kind: Optional[str] = None
    metadata: Optional[V1ResourceMetaRequest] = None
    container: Optional[V1ContainerRequest] = None
    stream: Optional[str] = None
    min_replicas: Optional[int] = None
    max_replicas: Optional[int] = None
    scale: Optional[V1Scale] = None
    schema_: Optional[Any] = None
    common_schema: Optional[str] = None
    no_delete: Optional[bool] = None


class V1StreamData(BaseModel):
    content: Any = None
    wait: Optional[bool] = None
    user_key: Optional[str] = None


class Message(BaseModel, Generic[T]):
    kind: str = "Message"
    id: str
    content: Optional[T] = None
    created_at: int
    return_stream: Optional[str] = None
    user_id: Optional[str] = None
    orgs: Optional[Any] = None
    handle: Optional[str] = None
    adapter: Optional[str] = None
    api_key: Optional[str] = None


class V1StreamMessage(BaseModel, Generic[T]):
    kind: str = "StreamMessage"
    id: str
    content: Optional[T] = None
    created_at: int
    return_stream: Optional[str] = None
    user_id: Optional[str] = None
    orgs: Optional[Any] = None
    handle: Optional[str] = None
    adapter: Optional[str] = None
    api_key: Optional[str] = None


class V1StreamResponseMessage(BaseModel):
    kind: str = "StreamResponseMessage"
    id: str
    content: Any = None
    status: Optional[str] = None
    created_at: int
    user_id: Optional[str] = None


class V1OpenAIStreamMessage(BaseModel):
    kind: str = "OpenAIStreamMessage"
    id: str
    content: Any  # Using Any for ChatCompletionRequest
    created_at: int
    return_stream: Optional[str] = None
    user_id: Optional[str] = None
    orgs: Optional[Any] = None
    handle: Optional[str] = None
    adapter: Optional[str] = None
    api_key: Optional[str] = None


class V1OpenAIStreamResponse(BaseModel):
    kind: str = "OpenAIStreamResponse"
    id: str
    content: Any  # Using Any for ChatCompletionResponse
    created_at: int
    user_id: Optional[str] = None


class V1ProcessorHealthResponse(BaseModel):
    status: str
    message: Optional[str] = None
    details: Optional[Any] = None
