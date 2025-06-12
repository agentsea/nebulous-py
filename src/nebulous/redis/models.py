from typing import Optional

from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from openai.types.responses import Response
from pydantic import BaseModel


class V1StreamMessage(BaseModel):
    kind: str = "V1StreamMessage"
    id: str
    content: dict  # type: ignore
    created_at: int
    return_stream: Optional[str] = None
    user_id: Optional[str] = None
    organizations: Optional[dict] = None  # type: ignore
    handle: Optional[str] = None
    adapter: Optional[str] = None


class V1StreamResponseMessage(BaseModel):
    kind: str = "V1StreamResponseMessage"
    id: str
    content: dict  # type: ignore
    created_at: int
    user_id: Optional[str] = None


class V1OpenAIStreamMessage(BaseModel):
    kind: str = "V1OpenAIStreamMessage"
    id: str
    content: CompletionCreateParamsBase
    created_at: int
    return_stream: Optional[str] = None
    user_id: Optional[str] = None
    organizations: Optional[dict] = None  # type: ignore
    handle: Optional[str] = None
    adapter: Optional[str] = None


class V1OpenAIStreamResponse(BaseModel):
    kind: str = "V1OpenAIStreamResponse"
    id: str
    content: Response
    created_at: int
    user_id: Optional[str] = None
