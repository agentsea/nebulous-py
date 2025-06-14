from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from nebulous.meta import V1ResourceMeta


class V1NamespaceMetaRequest(BaseModel):
    name: str
    labels: Optional[Dict[str, str]] = None
    owner: Optional[str] = None


class V1NamespaceRequest(BaseModel):
    metadata: V1NamespaceMetaRequest


class V1Namespace(BaseModel):
    kind: str = Field(default="Namespace")
    metadata: V1ResourceMeta


class V1Namespaces(BaseModel):
    namespaces: List[V1Namespace]
