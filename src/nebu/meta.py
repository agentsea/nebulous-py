from typing import Dict, Optional

from pydantic import BaseModel


# Match Rust "V1ResourceMetaRequest" struct
class V1ResourceMetaRequest(BaseModel):
    name: Optional[str] = None
    namespace: Optional[str] = None
    labels: Optional[Dict[str, str]] = None
    owner: Optional[str] = None
    owner_ref: Optional[str] = None


class V1ResourceMeta(BaseModel):
    name: str
    namespace: str
    id: str
    owner: str
    created_at: int
    updated_at: int
    created_by: str
    owner_ref: Optional[str] = None
    labels: Optional[Dict[str, str]] = None


class V1ResourceReference(BaseModel):
    kind: str
    name: str
    namespace: str

    def to_string(self) -> str:
        return f"{self.name}.{self.namespace}.{self.kind}"
