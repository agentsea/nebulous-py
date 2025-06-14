from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from nebulous.meta import V1ResourceMeta, V1ResourceMetaRequest


# 1) If you're still using V1ErrorResponse, no change needed here.
class V1ErrorResponse(BaseModel):
    response_type: str = Field(default="ErrorResponse", alias="type")
    request_id: str
    error: str
    traceback: Optional[str] = None


class V1Meter(BaseModel):
    cost: Optional[float] = None
    costp: Optional[float] = None
    currency: str
    unit: str
    metric: str
    json_path: Optional[str] = None


class V1EnvVar(BaseModel):
    key: str
    value: Optional[str] = None
    secret_name: Optional[str] = None


class V1ContainerResources(BaseModel):
    min_cpu: Optional[float] = None
    min_memory: Optional[float] = None
    max_cpu: Optional[float] = None
    max_memory: Optional[float] = None


class V1SSHKey(BaseModel):
    public_key: Optional[str] = None
    public_key_secret: Optional[str] = None
    copy_local: Optional[bool] = None


DEFAULT_RESTART_POLICY = "Never"


class V1VolumeDriver(str, Enum):
    RCLONE_SYNC = "RCLONE_SYNC"
    RCLONE_BISYNC = "RCLONE_BISYNC"
    RCLONE_MOUNT = "RCLONE_MOUNT"
    RCLONE_COPY = "RCLONE_COPY"


class V1VolumePath(BaseModel):
    source: str
    dest: str
    resync: bool = False
    continuous: bool = False
    driver: V1VolumeDriver = V1VolumeDriver.RCLONE_SYNC

    model_config = ConfigDict(use_enum_values=True)


class V1VolumeConfig(BaseModel):
    paths: List[V1VolumePath]
    cache_dir: str = "/nebu/cache"

    model_config = ConfigDict(use_enum_values=True)


class V1ContainerStatus(BaseModel):
    status: Optional[str] = None
    message: Optional[str] = None
    accelerator: Optional[str] = None
    tailnet_url: Optional[str] = None
    cost_per_hr: Optional[float] = None
    ready: Optional[bool] = None


class V1AuthzSecretRef(BaseModel):
    name: Optional[str] = None
    key: Optional[str] = None


class V1AuthzJwt(BaseModel):
    secret_ref: Optional[V1AuthzSecretRef] = None


class V1AuthzPathMatch(BaseModel):
    path: Optional[str] = None
    pattern: Optional[str] = None


class V1AuthzFieldMatch(BaseModel):
    json_path: Optional[str] = None
    pattern: Optional[str] = None


class V1AuthzRuleMatch(BaseModel):
    roles: Optional[List[str]] = None


class V1AuthzRule(BaseModel):
    name: str
    rule_match: Optional[V1AuthzRuleMatch] = Field(default=None, alias="match")
    allow: bool
    field_match: Optional[List[V1AuthzFieldMatch]] = None
    path_match: Optional[List[V1AuthzPathMatch]] = None


class V1AuthzConfig(BaseModel):
    enabled: bool = False
    default_action: str = "deny"
    auth_type: str = "jwt"
    jwt: Optional[V1AuthzJwt] = None
    rules: Optional[List[V1AuthzRule]] = None


class V1ContainerHealthCheck(BaseModel):
    interval: Optional[str] = None
    timeout: Optional[str] = None
    retries: Optional[int] = None
    start_period: Optional[str] = None
    path: Optional[str] = None
    port: Optional[int] = None
    protocol: Optional[str] = None


class V1PortRequest(BaseModel):
    port: int
    protocol: Optional[str] = None
    public: Optional[bool] = None


class V1Port(BaseModel):
    port: int
    protocol: Optional[str] = None
    public_ip: Optional[str] = None


class V1ContainerRequest(BaseModel):
    kind: str = Field(default="Container")
    platform: Optional[str] = None
    metadata: Optional[V1ResourceMetaRequest] = None
    image: str
    env: Optional[List[V1EnvVar]] = None
    command: Optional[str] = None
    volumes: Optional[List[V1VolumePath]] = None
    accelerators: Optional[List[str]] = None
    resources: Optional[V1ContainerResources] = None
    meters: Optional[List[V1Meter]] = None
    restart: str = Field(default=DEFAULT_RESTART_POLICY)
    queue: Optional[str] = None
    timeout: Optional[str] = None
    ssh_keys: Optional[List[V1SSHKey]] = None
    ports: Optional[List[V1PortRequest]] = None
    proxy_port: Optional[int] = None
    authz: Optional[V1AuthzConfig] = None
    health_check: Optional[V1ContainerHealthCheck] = None

    model_config = ConfigDict(use_enum_values=True)


class V1Container(BaseModel):
    kind: str = Field(default="Container")
    platform: str
    metadata: V1ResourceMeta
    image: str
    env: Optional[List[V1EnvVar]] = None
    command: Optional[str] = None
    volumes: Optional[List[V1VolumePath]] = None
    accelerators: Optional[List[str]] = None
    meters: Optional[List[V1Meter]] = None
    restart: str = Field(default=DEFAULT_RESTART_POLICY)
    queue: Optional[str] = None
    timeout: Optional[str] = None
    resources: Optional[V1ContainerResources] = None
    status: Optional[V1ContainerStatus] = None
    ssh_keys: Optional[List[V1SSHKey]] = None
    ports: Optional[List[V1Port]] = None
    proxy_port: Optional[int] = None
    authz: Optional[V1AuthzConfig] = None
    health_check: Optional[V1ContainerHealthCheck] = None

    model_config = ConfigDict(use_enum_values=True)


class V1UpdateContainer(BaseModel):
    image: Optional[str] = None
    env: Optional[List[V1EnvVar]] = None
    command: Optional[str] = None
    volumes: Optional[List[V1VolumePath]] = None
    accelerators: Optional[List[str]] = None
    labels: Optional[Dict[str, str]] = None
    cpu_request: Optional[str] = None
    memory_request: Optional[str] = None
    platform: Optional[str] = None
    meters: Optional[List[V1Meter]] = None
    restart: Optional[str] = None
    queue: Optional[str] = None
    timeout: Optional[str] = None
    resources: Optional[V1ContainerResources] = None
    proxy_port: Optional[int] = None
    authz: Optional[V1AuthzConfig] = None
    health_check: Optional[V1ContainerHealthCheck] = None
    no_delete: Optional[bool] = None

    model_config = ConfigDict(use_enum_values=True)


class V1Containers(BaseModel):
    containers: List[V1Container]

    model_config = ConfigDict(use_enum_values=True)


class V1ContainerSearch(BaseModel):
    namespace: Optional[str] = None
    image: Optional[str] = None
    env: Optional[List[V1EnvVar]] = None
    command: Optional[str] = None
    args: Optional[str] = None
    volumes: Optional[List[V1VolumePath]] = None
    accelerators: Optional[List[str]] = None
    labels: Optional[Dict[str, str]] = None
    cpu_request: Optional[str] = None
    memory_request: Optional[str] = None
    platform: Optional[str] = None
    health_check: Optional[V1ContainerHealthCheck] = None
    meters: Optional[List[V1Meter]] = None
    restart: Optional[str] = None
    queue: Optional[str] = None
    timeout: Optional[str] = None
    resources: Optional[V1ContainerResources] = None
    proxy_port: Optional[int] = None
    authz: Optional[V1AuthzConfig] = None

    model_config = ConfigDict(use_enum_values=True)
