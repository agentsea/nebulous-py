from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from nebu.meta import V1ResourceMeta, V1ResourceMetaRequest


# Match Rust "V1ErrorResponse" struct
class V1ErrorResponse(BaseModel):
    response_type: str = Field(default="ErrorResponse", alias="type")
    request_id: str
    error: str
    traceback: Optional[str] = None


# Match Rust "V1Meter" struct
class V1Meter(BaseModel):
    cost: Optional[float] = None
    costp: Optional[float] = None
    currency: str
    unit: str
    metric: str
    request_json_path: Optional[str] = None
    response_json_path: Optional[str] = None


# Match Rust "V1EnvVar" struct
class V1EnvVar(BaseModel):
    key: str
    value: str


# Match Rust "V1ContainerResources" struct
class V1ContainerResources(BaseModel):
    min_cpu: Optional[float] = None
    min_memory: Optional[float] = None
    max_cpu: Optional[float] = None
    max_memory: Optional[float] = None


# Match Rust "V1SSHKey" struct
class V1SSHKey(BaseModel):
    public_key: Optional[str] = None
    public_key_secret: Optional[str] = None
    copy_local: Optional[bool] = None


# Match Rust "RestartPolicy" enum with default ("Never") for now
# If you need a typed enum, consider using from enum import Enum in Python
DEFAULT_RESTART_POLICY = "Never"


# Match Rust "V1VolumeDriver" enum
class V1VolumeDriver(str, Enum):
    RCLONE_SYNC = "RCLONE_SYNC"
    RCLONE_BISYNC = "RCLONE_BISYNC"
    RCLONE_MOUNT = "RCLONE_MOUNT"


# Match Rust "V1VolumePath" struct
# "resync" defaults to false, "continuous" defaults to true, "driver" defaults to RCLONE_SYNC
class V1VolumePath(BaseModel):
    source: str
    dest: str
    resync: bool = False
    continuous: bool = False
    driver: V1VolumeDriver = V1VolumeDriver.RCLONE_SYNC


# Match Rust "V1VolumeConfig" struct
# If you know your default_cache_dir, replace "cache" with the actual default
class V1VolumeConfig(BaseModel):
    paths: List[V1VolumePath]
    cache_dir: str = "/nebu/cache"


# Match Rust "V1ContainerStatus" struct
class V1ContainerStatus(BaseModel):
    status: Optional[str] = None
    message: Optional[str] = None
    accelerator: Optional[str] = None
    public_ip: Optional[str] = None
    cost_per_hr: Optional[float] = None


# pub struct V1PortRequest {
#     pub port: u16,
#     pub protocol: Option<String>,
#     pub public: Option<bool>,
# }


class V1PortRequest(BaseModel):
    port: int
    protocol: Optional[str] = None
    public: bool = False


class V1Port(BaseModel):
    port: int
    protocol: Optional[str] = None
    public: bool = False


# Match Rust "V1ContainerRequest" struct
# "kind" defaults to "Container", "restart" defaults to "Never"
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
    public_ip: Optional[bool] = None


# Match Rust "V1Container" struct
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
    public_ip: bool = False
    ports: Optional[List[V1Port]] = None


# Match Rust "V1UpdateContainer" struct
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
