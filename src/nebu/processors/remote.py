from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from nebu.config import GlobalConfig
from nebu.processors.models import V1ContainerRequest, V1Scale

from .models import Message
from .processor import Processor

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)


class RemoteProcessor(ABC, Processor):
    def __init__(
        self,
        name: str,
        namespace: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        container: Optional[V1ContainerRequest] = None,
        schema_: Optional[Any] = None,
        common_schema: Optional[str] = None,
        min_replicas: Optional[int] = None,
        max_replicas: Optional[int] = None,
        scale_config: Optional[V1Scale] = None,
        config: Optional[GlobalConfig] = None,
        no_delete: bool = False,
    ):
        super().__init__(
            name,
            namespace,
            labels,
            container,
            schema_,
            common_schema,
            min_replicas,
            max_replicas,
            scale_config,
            config,
            no_delete,
        )

    @abstractmethod
    def process(self, message: Message[I]) -> Type[BaseModel]:
        pass
