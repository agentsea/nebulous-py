from typing import Dict, List, Optional

import requests

from nebulous.config import GlobalConfig
from nebulous.logging import logger
from nebulous.namespaces.models import (
    V1Namespace,
    V1NamespaceMetaRequest,
    V1NamespaceRequest,
    V1Namespaces,
)


class Namespace:
    """
    A class for managing Namespace instances.
    """

    def __init__(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        owner: Optional[str] = None,
        config: Optional[GlobalConfig] = None,
        api_key: Optional[str] = None,
    ):
        self.config = config or GlobalConfig.read()
        if not self.config:
            raise ValueError("No config found")
        current_server = self.config.get_current_server_config()
        if not current_server:
            raise ValueError("No server config found")
        self.current_server = current_server
        self.api_key = api_key or current_server.api_key
        self.nebu_host = current_server.server
        self.name = name
        self.labels = labels
        self.owner = owner
        self.namespaces_url = f"{self.nebu_host}/v1/namespaces"

        # Fetch existing Namespaces
        response = requests.get(
            self.namespaces_url, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        response.raise_for_status()
        resp_json = response.json()
        logger.debug(resp_json)
        existing_namespaces = V1Namespaces.model_validate(resp_json)
        self.namespace: Optional[V1Namespace] = next(
            (
                namespace_val
                for namespace_val in existing_namespaces.namespaces
                if namespace_val.metadata.name == name
            ),
            None,
        )

        # If not found, create
        if not self.namespace:
            logger.info(f"Creating namespace {name}")
            # Create metadata and namespace request
            metadata = V1NamespaceMetaRequest(name=name, labels=labels, owner=owner)

            namespace_request = V1NamespaceRequest(
                metadata=metadata,
            )

            create_response = requests.post(
                self.namespaces_url,
                json=namespace_request.model_dump(exclude_none=True),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            create_response.raise_for_status()
            self.namespace = V1Namespace.model_validate(create_response.json())
            logger.info(f"Created Namespace {self.namespace.metadata.name}")
        else:
            logger.info(
                f"Found Namespace {self.namespace.metadata.name}, no update needed"
            )

    @classmethod
    def load(
        cls,
        name: str,
        config: Optional[GlobalConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Get a Namespace from the remote server.
        """
        namespaces = cls.get(name=name, config=config, api_key=api_key)
        if not namespaces:
            raise ValueError("Namespace not found")
        namespace_v1 = namespaces[0]

        out = cls.__new__(cls)
        out.namespace = namespace_v1
        out.config = config or GlobalConfig.read()
        if not out.config:
            raise ValueError("No config found")
        out.current_server = out.config.get_current_server_config()
        if not out.current_server:
            raise ValueError("No server config found")
        out.api_key = api_key or out.current_server.api_key
        out.nebu_host = out.current_server.server
        out.namespaces_url = f"{out.nebu_host}/v1/namespaces"
        out.name = name

        return out

    @classmethod
    def get(
        cls,
        name: Optional[str] = None,
        config: Optional[GlobalConfig] = None,
        api_key: Optional[str] = None,
    ) -> List[V1Namespace]:
        """
        Get a list of Namespaces that optionally match the name filter.
        """
        config = config or GlobalConfig.read()
        if not config:
            raise ValueError("No config found")
        current_server = config.get_current_server_config()
        if not current_server:
            raise ValueError("No server config found")
        api_key = api_key or current_server.api_key
        namespaces_url = f"{current_server.server}/v1/namespaces"

        response = requests.get(
            namespaces_url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()

        namespaces_response = V1Namespaces.model_validate(response.json())
        filtered_namespaces = namespaces_response.namespaces

        if name:
            filtered_namespaces = [
                n for n in filtered_namespaces if n.metadata.name == name
            ]

        return filtered_namespaces

    def delete(self):
        """
        Delete the Namespace.
        """
        if not self.namespace or not self.namespace.metadata.name:
            raise ValueError("Namespace not found")

        url = f"{self.namespaces_url}/{self.namespace.metadata.name}"
        response = requests.delete(
            url, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        response.raise_for_status()
        return
