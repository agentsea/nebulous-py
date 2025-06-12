from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class ServerConfig:
    """
    Python equivalent of the Rust ServerConfig struct.
    """

    name: Optional[str] = None
    api_key: Optional[str] = None
    server: Optional[str] = None
    auth_server: Optional[str] = None


@dataclass
class GlobalConfig:
    """
    Python equivalent of the Rust GlobalConfig struct.
    Manages multiple ServerConfig entries and a current_server pointer.
    """

    servers: List[ServerConfig] = field(default_factory=list)
    current_server: Optional[str] = None

    @classmethod
    def read(cls) -> GlobalConfig:
        """
        Read the config from ~/.agentsea/nebu.yaml, or create a default if it doesn't exist.
        Then ensure that we either find or create a matching server from environment variables,
        and set that as the `current_server` if relevant (mimicking the Rust logic).
        """
        path = _get_config_file_path()
        path_exists = os.path.exists(path)

        # Load from disk or create a default
        if path_exists:
            with open(path, "r") as yaml_file:
                data = yaml.safe_load(yaml_file) or {}
            # Convert each server entry into a ServerConfig
            servers_data = data.get("servers", [])
            servers = [ServerConfig(**srv) for srv in servers_data]
            current_server = data.get("current_server")
            config = cls(servers=servers, current_server=current_server)
        else:
            config = cls()  # default

        # Collect environment variables (no fallback defaults here)
        env_api_key = (
            os.environ.get("NEBU_API_KEY")
            or os.environ.get("NEBULOUS_API_KEY")
            or os.environ.get("AGENTSEA_API_KEY")
            or os.environ.get("ORIGN_API_KEY")
            or os.environ.get("ORIGIN_API_KEY")
        )
        env_server = os.environ.get("NEBU_SERVER") or os.environ.get("NEBULOUS_SERVER")
        env_auth_server = (
            os.environ.get("NEBU_AUTH_SERVER")
            or os.environ.get("NEBULOUS_AUTH_SERVER")
            or os.environ.get("AGENTSEA_AUTH_SERVER")
            or os.environ.get("ORIGN_AUTH_SERVER")
            or os.environ.get("ORIGIN_AUTH_SERVER")
            or os.environ.get("AGENTSEA_AUTH_URL")
        )

        # Only proceed if all three environment variables are present
        if env_api_key and env_server and env_auth_server:
            # Find a matching server
            found_server = None
            for srv in config.servers:
                if (
                    srv.api_key == env_api_key
                    and srv.server == env_server
                    and srv.auth_server == env_auth_server
                ):
                    found_server = srv
                    break

            server_name = "env-based-server"
            if found_server:
                # Ensure it has a name, so we can set current_server to it
                if found_server.name is None:
                    found_server.name = server_name
                # Use that server's name as current
                config.current_server = found_server.name
            else:
                # Create a new server entry
                new_server = ServerConfig(
                    name=server_name,
                    api_key=env_api_key,
                    server=env_server,
                    auth_server=env_auth_server,
                )
                config.servers.append(new_server)
                config.current_server = server_name

        # Write if the file didn't already exist
        if not path_exists:
            config.write()

        return config

    def write(self) -> None:
        """
        Write the current GlobalConfig to disk as YAML.
        """
        path = _get_config_file_path()
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Convert our dataclass-based objects into simple dictionaries
        data = {
            "servers": [srv.__dict__ for srv in self.servers],
            "current_server": self.current_server,
        }

        with open(path, "w") as yaml_file:
            yaml.dump(data, yaml_file)
            yaml_file.flush()

    def get_current_server_config(self) -> Optional[ServerConfig]:
        """
        Get the server config for the current_server name, or None if unset/missing.
        """
        if self.current_server:
            for srv in self.servers:
                if srv.name == self.current_server:
                    return srv
        return None

    @classmethod
    def get_server_url(cls) -> str:
        """
        Get the server URL for the current_server name, or None if unset/missing.
        """
        config = cls.read()
        server_config = config.get_current_server_config()
        server = os.environ.get("NEBU_SERVER") or os.environ.get("NEBULOUS_SERVER")
        if not server:
            server = server_config.server if server_config else None
        if not server:
            raise ValueError("NEBULOUS_SERVER environment variable is not set")
        return server


def _get_config_file_path() -> str:
    """
    Return the path to ~/.agentsea/nebu.yaml
    """
    home = os.path.expanduser("~")
    config_dir = os.path.join(home, ".agentsea")
    return os.path.join(config_dir, "nebu.yaml")


@dataclass
class ContainerConfig:
    """
    Configuration loaded from environment variables inside the container.
    """

    api_key: Optional[str] = None
    server: Optional[str] = None
    namespace: Optional[str] = None
    name: Optional[str] = None
    container_id: Optional[str] = None
    date: Optional[str] = None
    hf_home: Optional[str] = None
    namespace_volume_uri: Optional[str] = None
    name_volume_uri: Optional[str] = None
    ts_authkey: Optional[str] = None

    @classmethod
    def from_env(cls) -> ContainerConfig:
        """
        Load configuration from environment variables.
        """
        return cls(
            api_key=os.environ.get("NEBU_API_KEY")
            or os.environ.get("NEBULOUS_API_KEY")
            or os.environ.get("ORIGN_API_KEY")
            or os.environ.get("ORIGIN_API_KEY"),
            server=os.environ.get("NEBU_SERVER") or os.environ.get("NEBULOUS_SERVER"),
            namespace=os.environ.get("NEBU_NAMESPACE")
            or os.environ.get("NEBULOUS_NAMESPACE")
            or os.environ.get("ORIGN_NAMESPACE")
            or os.environ.get("ORIGIN_NAMESPACE"),
            name=os.environ.get("NEBU_NAME"),
            container_id=os.environ.get("NEBU_CONTAINER_ID"),
            date=os.environ.get("NEBU_DATE"),
            hf_home=os.environ.get("HF_HOME"),
            namespace_volume_uri=os.environ.get("NAMESPACE_VOLUME_URI"),
            name_volume_uri=os.environ.get("NAME_VOLUME_URI"),
            ts_authkey=os.environ.get("TS_AUTHKEY"),
        )
