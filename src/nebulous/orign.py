import os
from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class OrignServerConfig:
    """
    Python equivalent of the Rust ServerConfig struct.
    """

    name: Optional[str] = None
    api_key: Optional[str] = None
    server: Optional[str] = None
    auth_server: Optional[str] = None


def get_orign_server() -> Optional[str]:
    """
    Get the Orign server URL from environment variables.
    """
    env = os.environ.get("ORIGN_SERVER") or os.environ.get("ORIGIN_SERVER")
    if env:
        return env

    path = _get_config_file_path()
    path_exists = os.path.exists(path)

    # Load from disk or create a default
    if path_exists:
        try:
            with open(path, "r") as yaml_file:
                data = yaml.safe_load(yaml_file) or {}
            # Convert each server entry into a ServerConfig
            servers_data = data.get("servers", [])
            # Ensure data is a list and elements are dicts before comprehension
            if isinstance(servers_data, list):
                servers = [
                    OrignServerConfig(**srv)
                    for srv in servers_data
                    if isinstance(srv, dict)
                ]
            else:
                servers = []  # Or handle error appropriately
            current_server = data.get("current_server")

            for srv in servers:
                if srv.name == current_server:
                    return srv.server

            return None
        except (yaml.YAMLError, TypeError, FileNotFoundError) as e:
            # Handle potential issues during loading or parsing
            print(
                f"Warning: Could not load or parse config file '{path}': {e}. Starting with default config."
            )
            return None

    return None


def _get_config_file_path() -> str:
    """
    Return the path to ~/.agentsea/orign.yaml
    """
    home = os.path.expanduser("~")
    config_dir = os.path.join(home, ".agentsea")
    return os.path.join(config_dir, "orign.yaml")
