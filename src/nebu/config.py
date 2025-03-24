from __future__ import annotations

import os
from typing import Optional

import yaml


class Config:
    NEBU_ADDR = os.getenv("NEBU_URL", "https://orign.agentlabs.xyz")
    AGENTSEA_API_KEY = os.getenv("AGENTSEA_API_KEY")


class GlobalConfig:
    def __init__(
        self,
        api_key: Optional[str] = None,
        server: Optional[str] = None,
        debug: bool = False,
    ):
        self.api_key = api_key or Config.AGENTSEA_API_KEY
        self.server = server or Config.NEBU_ADDR
        self.debug = debug

    def write(self) -> None:
        home = os.path.expanduser("~")
        dir = os.path.join(home, ".agentsea")
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, "nebu.yaml")

        with open(path, "w") as yaml_file:
            yaml.dump(self.__dict__, yaml_file)
            yaml_file.flush()
            yaml_file.close()

    @classmethod
    def read(cls) -> GlobalConfig:
        home = os.path.expanduser("~")
        dir = os.path.join(home, ".agentsea")
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, "nebu.yaml")

        if not os.path.exists(path):
            return GlobalConfig()

        with open(path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            return GlobalConfig(**config)
