from typing import Dict, Optional

import requests
from pydantic import BaseModel

from nebulous.config import GlobalConfig


class V1UserProfile(BaseModel):
    email: str
    display_name: Optional[str] = None
    handle: Optional[str] = None
    picture: Optional[str] = None
    organization: Optional[str] = None
    role: Optional[str] = None
    external_id: Optional[str] = None
    actor: Optional[str] = None
    # structure is {"org_id": {"org_name": <name>, "org_role": <role>}}
    organizations: Optional[Dict[str, Dict[str, str]]] = None
    created: Optional[int] = None
    updated: Optional[int] = None
    token: Optional[str] = None


def get_user_profile(api_key: str) -> V1UserProfile:
    config = GlobalConfig.read()
    current_server_config = config.get_current_server_config()
    if current_server_config is None:
        raise ValueError("No current server config found")
    url = f"{current_server_config.server}/v1/users/me"

    response = requests.get(url, headers={"Authorization": f"Bearer {api_key}"})
    response.raise_for_status()

    return V1UserProfile.model_validate(response.json())


def is_allowed(
    resource_owner: str,
    user_id: Optional[str] = None,
    orgs: Optional[Dict[str, Dict[str, str]]] = None,
) -> bool:
    if orgs is None:
        orgs = {}
    owners = []
    for org_id, _ in orgs.items():
        owners.append(org_id)
    if user_id:
        owners.append(user_id)
    return resource_owner in owners
