import os
import time
from typing import Any, Optional, cast

import redis
from pydantic import BaseModel, Field


class OwnedValue(BaseModel):
    created_at: int = Field(default_factory=lambda: int(time.time()))
    value: str
    user_id: Optional[str] = None
    orgs: Optional[Any] = None
    handle: Optional[str] = None
    owner: Optional[str] = None


class Cache:
    """
    A simple cache class that connects to Redis.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        Initializes the Redis connection.
        Pulls connection details from environment variables REDIS_HOST,
        REDIS_PORT, and REDIS_DB if available, otherwise uses defaults.
        Also checks for REDIS_URL and prefers that if set.
        """
        redis_url = os.environ.get("REDIS_URL")
        namespace = os.environ.get("NEBU_NAMESPACE")
        if not namespace:
            raise ValueError("NEBU_NAMESPACE environment variable is not set")

        self.redis_client = None
        connection_info = ""

        try:
            if redis_url:
                # Use REDIS_URL if available
                self.redis_client = redis.StrictRedis.from_url(
                    redis_url, decode_responses=True
                )
                connection_info = f"URL {redis_url}"
            else:
                # Fallback to individual host, port, db
                redis_host = os.environ.get("REDIS_HOST", host)
                redis_port = int(os.environ.get("REDIS_PORT", port))
                redis_db = int(os.environ.get("REDIS_DB", db))
                self.redis_client = redis.StrictRedis(
                    host=redis_host, port=redis_port, db=redis_db, decode_responses=True
                )
                connection_info = f"{redis_host}:{redis_port}/{redis_db}"

            # Ping the server to ensure connection is established
            self.redis_client.ping()
            print(f"Successfully connected to Redis using {connection_info}")

            self.prefix = f"cache:{namespace}"
        except Exception as e:
            print(f"Error connecting to Redis: {e}")
            # Ensure client is None if connection fails at any point
            self.redis_client = None

    def get(self, key: str) -> str | None:
        """
        Gets the value associated with a key from Redis.
        Returns None if the key does not exist or connection failed.
        """
        if not self.redis_client:
            print("Redis client not connected.")
            return None
        try:
            key = f"{self.prefix}:{key}"
            # Cast the result to str | None as expected
            result = self.redis_client.get(key)
            return cast(str | None, result)
        except Exception as e:
            print(f"Error getting key '{key}' from Redis: {e}")
            return None

    def set(self, key: str, value: str, expiry_seconds: int | None = None) -> bool:
        """
        Sets a key-value pair in Redis.
        Optionally sets an expiry time for the key in seconds.
        Returns True if successful, False otherwise (e.g., connection failed).
        """
        if not self.redis_client:
            print("Redis client not connected.")
            return False
        try:
            key = f"{self.prefix}:{key}"
            if expiry_seconds:
                # Cast the result to bool
                result = self.redis_client.setex(key, expiry_seconds, value)
                return cast(bool, result)
            else:
                # Cast the result to bool
                result = self.redis_client.set(key, value)
                return cast(bool, result)
        except Exception as e:
            print(f"Error setting key '{key}' in Redis: {e}")
            return False
