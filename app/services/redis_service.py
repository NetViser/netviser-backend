from typing import Any, Dict, Optional
from fastapi import HTTPException
from app.configs.config import get_settings
import redis
import json

settings = get_settings()

class RedisClient:
    """
    Singleton class for interacting with Redis.
    """

    _instance = None

    def __new__(cls):
        """Create a new instance of the Redis client if it doesn't already exist."""
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
            cls._instance.initialize_client()
        return cls._instance

    def initialize_client(self):
        """Initialize the Redis client."""
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True,
            username=settings.REDIS_USERNAME,
            password=settings.REDIS_PASSWORD,
        )

    def set_session_data(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl_in_seconds: int = 300  # Default to 5 minutes
    ) -> None:
        """
        Store session data in Redis with an optional TTL (default 5 minutes).
        """
        try:
            # Serialize the dictionary to a JSON string before storing in Redis
            json_data = json.dumps(data)
            self.client.set(name=session_id, value=json_data, ex=ttl_in_seconds)
        except redis.RedisError as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data from Redis by session_id. Returns None if it doesn't exist.
        """
        try:
            session_data = self.client.get(session_id)
            if session_data is None:
                return None
            # Deserialize the JSON string back to a Python dictionary
            return json.loads(session_data)
        except redis.RedisError as e:
            raise HTTPException(status_code=500, detail=str(e))

    def delete_session_data(self, session_id: str) -> None:
        """
        Delete session data from Redis if it exists.
        """
        try:
            self.client.delete(session_id)
        except redis.RedisError as e:
            raise HTTPException(status_code=500, detail=str(e))
