import os
import redis
import json
from typing import Optional, Dict, Any

_IN_MEMORY_STORE = {}


class RedisManager:
    def __init__(self):
        self.use_redis = False
        self.client = None

        try:
            REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self.client = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=5)
            # Test connection
            self.client.ping()
            self.use_redis = True
            print("✓ Redis connected successfully")
        except Exception as e:
            print(f"⚠ Redis connection failed, using in-memory storage: {e}")
            self.use_redis = False

    def set_job_status(self, job_id: str, status: str, ttl: int = 3600):
        if self.use_redis and self.client:
            self.client.setex(f"job:{job_id}:status", ttl, status)
        else:
            _IN_MEMORY_STORE[f"job:{job_id}:status"] = status

    def get_job_status(self, job_id: str) -> Optional[str]:
        if self.use_redis and self.client:
            return self.client.get(f"job:{job_id}:status")
        return _IN_MEMORY_STORE.get(f"job:{job_id}:status")

    def set_job_result(self, job_id: str, result: Dict[str, Any], ttl: int = 3600):
        if self.use_redis and self.client:
            self.client.setex(f"job:{job_id}:result", ttl, json.dumps(result))
        else:
            _IN_MEMORY_STORE[f"job:{job_id}:result"] = result

    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        if self.use_redis and self.client:
            result = self.client.get(f"job:{job_id}:result")
            return json.loads(result) if result else None
        return _IN_MEMORY_STORE.get(f"job:{job_id}:result")

    def job_exists(self, job_id: str) -> bool:
        if self.use_redis and self.client:
            return self.client.exists(f"job:{job_id}:status") > 0
        return f"job:{job_id}:status" in _IN_MEMORY_STORE


redis_manager = RedisManager()