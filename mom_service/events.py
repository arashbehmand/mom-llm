"""
Event definitions and Redis publisher for MoM Service.
"""

import logging
import os
from typing import Any, Optional

import redis.asyncio as redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MoMEvent(BaseModel):
    """Base class for all MoM events."""

    type: str
    request_id: str
    timestamp: float
    data: dict[str, Any]


class RedisPublisher:
    """
    Handles publishing events to Redis.
    Designed to be resilient: failures to publish are logged but do not raise exceptions.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.redis_client: Optional[redis.Redis] = None
        self.enabled = False

        if self.redis_url:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                self.enabled = True
                logger.info(f"RedisPublisher initialized with URL: {self.redis_url}")
            except Exception as e:
                logger.error(f"Failed to initialize Redis client: {e}")
        else:
            logger.info("RedisPublisher disabled (no REDIS_URL provided).")

    async def close(self):
        """Close the Redis connection."""
        if self.redis_client:
            await self.redis_client.close()

    async def publish(self, event: MoMEvent):
        """
        Publish an event to the 'mom_events' channel.
        """
        if not self.enabled or not self.redis_client:
            return

        try:
            message = event.model_dump_json()
            await self.redis_client.publish("mom_events", message)
            logger.debug(f"Published event: {event.type} for {event.request_id}")
        except Exception as e:
            # Fire-and-forget: don't let reporting failures impact the main service
            logger.warning(f"Failed to publish event to Redis: {e}")
