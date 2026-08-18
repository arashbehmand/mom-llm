"""Concrete Clock / IdFactory adapters for production."""

from __future__ import annotations

import time
import uuid


class SystemClock:
    def now(self) -> float:
        return time.time()


class UuidIds:
    def new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"
