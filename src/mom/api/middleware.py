"""ASGI middleware: a soft daily-budget alarm.

When ``budgets.daily_usd`` is configured and today's spend (summed since local midnight from the
metrics reader) exceeds it, the response is tagged ``X-MoM-Budget: exceeded`` and a warning is
logged. It is an alarm, never a gate — the request is always served. With no budget configured the
middleware short-circuits before any query, so the common case pays nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from starlette.datastructures import MutableHeaders

from mom.runtime.logging import get_logger


if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    from mom.runtime.container import Container


logger = get_logger("mom.budget")


def _local_midnight(now_epoch: float) -> float:
    """Epoch seconds of the most recent local midnight (the start of 'today')."""
    local = datetime.fromtimestamp(now_epoch, tz=UTC).astimezone()
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp()


class BudgetAlarmMiddleware:
    """Tags responses with ``X-MoM-Budget: exceeded`` when the day's spend is over budget."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not await self._exceeded(scope):
            await self.app(scope, receive, send)
            return

        async def send_with_alarm(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["X-MoM-Budget"] = "exceeded"
            await send(message)

        await self.app(scope, receive, send_with_alarm)

    async def _exceeded(self, scope: Scope) -> bool:
        app = scope.get("app")
        container: Container | None = getattr(getattr(app, "state", None), "container", None)
        if container is None:
            return False
        budget = container.catalog.config.budgets.daily_usd
        reader = container.metrics_reader
        if budget is None or reader is None:
            return False
        try:
            agg = await reader.aggregate(start=_local_midnight(container.clock.now()))
            cost = agg.get("cost_usd", 0.0)
            spent = float(cost) if isinstance(cost, int | float) else 0.0
        except Exception:  # a soft alarm must never break the request it is checking
            return False
        if spent > budget:
            logger.warning("daily budget exceeded", spent_usd=spent, budget_usd=budget)
            return True
        return False
