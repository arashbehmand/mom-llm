"""In-memory custody of provider-native tool-call ids behind MoM-minted client ids.

MoM mints a stable ``call_...`` id for every synthesizer tool call so a provider-native id — most
importantly Gemini's ``call_..__thought__..`` thought-signature — never reaches the client. The raw
id is stashed here under the minted id so a later relay continuation can restore it for the *same*
synthesizer (matched by ``owner``).

Deliberately best-effort and process-local: this is a per-process cache, not durable state, so MoM
stays stateless at the protocol boundary (clients still resend the full transcript each turn). A
miss — a restart, a different worker, or an evicted entry — simply relays the minted id, which
providers accept; only Gemini's thought-signature optimization is lost on a miss, never correctness.
"""

from __future__ import annotations

from collections import OrderedDict


class InMemoryToolCallCustody:
    """A bounded LRU map: minted client id -> (provider id, owner). Process-local, not shared."""

    def __init__(self, *, max_entries: int = 100_000) -> None:
        self._max_entries = max_entries
        self._store: OrderedDict[str, tuple[str, str]] = OrderedDict()

    def remember(self, client_id: str, provider_id: str, owner: str) -> None:
        self._store[client_id] = (provider_id, owner)
        self._store.move_to_end(client_id)
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)  # evict the oldest

    def provider_id(self, client_id: str, owner: str) -> str | None:
        record = self._store.get(client_id)
        if record is None or record[1] != owner:
            return None
        self._store.move_to_end(client_id)  # touch: keep live conversations warm
        return record[0]
