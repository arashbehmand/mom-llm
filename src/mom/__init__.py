"""MoM — a Mixture-of-Models OpenAI/Anthropic-compatible LLM gateway.

Importing this package has no side effects: it defines the version and nothing else. All
wiring happens in :func:`mom.api.app.create_app` via an explicit lifespan/composition root.
"""

from __future__ import annotations


__version__ = "2.0.0.dev0"

__all__ = ["__version__"]
