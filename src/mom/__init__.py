"""MoM — a Mixture-of-Models OpenAI/Anthropic-compatible LLM gateway.

Importing this package has exactly one side effect — the ``LITELLM_LOCAL_MODEL_COST_MAP``
default below — and otherwise defines the version and nothing else. All wiring happens in
:func:`mom.api.app.create_app` via an explicit lifespan/composition root.
"""

from __future__ import annotations

import os


# litellm reads this the moment IT is imported, and decides then whether to use the catalog
# bundled in its wheel or fetch one over the network. mom wants the bundled one: fetching is
# slow at import time and pytest-socket blocks it outright. So the default has to be in place
# before *anything* imports litellm — and this module is the one place that is guaranteed,
# since every entry into the codebase (the `mom` console script, the ASGI app, a test's
# `import mom.<anything>`) executes it first.
#
# It used to live in mom.adapters.litellm_client instead, which reads as the natural home but
# only wins the race while nothing imports litellm before that module does. Today nothing does
# (runtime.wiring imports the adapter at module scope, and the adapter's own litellm imports are
# lazy) — but that is a property of the current import graph, not a guarantee, and when it flips
# the catalog silently changes underneath every model call. Setting it here removes the ordering
# question rather than continuing to win it by luck.
#
# An explicit value in the *process* environment still wins: `LITELLM_LOCAL_MODEL_COST_MAP=False`
# forces the network fetch, which is the escape hatch when config.yaml adopts a model that the
# pinned litellm's bundled catalog predates (see the litellm floor in pyproject.toml). It has to
# be a real env var, not a `.env` entry: this line runs at import, long before any `.env` is read
# (mom.runtime.secrets), and both this setdefault and dotenv are first-wins.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


__version__ = "2.0.0"

__all__ = ["__version__"]
