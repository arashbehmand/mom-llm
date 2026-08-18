"""``GET /v1/models`` (+/{id}) and ``GET /v1/model/info`` with capability metadata.

Serves OpenAI's required fields plus OpenRouter-convention fields and a ``mom`` vendor block. Model
discovery is the one place where compatible clients disagree on the *envelope*, so ``/v1/models``
answers in the dialect the request asks for: the Anthropic list shape when it carries
anthropic-version / x-api-key, and the Codex catalog shape when it carries ``?client_version=``.
A LiteLLM-shaped ``/v1/model/info`` is provided for agents that probe it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Request

from mom.api.auth import require_api_key
from mom.api.deps import ContainerDep
from mom.config.capabilities import ModelCard, ensemble_card
from mom.domain.errors import UnknownModelError


router = APIRouter(dependencies=[Depends(require_api_key)])


def _cards(container: ContainerDep) -> list[ModelCard]:
    return [
        ensemble_card(name, ens, container.catalog)
        for name, ens in container.catalog.ensembles.items()
    ]


def _supported_params(card: ModelCard) -> list[str]:
    params = ["temperature", "top_p", "max_tokens", "stop", "stream", "seed"]
    if card.supports_tools:
        params += ["tools", "tool_choice", "parallel_tool_calls", "response_format"]
    if card.supports_reasoning:
        params += ["reasoning_effort", "reasoning"]
    if card.supports_web_search:
        params += ["web_search", "web_search_options"]
    return params


def _openai_model(card: ModelCard, created: int) -> dict[str, Any]:
    modalities = ["text", "image"] if card.supports_vision else ["text"]
    obj: dict[str, Any] = {
        "id": card.id,
        "object": "model",
        "created": created,
        "owned_by": "mom",
        "name": card.id,
        "description": card.description,
        "architecture": {
            "modality": "text+image->text" if card.supports_vision else "text->text",
            "input_modalities": modalities,
            "output_modalities": ["text"],
        },
        "supported_parameters": _supported_params(card),
        "reasoning_effort_levels": list(card.effort_levels),
        # Flat capability booleans some OpenAI-compatible clients (e.g. lobe-chat) read directly
        # off each /models list entry instead of a nested vendor block.
        "functionCall": card.supports_tools,
        "vision": card.supports_vision,
        "reasoning": card.supports_reasoning,
        "search": card.supports_web_search,
        "mom": {
            "members": list(card.members),
            "synthesizer": card.synthesizer,
            "supports_tools": card.supports_tools,
            "supports_vision": card.supports_vision,
            "supports_reasoning": card.supports_reasoning,
            "supports_web_search": card.supports_web_search,
            "client_managed_mcp": True,
            "remote_mcp": False,
        },
    }
    if card.context_length is not None:
        obj["context_length"] = card.context_length
        obj["top_provider"] = {
            "context_length": card.context_length,
            "max_completion_tokens": card.max_output_tokens,
            "is_moderated": False,
        }
    return obj


def _is_anthropic(request: Request) -> bool:
    return "anthropic-version" in request.headers or "x-api-key" in request.headers


@router.get("/models")
async def list_models(
    request: Request, container: ContainerDep, client_version: str | None = None
) -> dict[str, Any]:
    # Codex CLI's model-picker refresh calls GET /v1/models?client_version=<v> and decodes its own
    # catalog envelope, {"models": [...]}. Handed OpenAI's list shape it logs `failed to refresh
    # available models: missing field 'models'` and falls back to its bundled metadata. Presence of
    # the parameter (not its value) selects the dialect; absent, every other client gets the
    # unchanged OpenAI shape.
    #
    # The catalog is deliberately *empty*. Codex requires each entry to carry `base_instructions`
    # and uses what it receives verbatim as that model's system prompt, so any entry MoM emitted
    # would replace Codex's own agent prompt rather than describe a model — measured against
    # codex-cli 0.147.0, an entry with `"base_instructions": ""` dropped the whole 20.7 KB prompt
    # from the request. An empty catalog clears the error and leaves Codex on its own metadata.
    if client_version is not None:
        return {"models": []}
    created = int(container.clock.now())
    cards = _cards(container)
    if _is_anthropic(request):
        created_at = datetime.fromtimestamp(created, tz=UTC).isoformat()
        data = [
            {"type": "model", "id": c.id, "display_name": c.id, "created_at": created_at}
            for c in cards
        ]
        return {
            "data": data,
            "has_more": False,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
        }
    return {"object": "list", "data": [_openai_model(c, created) for c in cards]}


@router.get("/models/{model_id}")
async def get_model(model_id: str, container: ContainerDep) -> dict[str, Any]:
    ensemble = container.catalog.ensembles.get(model_id)
    if ensemble is None:
        raise UnknownModelError(f"unknown model {model_id!r}")
    card = ensemble_card(model_id, ensemble, container.catalog)
    return _openai_model(card, int(container.clock.now()))


def _model_info_entry(card: ModelCard) -> dict[str, Any]:
    return {
        "model_name": card.id,
        "litellm_params": {"model": f"mom/{card.id}"},
        "model_info": {
            "max_input_tokens": card.context_length,
            "max_output_tokens": card.max_output_tokens,
            "supports_function_calling": card.supports_tools,
            "supports_vision": card.supports_vision,
            "supports_reasoning": card.supports_reasoning,
            "supports_web_search": card.supports_web_search,
        },
    }


@router.get("/model/info")
async def model_info(container: ContainerDep) -> dict[str, Any]:
    return {"data": [_model_info_entry(card) for card in _cards(container)]}
