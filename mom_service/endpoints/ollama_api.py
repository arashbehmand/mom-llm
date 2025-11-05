import hashlib
import os
import time
from datetime import datetime, timezone

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..config import load_config
from .models import (
    ChatMessage,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaModelDetails,
    OllamaShowRequest,
    OllamaShowResponse,
    OllamaTagInfo,
    OllamaTagsResponse,
)

config = load_config()
ollama_router = APIRouter(prefix="/ollama", tags=["Ollama"])


def format_ollama_error(status_code: int, detail: any) -> JSONResponse:
    """Convert FastAPI exceptions to Ollama-style error responses."""
    error_message = detail
    if isinstance(detail, dict):
        if "message" in detail:
            error_message = detail["message"]
        elif "error" in detail:
            error_message = detail["error"]
    elif not isinstance(detail, str):
        error_message = str(detail)

    return JSONResponse(status_code=status_code, content={"error": error_message})


# Exception handlers should be added to the main app, not to the router


def check_token(request: Request):
    """Checks the API token for the request."""
    # Get token from environment variable
    service_api_token = os.getenv("API_TOKEN")

    # If no API_TOKEN is set in the environment, access is denied
    if not service_api_token:
        raise HTTPException(
            status_code=503,
            detail={"error": "Service API token not configured by administrator."},
        )

    auth_header = request.headers.get("Authorization", "")

    # Standard check for Bearer token
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid authentication scheme. Use Bearer token."},
        )

    token = auth_header.split(" ", 1)[1]

    if token != service_api_token:
        raise HTTPException(status_code=401, detail={"error": "Invalid or missing API token."})


# Primary chat endpoint
@ollama_router.post("/chat/completions", response_model=OllamaChatResponse)
async def chat_completions_ollama(req_data: OllamaChatRequest, request: Request):
    try:
        check_token(request)
        model_conf_check = next((m for m in config.models if m.name == req_data.model), None)
        if not model_conf_check:
            raise HTTPException(
                status_code=404,
                detail=f"model '{req_data.model}' not found, try pulling it first",
            )

        from ..main import _process_mom_chat_request

        start_time = time.perf_counter_ns()
        try:
            (
                final_content,
                _raw_thinking_ctx,
                concluding_usage,
                _total_cost,
                mom_model_name_used,
                _thinking_embedded,
                trace_obj,  # Added to receive the trace object
            ) = await _process_mom_chat_request(
                req_data.model,
                [m.dict(exclude_none=True) for m in req_data.messages],
                request,
            )
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

        end_time = time.perf_counter_ns()
        total_duration = end_time - start_time

        ollama_response = OllamaChatResponse(
            model=mom_model_name_used,
            created_at=datetime.now(timezone.utc).isoformat(),
            message=ChatMessage(role="assistant", content=final_content),
            done=True,
            total_duration=total_duration,
            load_duration=0,  # MoM doesn't load models in the same way
            prompt_eval_count=(concluding_usage.prompt_tokens if concluding_usage else None),
            prompt_eval_duration=(
                int(total_duration * 0.3) if concluding_usage else None
            ),  # Estimate
            eval_count=concluding_usage.completion_tokens if concluding_usage else None,
            eval_duration=(int(total_duration * 0.7) if concluding_usage else None),  # Estimate
        )

        if trace_obj:
            try:
                trace_obj.update(output=ollama_response.dict(exclude_none=True))
            except Exception as e:
                # Log error if trace update fails, but don't fail the request
                from ..main import logger  # Import logger if not already available

                logger.error(f"Ollama Endpoint: Failed to update Langfuse trace: {e}")

        return ollama_response
    except HTTPException as exc:
        return format_ollama_error(exc.status_code, exc.detail)
    except Exception as e:
        return format_ollama_error(500, f"Unexpected server error: {str(e)}")


# Alias to match common Ollama client path: POST /ollama/api/chat
@ollama_router.post("/api/chat", response_model=OllamaChatResponse)
async def chat_alias(req_data: OllamaChatRequest, request: Request):
    return await chat_completions_ollama(req_data, request)


# Primary models/tags endpoint
@ollama_router.get("/models", response_model=OllamaTagsResponse)
async def get_ollama_models_list(request: Request):
    """Returns a list of available Ollama models (tags)."""
    try:
        check_token(request)

        models_data = []
        for m_config in config.models:
            # Create a unique digest based on the model name
            model_digest = f"sha256:{hashlib.sha256(m_config.name.encode()).hexdigest()}"

            models_data.append(
                OllamaTagInfo(
                    name=f"{m_config.name}:latest",  # Ollama uses name:tag format
                    modified_at=datetime.now(timezone.utc).isoformat(),
                    size=0,  # Size is not applicable for MoM configurations
                    digest=model_digest,
                    details=OllamaModelDetails(),  # Uses defaults from the model
                )
            )
        return OllamaTagsResponse(models=models_data)
    except HTTPException as exc:
        return format_ollama_error(exc.status_code, exc.detail)
    except Exception as e:
        return format_ollama_error(500, f"Unexpected server error: {str(e)}")


# Alias to match common Ollama client path: GET /ollama/api/tags
@ollama_router.get("/api/tags", response_model=OllamaTagsResponse)
async def tags_alias(request: Request):
    return await get_ollama_models_list(request)


@ollama_router.post("/api/show", response_model=OllamaShowResponse)
async def show_ollama_model_details(req_data: OllamaShowRequest, request: Request):
    """Return detailed information about a specific model."""
    try:
        check_token(request)

        # Strip ":latest" or other tags to get base model name
        model_name_base = req_data.name.split(":")[0]

        # Find the model configuration
        model_conf = next((m for m in config.models if m.name == model_name_base), None)
        if not model_conf:
            raise HTTPException(
                status_code=404, detail={"error": f"model '{req_data.name}' not found"}
            )

        # Convert model config to YAML-style string for modelfile
        modelfile_dict = {
            "MoM_Name": model_conf.name,
            "Query_LLMs": model_conf.llms_to_query,
            "Concluding_LLM": model_conf.concluding_llm,
            "Concluding_Prompt": model_conf.concluding_prompt,
            "Include_Thinking_Context": model_conf.include_thinking_context,
        }
        modelfile_str = yaml.dump(modelfile_dict, indent=2, sort_keys=False)

        # Create parameters string listing query and concluding LLMs
        params_list = [f"Query LLM: {q_llm}" for q_llm in model_conf.llms_to_query]
        params_list.append(f"Concluding LLM: {model_conf.concluding_llm}")
        parameters_str = "\n".join(params_list)

        return OllamaShowResponse(
            modelfile=modelfile_str,
            parameters=parameters_str,
            template="MoM: Uses internal prompting; user provides direct input.",
            details=OllamaModelDetails(),  # Uses defaults
            license="MiT",
        )
    except HTTPException as exc:
        return format_ollama_error(exc.status_code, exc.detail)
    except Exception as e:
        return format_ollama_error(500, f"Unexpected server error: {str(e)}")
