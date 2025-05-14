import asyncio
import logging
import os
import sys
import time
import uuid
from typing import List, Literal, Optional

import litellm
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import load_config
from .llm_calls import _call_lite_llm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info("--- mom_service.main.py: Logging configured ---")

LITELLM_VERBOSE_ENV = os.getenv("LITELLM_VERBOSE", "false").lower()
litellm.set_verbose = LITELLM_VERBOSE_ENV in ("true", "1", "yes")
logger.info(
    f"--- LiteLLM verbose logging {'ENABLED' if litellm.set_verbose else 'DISABLED'} ---"
)

# Load configuration
try:
    config = load_config()
except Exception as e:
    logger.error(f"Error loading config: {e}")
    raise

# Initialize Langfuse if configured
langfuse_client = None
if config.langfuse:
    try:
        from langfuse import Langfuse

        pub = os.getenv(config.langfuse.public_key_env)
        sec = os.getenv(config.langfuse.secret_key_env)
        host = os.getenv(config.langfuse.host_env)
        if pub and sec and host:
            langfuse_client = Langfuse(public_key=pub, secret_key=sec, host=host)
            logger.info("--- Langfuse client initialized ---")
        else:
            logger.warning("--- Langfuse configured but missing env vars ---")
    except Exception as e:
        logger.error(f"Langfuse init error: {e}")

app = FastAPI()
logger.info("--- FastAPI app initialized ---")


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    error: ErrorDetail


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    mapping = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_exceeded_error",
    }
    typ = mapping.get(exc.status_code, "api_error")
    detail = (
        exc.detail
        if isinstance(exc.detail, dict)
        else {"message": str(exc.detail), "type": typ}
    )
    error_detail = ErrorDetail(**detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=OpenAIErrorResponse(error=error_detail).dict(),
    )


ALLOWED_CORS_ORIGINS = os.getenv("ALLOWED_CORS_ORIGINS", "")
if ALLOWED_CORS_ORIGINS:
    origins = [o.strip() for o in ALLOWED_CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

API_TOKEN = os.getenv("API_TOKEN")


def check_token(request: Request):
    token = request.headers.get("Authorization", "")
    if API_TOKEN and token.replace("Bearer ", "") != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail={
                "message": "Invalid or missing API token.",
                "type": "authentication_error",
            },
        )


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[List[str]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None  # Cost for this specific usage


class ThinkingContextItem(BaseModel):
    model: str  # The actual model used, e.g. "gpt-4"
    content: str
    usage: UsageInfo
    # cost is already in usage


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str  # This would be the 'mom' model name (e.g., "mom-default")
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[UsageInfo] = None  # Usage of the *concluding* LLM
    thinking_context: Optional[List[ThinkingContextItem]] = None  # Custom field
    total_cost_usd: Optional[float] = None  # Custom field for aggregated cost


@app.get("/v1/models")
def get_models():
    data = []
    for m in config.models:
        data.append(
            {
                "id": m.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "MoM-Service",
                "permission": [],
                "root": m.name,
                "parent": None,
            }
        )
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request_data: ChatCompletionRequest, request: Request):
    logger.info("--- chat_completions endpoint HIT ---")
    check_token(request)
    timeout = config.service.timeout_seconds

    model_conf = next((m for m in config.models if m.name == request_data.model), None)
    if not model_conf:
        logger.error(f"Model {request_data.model} not found.")
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Model {request_data.model} not found.",
                "type": "invalid_request_error",
            },
        )
    logger.info(f"Using model config: {model_conf.name}")

    llm_map = {ld.name: ld for ld in config.llm_definitions}
    total_cost_accumulator = 0.0

    trace = None
    if langfuse_client:
        trace = langfuse_client.trace(
            name=f"MoM-{model_conf.name}-{str(uuid.uuid4())[:8]}",
            user_id=request.headers.get("x-user-id", "anon"),
            metadata={
                "model_requested": model_conf.name,
                "num_messages": len(request_data.messages),
            },
            input=request_data.dict(),
        )

    # Step 1: Fan-out
    tasks = []
    fanout_llm_defs_in_order = []
    for idx, llm_name in enumerate(model_conf.llms_to_query):
        ld = llm_map.get(llm_name)
        if not ld:
            logger.warning(
                f"LLMDefinition '{llm_name}' for fan-out not found. Skipping."
            )
            continue
        fanout_llm_defs_in_order.append(ld)
        gen_name = f"fanout-{idx}-{ld.name}" if trace else None
        tasks.append(
            _call_lite_llm(
                ld,
                [m.dict() for m in request_data.messages],
                timeout,
                call_type="fanout",
                trace=trace,
                generation_name=gen_name,
            )
        )

    fanout_responses = await asyncio.gather(*tasks, return_exceptions=True)

    intermediate_thinking_context = []
    for ld, res_obj in zip(fanout_llm_defs_in_order, fanout_responses):
        cost = None  # Initialize cost to None
        if isinstance(res_obj, Exception):
            logger.error(f"Fan-out call to {ld.model} failed: {res_obj}")
            # Optionally, create a ThinkingContextItem with error info
            # For now, we just skip it for cost calculation and context
            content = f"Error: Call to {ld.model} failed. Details: {str(res_obj)}"
            usage_data_error = UsageInfo(
                prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0
            )
            intermediate_thinking_context.append(
                ThinkingContextItem(
                    model=ld.model, content=content, usage=usage_data_error
                )
            )
            continue

        if res_obj and res_obj.choices and res_obj.choices[0].message.content:
            try:
                cost = litellm.completion_cost(completion_response=res_obj)
                if cost is not None:
                    total_cost_accumulator += cost
            except Exception as e:
                logger.warning(
                    f"Could not calculate cost for {ld.model}: {e}. Cost will be omitted for this item."
                )
                cost = None  # Ensure cost is None if calculation fails

            usage_data = res_obj.usage or {}
            current_usage_info = UsageInfo(
                prompt_tokens=getattr(usage_data, "prompt_tokens", 0),
                completion_tokens=getattr(usage_data, "completion_tokens", 0),
                total_tokens=getattr(usage_data, "total_tokens", 0),
                cost=cost,  # This will be None if cost calculation failed
            )
            intermediate_thinking_context.append(
                ThinkingContextItem(
                    model=ld.model,
                    content=res_obj.choices[0].message.content,
                    usage=current_usage_info,
                )
            )
        else:  # Handle cases where res_obj is None or malformed, though _call_lite_llm should prevent this
            logger.warning(
                f"Fan-out call to {ld.model} returned an unexpected or empty response."
            )
            content = (
                f"Warning: Call to {ld.model} returned an empty or malformed response."
            )
            usage_data_empty = UsageInfo(
                prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0
            )
            intermediate_thinking_context.append(
                ThinkingContextItem(
                    model=ld.model, content=content, usage=usage_data_empty
                )
            )

    if not any(
        item.content
        and not item.content.startswith("Error:")
        and not item.content.startswith("Warning:")
        for item in intermediate_thinking_context
    ):
        logger.error("No successful fan-out responses with content.")
        if trace:
            trace.update(
                level="ERROR",
                status_message="All fan-out calls failed or returned no content.",
            )
        # We might still proceed if some calls failed but others succeeded,
        # but if ALL failed or returned no usable content, then raise.
        # The check above ensures at least one non-error/non-warning item exists.
        # If intermediate_thinking_context is empty or all are errors/warnings, then it's an issue.
        # A more robust check might be needed depending on desired behavior for partial failures.
        # For now, if intermediate_thinking_context is populated but all are errors, it will proceed.
        # Let's refine this: if NO successful content, then raise.
        successful_fanout_count = sum(
            1 for item in intermediate_thinking_context if item.usage.cost is not None
        )  # A proxy for success
        if successful_fanout_count == 0:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "All fan-out LLM calls failed or pricing is unavailable for all.",
                    "type": "service_unavailable_error",
                },
            )

    # Step 2: Prepare concluding messages
    concl_msgs = [m.dict() for m in request_data.messages]
    concl_msgs.append({"role": "user", "content": "<<<<<<>>>>>>"})
    for item in intermediate_thinking_context:
        # Only append content from successful calls to the concluding prompt
        if (
            item.usage.cost is not None
            and not item.content.startswith("Error:")
            and not item.content.startswith("Warning:")
        ):
            concl_msgs.append({"role": "assistant", "content": item.content})

    if (
        model_conf.concluding_prompt and config.prompt_definitions is not None
    ):  # Added 'is not None' check
        pd = next(
            (
                p
                for p in config.prompt_definitions
                if p.name == model_conf.concluding_prompt
            ),
            None,
        )
        if pd:
            concl_msgs.append({"role": "user", "content": pd.content})

    # Step 3: Concluding LLM
    concl_def = llm_map.get(model_conf.concluding_llm)
    if not concl_def:
        logger.error(
            f"Concluding LLMDefinition '{model_conf.concluding_llm}' not found."
        )
        if trace:
            trace.update(
                level="ERROR",
                status_message=f"Concluding LLMDef '{model_conf.concluding_llm}' not found.",
            )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Concluding LLM definition not found.",
                "type": "internal_server_error",
            },
        )

    gen_name = f"concluding-{concl_def.name}" if trace else None
    concl_res_obj = await _call_lite_llm(
        concl_def,
        concl_msgs,
        timeout,
        call_type="concluding",
        trace=trace,
        generation_name=gen_name,
    )

    if (
        not concl_res_obj
        or not concl_res_obj.choices
        or not concl_res_obj.choices[0].message.content
    ):
        logger.error("Concluding LLM call failed or returned empty content.")
        if trace:
            trace.update(level="ERROR", status_message="Concluding LLM call failed.")
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Concluding LLM failed to generate response.",
                "type": "bad_gateway_error",
            },
        )

    final_content_from_concluding_llm = concl_res_obj.choices[0].message.content
    # Default to current behavior for the separate thinking_context field
    processed_thinking_context_for_response_field = intermediate_thinking_context

    if model_conf.include_thinking_context:
        thinking_steps_str_parts = []
        if intermediate_thinking_context:  # Ensure it's not None
            for item in intermediate_thinking_context:
                # Consider html.escape(item.content) if content can have XML special chars
                thinking_steps_str_parts.append(
                    f"Model: {item.model}\nContent: {item.content}"
                )

        if thinking_steps_str_parts:
            full_thinking_block = (
                "<think>\n" + "\n---\n".join(thinking_steps_str_parts) + "\n</think>"
            )
            final_content = (
                f"{full_thinking_block}\n{final_content_from_concluding_llm}"
            )
        else:
            # No thinking steps to embed, use original content
            final_content = final_content_from_concluding_llm
        # If thinking context is embedded, do not populate the separate field
        processed_thinking_context_for_response_field = None
    else:
        # If not embedding, use original content and keep intermediate_thinking_context for the separate field
        final_content = final_content_from_concluding_llm
        # processed_thinking_context_for_response_field is already correctly set

    concluding_llm_cost = None
    try:
        concluding_llm_cost = litellm.completion_cost(completion_response=concl_res_obj)
        if concluding_llm_cost is not None:
            total_cost_accumulator += concluding_llm_cost
    except Exception as e:
        logger.warning(
            f"Could not calculate cost for concluding LLM {concl_def.model}: {e}. Cost will be omitted."
        )
        concluding_llm_cost = None

    concluding_usage_data = concl_res_obj.usage or {}
    concluding_usage_info = UsageInfo(
        prompt_tokens=getattr(concluding_usage_data, "prompt_tokens", 0),
        completion_tokens=getattr(concluding_usage_data, "completion_tokens", 0),
        total_tokens=getattr(concluding_usage_data, "total_tokens", 0),
        cost=concluding_llm_cost,
    )

    # Step 4: Build response
    response_id = f"mom-{model_conf.name}-{str(uuid.uuid4())}"
    final_response = ChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=model_conf.name,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=final_content),
                finish_reason="stop",
            )
        ],
        usage=concluding_usage_info,
        thinking_context=processed_thinking_context_for_response_field,
        total_cost_usd=(
            round(total_cost_accumulator, 8) if total_cost_accumulator > 0 else None
        ),
    )

    if trace:
        trace.update(
            output=final_response.dict(),
            status_message="Successfully processed request.",
        )
        try:
            langfuse_client.flush()
        except Exception as e:
            logger.error(f"Error flushing Langfuse data: {e}")

    return final_response
