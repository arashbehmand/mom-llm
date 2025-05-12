import os
import time
import asyncio
import logging
import uuid # Added
import sys # Added for explicit stdout logging
from dotenv import load_dotenv # Moved up
load_dotenv() # Load .env file at the very beginning

from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from typing import List, Optional, Literal
import litellm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Response

# --- Early and Explicit Logging Configuration ---
# Ensure this runs before other application-level imports if possible,
# though module-level imports like litellm and FastAPI will already have occurred.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Explicitly direct to stdout
    ]
)
logger = logging.getLogger(__name__) # Get logger for this module
logger.info("--- mom_service.main.py: Logging configured ---")

# Control LiteLLM verbosity with an environment variable
# Note: load_dotenv() is called later. If LITELLM_VERBOSE is in .env,
# ensure it's loaded before this point or set it in the shell environment.
LITELLM_VERBOSE_ENV = os.getenv("LITELLM_VERBOSE", "false").lower()
if LITELLM_VERBOSE_ENV in ["true", "1", "yes"]:
    litellm.set_verbose = True
    logger.info("--- mom_service.main.py: LiteLLM verbose logging ENABLED via LITELLM_VERBOSE env var ---")
else:
    litellm.set_verbose = False
    logger.info("--- mom_service.main.py: LiteLLM verbose logging DISABLED ---")

from .config import load_config # Corrected to relative import
from .llm_calls import _call_lite_llm # Import the refactored function

# --- Langfuse Initialization ---
try:
    config = load_config()
    if config.langfuse:
        from langfuse import Langfuse # Import only if configured
        langfuse_public_key = os.getenv(config.langfuse.public_key_env)
        langfuse_secret_key = os.getenv(config.langfuse.secret_key_env)
        langfuse_host = os.getenv(config.langfuse.host_env)

        if langfuse_public_key and langfuse_secret_key and langfuse_host:
            langfuse_client = Langfuse(
                public_key=langfuse_public_key,
                secret_key=langfuse_secret_key,
                host=langfuse_host
            )
            logger.info("--- mom_service.main.py: Langfuse client initialized ---")
        else:
            langfuse_client = None
            logger.warning("--- mom_service.main.py: Langfuse configured but missing environment variables (public_key, secret_key, or host). Langfuse disabled. ---")
    else:
        langfuse_client = None
        logger.info("--- mom_service.main.py: Langfuse not configured. ---")
except Exception as e:
    langfuse_client = None
    logger.error(f"--- mom_service.main.py: Error during Langfuse initialization: {e}. Langfuse disabled. ---")

app = FastAPI()
logger.info("--- mom_service.main.py: FastAPI app initialized ---")

# Pydantic models for OpenAI-compatible error response
class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None

class OpenAIErrorResponse(BaseModel):
    error: ErrorDetail

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    error_type = "api_error" # Default type
    if exc.status_code == 400:
        error_type = "invalid_request_error"
    elif exc.status_code == 401:
        error_type = "authentication_error"
    elif exc.status_code == 403:
        error_type = "permission_error"
    elif exc.status_code == 404:
        error_type = "not_found_error"
    elif exc.status_code == 429:
        error_type = "rate_limit_exceeded_error"
    
    detail_message = exc.detail
    if isinstance(exc.detail, dict) and "message" in exc.detail and "type" in exc.detail:
        # If detail is already structured, use it
        error_detail = ErrorDetail(**exc.detail)
    else:
        # Otherwise, use the string detail as the message
        error_detail = ErrorDetail(message=str(exc.detail), type=error_type)

    return JSONResponse(
        status_code=exc.status_code,
        content=OpenAIErrorResponse(error=error_detail).dict(),
    )

# CORS control: set ALLOWED_CORS_ORIGINS to a comma-separated list of allowed origins
ALLOWED_CORS_ORIGINS = os.getenv("ALLOWED_CORS_ORIGINS", "")

if ALLOWED_CORS_ORIGINS:
    origins = [origin.strip() for origin in ALLOWED_CORS_ORIGINS.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Simple token auth: set API_TOKEN in environment
API_TOKEN = os.getenv("API_TOKEN")

def check_token(request: Request):
    token = request.headers.get("Authorization")
    if API_TOKEN and (not token or token.replace("Bearer ", "") != API_TOKEN):
        raise HTTPException(
            status_code=401, 
            detail={
                "message": "Invalid or missing API token. Ensure 'Authorization: Bearer YOUR_TOKEN' is provided.",
                "type": "authentication_error"
            }
        )

# Pydantic models for OpenAI-compatible API

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

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[UsageInfo] = None

@app.get("/v1/models")
def get_models():
    logger.info("--- /v1/models endpoint HIT ---")
    # Ensure config is loaded inside the function scope
    from .config import load_config
    config = load_config()
    model_data = []
    for model_conf in config.models:
        model_data.append({
            "id": model_conf.name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "MoM-Service",
            "permission": [],
            "root": model_conf.name,
            "parent": None
        })
    if not model_data:
        logger.warning("No models configured in config.yaml for /v1/models endpoint.")
    return {
        "object": "list",
        "data": model_data
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request_data: ChatCompletionRequest, request: Request):
    logger.info("--- chat_completions endpoint HIT ---")
    logger.info(f"Received request for model: {request_data.model}")
    check_token(request)
    # config is loaded globally at startup
    timeout = config.service.timeout_seconds

    # --- Find the requested model configuration ---
    target_model_name = request_data.model
    model_config_to_use: Optional[config.ModelConfig] = None # Type hint from .config
    for mc in config.models:
        if mc.name == target_model_name:
            model_config_to_use = mc
            break
    
    if model_config_to_use is None:
        logger.error(f"Model '{target_model_name}' not found in configuration.")
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"The model `{target_model_name}` does not exist. Please check the available models at /v1/models.",
                "type": "invalid_request_error",
                "param": "model"
            }
        )
    logger.info(f"Using model configuration: {model_config_to_use.name}")

    # --- Create a mapping of LLMDefinition names to their objects for easy lookup ---
    llm_definitions_map = {llm_def.name: llm_def for llm_def in config.llm_definitions}

    trace = None
    if langfuse_client:
        trace_name = f"MoM Request - {request_data.model} - {str(uuid.uuid4())[:8]}"
        trace = langfuse_client.trace(
            name=trace_name,
            user_id=request.headers.get("x-user-id", "anonymous"), # Optional: get user_id from header
            metadata={
                "model_requested": request_data.model,
                "num_messages": len(request_data.messages),
            },
            tags=["mom-service", "phase3"],
            input=request_data.dict() # Set input for the main trace
        )

    # _call_lite_llm function has been moved to llm_calls.py

    # --- Step 1: Asynchronous Fan-out to Multiple LLMs ---
    # Query LLMs specified in the chosen model_config_to_use.
    fanout_llm_names = model_config_to_use.llms_to_query
    logger.info(f"Starting fan-out to {len(fanout_llm_names)} LLMs for model '{model_config_to_use.name}'.")
    fanout_tasks = []
    actual_fanout_llms_configs = []

    for i, llm_name in enumerate(fanout_llm_names):
        llm_def = llm_definitions_map.get(llm_name)
        if not llm_def:
            logger.error(f"LLMDefinition '{llm_name}' (for fan-out in model '{model_config_to_use.name}') not found in llm_definitions. Skipping.")
            # Optionally, could raise an error here if a defined LLM is critical
            continue 
        actual_fanout_llms_configs.append(llm_def)
        langfuse_generation_id = f"fanout-{i}-{llm_def.name}" if trace else None
        fanout_tasks.append(
            _call_lite_llm(
                llm_def, # Use the looked-up LLMDefinition
                [m.dict() for m in request_data.messages],
                timeout,
                call_type="Fan-out",
                trace=trace,
                generation_name=langfuse_generation_id
            )
        )
    
    if not fanout_tasks: # If all specified fanout LLMs were not found or none were specified
        logger.error(f"No valid fan-out LLMs to query for model '{model_config_to_use.name}'.")
        if trace:
            trace.update(level="ERROR", status_message=f"No valid fan-out LLMs for model {model_config_to_use.name}.")
        raise HTTPException(
            status_code=500, 
            detail={
                "message": f"Configuration error: No valid fan-out LLMs found for model '{model_config_to_use.name}'.",
                "type": "internal_server_error"
            }
        )

    fanout_response_objects = await asyncio.gather(*fanout_tasks)
    
    successful_intermediate_answers = []
    for res_obj in fanout_response_objects:
        if res_obj and hasattr(res_obj, 'choices') and res_obj.choices and \
           hasattr(res_obj.choices[0], 'message') and res_obj.choices[0].message and \
           hasattr(res_obj.choices[0].message, 'content'):
            successful_intermediate_answers.append(res_obj.choices[0].message.content)

    logger.info(f"Fan-out for model '{model_config_to_use.name}' complete. Received {len(successful_intermediate_answers)} successful responses out of {len(actual_fanout_llms_configs)} attempted.")

    if not successful_intermediate_answers:
        logger.error(f"All fan-out LLM calls failed for model '{model_config_to_use.name}'.")
        if trace:
            trace.update(level="ERROR", status_message=f"All fan-out LLM calls failed for model '{model_config_to_use.name}'.")
        raise HTTPException(
            status_code=503,
            detail={
                "message": f"All fan-out LLM calls for model '{model_config_to_use.name}' failed. The service could not get a response from any of the intermediary models.",
                "type": "service_unavailable_error"
            }
        )

    # --- Step 2: Prepare Messages for the Concluding LLM ---
    logger.info(f"Preparing messages for Concluding LLM for model '{model_config_to_use.name}'.")
    concluding_messages = [m.dict() for m in request_data.messages]

    concluding_messages.append({"role": "user", "content": "<<<<<<>>>>>>"})
    logger.info("Appended separator message for expert responses.")

    for answer in successful_intermediate_answers:
        concluding_messages.append({"role": "assistant", "content": answer})

    # Look up the prompt content by name if defined
    prompt_content = None
    if hasattr(config, "prompt_definitions") and config.prompt_definitions and model_config_to_use.concluding_prompt:
        for prompt_def in config.prompt_definitions:
            if prompt_def.name == model_config_to_use.concluding_prompt:
                prompt_content = prompt_def.content
                break
    if prompt_content:
        concluding_messages.append({"role": "user", "content": prompt_content})
        logger.info(f"Appended concluding_prompt '{model_config_to_use.concluding_prompt}' from prompt_definitions for model '{model_config_to_use.name}'.")

    # --- Step 3: Call the Concluding LLM ---
    concluding_llm_name = model_config_to_use.concluding_llm
    concluding_llm_def = llm_definitions_map.get(concluding_llm_name)

    if not concluding_llm_def:
        logger.error(f"Concluding LLMDefinition '{concluding_llm_name}' (for model '{model_config_to_use.name}') not found in llm_definitions.")
        if trace:
            trace.update(level="ERROR", status_message=f"Concluding LLMDefinition '{concluding_llm_name}' not found.")
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Configuration error: Concluding LLM '{concluding_llm_name}' for model '{model_config_to_use.name}' not defined.",
                "type": "internal_server_error"
            }
        )
    
    logger.info(f"Calling Concluding LLM: {concluding_llm_def.name} (Model: {concluding_llm_def.model}) for model '{model_config_to_use.name}'.")
    
    concluding_generation_name = f"concluding-{concluding_llm_def.name}" if trace else None
    concluding_llm_response_obj = await _call_lite_llm(
        concluding_llm_def,
        concluding_messages,
        timeout,
        call_type="Concluding",
        trace=trace,
        generation_name=concluding_generation_name
    )

    if concluding_llm_response_obj is None:
        logger.error(f"Concluding LLM call failed for {concluding_llm_def.name} (model={concluding_llm_def.model}) on model '{model_config_to_use.name}'.")
        if trace:
            trace.update(level="ERROR", status_message=f"Concluding LLM call failed for {concluding_llm_def.name} on model '{model_config_to_use.name}'.")
        raise HTTPException(
            status_code=502,
            detail={
                "message": f"The concluding LLM ({concluding_llm_def.name}) call failed for model '{model_config_to_use.name}'. Unable to generate a final response.",
                "type": "bad_gateway_error"
            }
        )

    final_content = ""
    if hasattr(concluding_llm_response_obj, 'choices') and concluding_llm_response_obj.choices and \
       hasattr(concluding_llm_response_obj.choices[0], 'message') and concluding_llm_response_obj.choices[0].message and \
       hasattr(concluding_llm_response_obj.choices[0].message, 'content'):
        final_content = concluding_llm_response_obj.choices[0].message.content
    else:
        logger.error(f"Concluding LLM response object for {concluding_llm_def.name} (model '{model_config_to_use.name}') did not have expected structure.")
        if trace:
            trace.update(level="ERROR", status_message=f"Concluding LLM response for {concluding_llm_def.name} (model '{model_config_to_use.name}') was malformed.")
        raise HTTPException(
            status_code=502,
            detail={
                "message": f"The concluding LLM ({concluding_llm_def.name}) response for model '{model_config_to_use.name}' was malformed.",
                "type": "bad_gateway_error"
            }
        )
        
    usage_info = UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    if hasattr(concluding_llm_response_obj, "usage") and concluding_llm_response_obj.usage is not None:
        usage_data = concluding_llm_response_obj.usage
        usage_info = UsageInfo(
            prompt_tokens=getattr(usage_data, "prompt_tokens", 0),
            completion_tokens=getattr(usage_data, "completion_tokens", 0),
            total_tokens=getattr(usage_data, "total_tokens", 0)
        )
        logger.info(f"Concluding LLM usage for model '{model_config_to_use.name}': {usage_info.dict()}")
    else:
        logger.warning(f"Could not retrieve usage information from Concluding LLM response for {concluding_llm_def.name} (model '{model_config_to_use.name}').")

    # --- Step 4: Format and Return OpenAI-Compatible Response ---
    response_id = f"mom-{model_config_to_use.name}-" + str(uuid.uuid4())
    current_time = int(time.time())
    
    response_payload = ChatCompletionResponse(
        id=response_id,
        created=current_time,
        model=model_config_to_use.name, # Use the requested model name in the response
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=final_content),
                finish_reason="stop" # Assuming stop, could be length if max_tokens is hit
            )
        ],
        usage=usage_info
    )
    logger.info(f"Sending final response. ID: {response_id}")

    if trace:
        # Log the final output to the trace (This trace.generation call is kept as per original logic for "Final Output")
        # Set the output for the main trace
        trace.update(output=response_payload.dict(), usage=litellm.utils.Usage(prompt_tokens=usage_info.prompt_tokens, completion_tokens=usage_info.completion_tokens) if usage_info else None)
        trace.update(level="DEFAULT", status_message="Successfully processed request.") # Mark trace as successful
        # Ensure Langfuse client flushes data before returning response
        # This might be blocking, consider background task if performance critical
        # For now, direct flush for simplicity and reliability of tracing.
        if hasattr(langfuse_client, 'flush') and callable(langfuse_client.flush):
            try:
                langfuse_client.flush()
                logger.info("--- mom_service.main.py: Langfuse data flushed. ---")
            except Exception as e:
                logger.error(f"--- mom_service.main.py: Error flushing Langfuse data: {e} ---")

    return response_payload
