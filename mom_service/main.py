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
    # Placeholder: mimic OpenAI API /v1/models response
    # See: https://platform.openai.com/docs/api-reference/models/list
    return {
        "object": "list",
        "data": [
            {
                "id": "MoM",
                "object": "model",
                "created": 1747010130,
                "owned_by": "MoM",
                "permission": [],
                "root": "MoM",
                "parent": None
            }
        ]
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request_data: ChatCompletionRequest, request: Request):
    logger.info("--- chat_completions endpoint HIT ---") # Added test log
    logger.info(f"Received request for model: {request_data.model}")
    check_token(request)
    # Config is already loaded for Langfuse initialization, re-use it.
    # config = load_config() # No need to load again
    timeout = config.service.timeout_seconds
    
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
            tags=["mom-service", "phase3"]
        )
        # Log initial user messages as input to the trace
        trace.generation( 
            name="User Input",
            input=request_data.dict(),
            output=None, # No output for user input
            model="user-provided",
            usage=litellm.utils.Usage(prompt_tokens=litellm.token_counter(messages=[msg.dict() for msg in request_data.messages]), completion_tokens=0)
        )


    # _call_lite_llm function has been moved to llm_calls.py

    # --- Step 1: Asynchronous Fan-out to Multiple LLMs ---
    # Query all LLMs defined in the `llms_to_query` section of the config.
    logger.info(f"Starting fan-out to {len(config.llms_to_query)} LLMs.")
    fanout_tasks = []
    for i, llm_cfg in enumerate(config.llms_to_query):
        # Pass trace and an ID for the fanout generation if langfuse is active
        langfuse_generation_id = f"fanout-{i}-{llm_cfg.name}" if trace else None
        fanout_tasks.append(
            _call_lite_llm(
                llm_cfg, 
                [m.dict() for m in request_data.messages], 
                timeout, 
                call_type="Fan-out",
                trace=trace, # Pass trace object
                generation_name=langfuse_generation_id # Pass unique name for this generation
            )
        )
    fanout_response_objects = await asyncio.gather(*fanout_tasks) # Now a list of ModelResponse objects or None
    
    successful_intermediate_answers = []
    # The _call_lite_llm function will now handle creating the Langfuse generation for each fan-out call.
    # We just need to collect the results.
    for res_obj in fanout_response_objects:
        if res_obj and hasattr(res_obj, 'choices') and res_obj.choices and \
           hasattr(res_obj.choices[0], 'message') and res_obj.choices[0].message and \
           hasattr(res_obj.choices[0].message, 'content'):
            successful_intermediate_answers.append(res_obj.choices[0].message.content)

    logger.info(f"Fan-out complete. Received {len(successful_intermediate_answers)} successful responses out of {len(config.llms_to_query)}.")

    if not successful_intermediate_answers:
        logger.error("All fan-out LLM calls failed.")
        if trace:
            trace.update(level="ERROR", status_message="All fan-out LLM calls failed.")
        raise HTTPException(
            status_code=503, 
            detail={
                "message": "All fan-out LLM calls failed. The service could not get a response from any of the intermediary models.",
                "type": "service_unavailable_error"
            }
        )

    # --- Step 2: Prepare Messages for the Concluding LLM ---
    # Aggregate successful intermediate answers and prepare the input for the concluding LLM.
    logger.info("Preparing messages for Concluding LLM.")
    concluding_messages = [m.dict() for m in request_data.messages] # Start with original user messages

    for answer in successful_intermediate_answers:
        concluding_messages.append({"role": "assistant", "content": answer})

    # Add concluding_llm_user_prompt as a user message at the end, if configured
    if config.concluding_llm_user_prompt:
        prompt_content = config.concluding_llm_user_prompt
        concluding_messages.append({"role": "user", "content": prompt_content})
        logger.info(f"Appended concluding_llm_user_prompt as user message.")

    # --- Step 3: Call the Concluding LLM ---
    # Use the aggregated responses to get a final answer from the concluding LLM.
    concluding_cfg = config.concluding_llm
    logger.info(f"Calling Concluding LLM: {concluding_cfg.name} (Model: {concluding_cfg.model})")
    
    # Use _call_lite_llm for the concluding call as well
    concluding_generation_name = f"concluding-{concluding_cfg.name}" if trace else None
    concluding_llm_response_obj = await _call_lite_llm(
        concluding_cfg, 
        concluding_messages, 
        timeout, 
        call_type="Concluding",
        trace=trace, # Pass trace object
        generation_name=concluding_generation_name # Pass unique name for this generation
    )

    if concluding_llm_response_obj is None: # Check if the call failed
        logger.error(f"Concluding LLM call failed for {concluding_cfg.name} (model={concluding_cfg.model}).")
        if trace:
            trace.update(level="ERROR", status_message=f"Concluding LLM call failed for {concluding_cfg.name}")
        raise HTTPException(
            status_code=502,
            detail={
                "message": f"The concluding LLM ({concluding_cfg.name}) call failed. Unable to generate a final response.",
                "type": "bad_gateway_error"
            }
        )

    final_content = ""
    if hasattr(concluding_llm_response_obj, 'choices') and concluding_llm_response_obj.choices and \
       hasattr(concluding_llm_response_obj.choices[0], 'message') and concluding_llm_response_obj.choices[0].message and \
       hasattr(concluding_llm_response_obj.choices[0].message, 'content'):
        final_content = concluding_llm_response_obj.choices[0].message.content
    else:
        logger.error(f"Concluding LLM response object for {concluding_cfg.name} did not have expected structure for content.")
        if trace:
            trace.update(level="ERROR", status_message=f"Concluding LLM response for {concluding_cfg.name} was malformed.")
        raise HTTPException(
            status_code=502,
            detail={
                "message": f"The concluding LLM ({concluding_cfg.name}) response was malformed. Unable to extract final content.",
                "type": "bad_gateway_error"
            }
        )
        
    usage_info = UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0) # Default
    if hasattr(concluding_llm_response_obj, "usage") and concluding_llm_response_obj.usage is not None:
        usage_data = concluding_llm_response_obj.usage
        usage_info = UsageInfo(
            prompt_tokens=getattr(usage_data, "prompt_tokens", 0),
            completion_tokens=getattr(usage_data, "completion_tokens", 0),
            total_tokens=getattr(usage_data, "total_tokens", 0)
        )
        logger.info(f"Concluding LLM usage: {usage_info.dict()}")
    else:
        logger.warning(f"Could not retrieve usage information from Concluding LLM response object for {concluding_cfg.name}.")

    # --- Step 4: Format and Return OpenAI-Compatible Response ---
    response_id = "mom-phase3-" + str(uuid.uuid4()) # Generate a unique ID for the response
    current_time = int(time.time()) # Get current timestamp
    
    response_payload = ChatCompletionResponse(
        id=response_id,
        created=current_time,
        model="mom-service-phase3", # Updated model name
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
        # Log the final output to the trace
        trace.generation(
            name="Final Output",
            input=concluding_messages, # Input to the concluding LLM
            output=response_payload.dict(),
            model=concluding_cfg.model, # Model used for final output
            usage=litellm.utils.Usage(prompt_tokens=usage_info.prompt_tokens, completion_tokens=usage_info.completion_tokens) if usage_info else None,
            metadata={"response_id": response_id}
        )
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
