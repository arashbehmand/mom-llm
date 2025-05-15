# MoM (Mixture of Models) Service

The MoM Service is a Python-based FastAPI application that acts as a meta-LLM (Large Language Model) service. It receives a request, fans it out to multiple configured LLMs, and then uses a concluding LLM to synthesize their responses into a single, cohesive answer. It aims to provide OpenAI and Ollama compatible API endpoints.

## Features

*   **OpenAI-Compatible API**: Exposes `/v1/chat/completions` and `/v1/models` endpoints for easy integration with existing tools.
*   **Ollama-Compatible API**: Exposes `/ollama/api/chat` endpoint for compatibility with Ollama clients.
*   **LLM Fan-Out**: Queries multiple LLMs simultaneously (defined in `config.yaml`) for a given prompt.
*   **Response Synthesis**: Uses a designated "concluding LLM" to process the responses from the fan-out LLMs and generate a final answer.
*   **Streaming Support**: Supports streaming responses for chat completions.
*   **Configuration-Driven**: LLM providers, models, API keys, prompts, and service behavior are configured through `config.yaml` and environment variables.
*   **Langfuse Integration**: Optional integration with [Langfuse](https://langfuse.com/) for tracing and observability of LLM calls.
*   **Token Authentication**: Secures the API endpoint with a configurable bearer token.
*   **CORS Support**: Allows cross-origin requests from specified domains.
*   **Dockerized**: Comes with a `Dockerfile` for easy containerization and deployment.

## Project Structure

```
.
├── Dockerfile
├── config.yaml
├── config.yaml_template
├── requirements.txt
├── mom_service/
│   ├── __init__.py
│   ├── config.py       # Handles loading config.yaml and defines config models
│   ├── core_logic.py   # Contains the core fan-out and synthesis logic
│   ├── llm_calls.py    # Logic for calling LLMs via LiteLLM
│   ├── main.py         # FastAPI application entry point, middleware, routers
│   └── endpoints/
│       ├── __init__.py
│       ├── models.py   # Defines Pydantic models for API requests/responses and internal data
│       ├── ollama_api.py # Implements Ollama compatible API endpoints
│       └── openai_v1.py  # Implements OpenAI compatible API endpoints
└── README.md
```

## Setup and Running

### Prerequisites

*   Python 3.9+
*   `pip` for installing dependencies.
*   Access to the LLM APIs you intend to use (e.g., OpenAI, Google Gemini, local Ollama instance).

### Environment Variables

Create a `.env` file in the project root or set these environment variables in your deployment environment:

*   **API Keys for LLMs**:
    *   `OPENAI_API_KEY`: Your OpenAI API key (if using OpenAI models).
    *   `GOOGLE_API_KEY`: Your Google API key (if using Gemini models).
    *   *(Add other keys as per `config.yaml`'s `api_key_env` fields)*
*   **Service Configuration**:
    *   `API_TOKEN`: A secret token for authorizing requests to this service.
    *   `ALLOWED_CORS_ORIGINS`: Comma-separated list of allowed CORS origins (e.g., `http://localhost:3000,https://yourfrontend.com`). Leave empty or unset to disable CORS.
    *   `LITELLM_VERBOSE`: Set to `true` to enable verbose logging from LiteLLM. Defaults to `false`.
*   **Langfuse (Optional)**:
    *   `LANGFUSE_PUBLIC_KEY`: Your Langfuse public key.
    *   `LANGFUSE_SECRET_KEY`: Your Langfuse secret key.
    *   `LANGFUSE_HOST`: The Langfuse host URL (e.g., `https://cloud.langfuse.com`).

Refer to `config.yaml_template` for a list of LLM-specific API key environment variables you might need based on your chosen models.

### Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running Locally

Run the FastAPI application using Uvicorn:

```bash
uvicorn mom_service.main:app --host 0.0.0.0 --port 8000 --reload --reload-include "config.yaml"
```

This command includes `--reload-include "config.yaml"` which tells Uvicorn to specifically watch `config.yaml` for changes and reload the application.
The service will be available at `http://localhost:8000`.

## Configuration (`config.yaml`)

The `config.yaml` file defines the core behavior of the service:

*   `llm_definitions`: A list of definitions for individual LLMs that can be used by the service. Each entry specifies:
    *   `name`: A unique identifier for this LLM definition.
    *   `model`: The model string recognized by LiteLLM (e.g., `openai/gpt-4.1`, `gemini/gemini-2.5-flash-preview-04-17`, `ollama/llama3`).
    *   `api_key_env`: The environment variable name holding the API key for this model (if required).
    *   `params` (optional): Additional parameters for the LiteLLM call (e.g., `temperature`, `max_tokens`).
*   `prompt_definitions` (optional): A list of reusable prompt templates that can be referenced by models. Each entry specifies:
    *   `name`: A unique identifier for the prompt.
    *   `content`: The text content of the prompt.
*   `models`: A list of MoM model configurations. Each entry defines how a specific MoM model behaves:
    *   `name`: A unique identifier for this MoM model (e.g., "MoM-Standard").
    *   `llms_to_query`: A list of names (referencing `llm_definitions`) to which the initial request will be fanned out.
    *   `concluding_llm`: The name (referencing `llm_definitions`) of the LLM that synthesizes the responses.
    *   `concluding_prompt` (optional): The name (referencing `prompt_definitions`) of a prompt to append to the messages sent to the concluding LLM.
    *   `include_thinking_context` (optional): Boolean, if true, the responses from the fan-out LLMs will be included in the final response within `<think>` tags. Defaults to `false`.
*   `service`:
    *   `timeout_seconds`: Timeout for individual LLM calls.
    *   `exposed_apis`: A list of API types to expose (e.g., `["openai", "ollama"]`).
*   `langfuse` (optional): Configuration for Langfuse integration, specifying environment variable names for keys and host.

## Docker

### Building the Image

```bash
docker build -t mom-service .
```

### Running the Container

```bash
docker run -d -p 8000:8000 \
  --env-file .env \
  mom-service
```

Make sure your `.env` file contains all necessary environment variables. The service inside the container will be accessible on port 8000 of the host machine.

## API Endpoints

### `POST /v1/chat/completions`

OpenAI-compatible endpoint for chat completions.

**Request Body:** (Matches OpenAI `ChatCompletionRequest`)

```json
{
  "model": "MoM-Standard", // Use the name of a model defined in config.yaml
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_tokens": 50,
  "temperature": 0.7,
  "stream": false // Set to true for streaming response
}
```

**Headers:**

*   `Authorization: Bearer YOUR_API_TOKEN`

**Response Body:** (Matches OpenAI `ChatCompletionResponse`)

Includes optional `thinking_context` if `include_thinking_context` is false for the model, or embedded within the `content` if true. Includes optional `total_cost_usd`.

### `GET /v1/models`

OpenAI-compatible endpoint to list available models defined in `config.yaml`.

**Headers:**

*   `Authorization: Bearer YOUR_API_TOKEN`

### `POST /ollama/api/chat`

Ollama-compatible endpoint for chat completions.

**Request Body:** (Matches Ollama Chat Request)

```json
{
  "model": "MoM-Standard", // Use the name of a model defined in config.yaml
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "options": {
    "temperature": 0.7
  },
  "stream": false // Set to true for streaming response
}
```

**Headers:**

*   `Authorization: Bearer YOUR_API_TOKEN` (Optional, if `API_TOKEN` is set)

**Response Body:** (Matches Ollama Chat Response)

## How it Works

1.  A request is made to `/v1/chat/completions` or `/ollama/api/chat`.
2.  The service authenticates the request using the `API_TOKEN` (for OpenAI endpoint).
3.  The user's messages are fanned out asynchronously to all LLMs defined in the chosen MoM model's `llms_to_query` list in `config.yaml`.
4.  Responses from these LLMs are collected.
5.  The original user messages, along with the collected fan-out LLM responses (and the optional `concluding_prompt`), are sent to the `concluding_llm` defined in the MoM model config.
6.  The response from the `concluding_llm` is formatted as an OpenAI or Ollama compatible chat completion response (including streaming if requested) and returned to the client.
7.  If Langfuse is configured, traces and generations are logged for observability.
