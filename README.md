# MoM (Mixture of Models) Service

The MoM Service is a Python-based FastAPI application that acts as a meta-LLM (Large Language Model) service. It receives a request, fans it out to multiple configured LLMs, and then uses a concluding LLM to synthesize their responses into a single, cohesive answer. It aims to provide an OpenAI-compatible API endpoint.

## Features

*   **OpenAI-Compatible API**: Exposes `/v1/chat/completions` and `/v1/models` endpoints for easy integration with existing tools.
*   **LLM Fan-Out**: Queries multiple LLMs simultaneously (defined in `config.yaml`) for a given prompt.
*   **Response Synthesis**: Uses a designated "concluding LLM" to process the responses from the fan-out LLMs and generate a final answer.
*   **Configuration-Driven**: LLM providers, models, API keys, and service behavior are configured through `config.yaml` and environment variables.
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
│   ├── config.py       # Handles loading config.yaml
│   ├── llm_calls.py    # Logic for calling LLMs via LiteLLM
│   └── main.py         # FastAPI application, API endpoints, core orchestration
└── README.md
```

## Setup and Running

### Prerequisites

*   Python 3.9+
*   `pip` for installing dependencies.
*   Access to the LLM APIs you intend to use (e.g., OpenAI, Google Gemini).

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
uvicorn mom_service.main:app --host 0.0.0.0 --port 8000 --reload
```

The service will be available at `http://localhost:8000`.

## Configuration (`config.yaml`)

The `config.yaml` file defines the core behavior of the service:

*   `llms_to_query`: A list of LLMs to which the initial request will be fanned out. Each entry specifies:
    *   `name`: A unique identifier for the LLM configuration.
    *   `model`: The model string recognized by LiteLLM (e.g., `openai/gpt-4.1`, `gemini/gemini-2.5-flash-preview-04-17`).
    *   `api_key_env`: The environment variable name holding the API key for this model.
    *   `params` (optional): Additional parameters for the LiteLLM call (e.g., `temperature`, `max_tokens`).
*   `concluding_llm`: Configuration for the LLM that synthesizes the responses. It has the same structure as an entry in `llms_to_query`.
*   `service`:
    *   `timeout_seconds`: Timeout for individual LLM calls.
*   `concluding_llm_user_prompt` (optional): A string that will be appended as a user message to the concluding LLM, guiding its synthesis process.
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
  "model": "MoM", // Or any string, actual models are from config.yaml
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_tokens": 50,
  "temperature": 0.7
}
```

**Headers:**

*   `Authorization: Bearer YOUR_API_TOKEN`

**Response Body:** (Matches OpenAI `ChatCompletionResponse`)

### `GET /v1/models`

OpenAI-compatible endpoint to list available models. Currently returns a placeholder indicating the "MoM" service.

**Headers:**

*   `Authorization: Bearer YOUR_API_TOKEN`

## How it Works

1.  A request is made to `/v1/chat/completions`.
2.  The service authenticates the request using the `API_TOKEN`.
3.  The user's messages are fanned out asynchronously to all LLMs defined in `llms_to_query` in `config.yaml`.
4.  Responses from these LLMs are collected.
5.  The original user messages, along with the collected fan-out LLM responses (and the optional `concluding_llm_user_prompt`), are sent to the `concluding_llm`.
6.  The response from the `concluding_llm` is formatted as an OpenAI-compatible chat completion response and returned to the client.
7.  If Langfuse is configured, traces and generations are logged for observability.
