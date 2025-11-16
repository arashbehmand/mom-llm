# API Reference

This document provides detailed information about all API endpoints available in the MoM Service.

## Table of Contents

- [Authentication](#authentication)
- [OpenAI-Compatible Endpoints](#openai-compatible-endpoints)
  - [List Models](#list-models)
  - [Chat Completions](#chat-completions)
- [Metrics & Observability Endpoints](#metrics--observability-endpoints)
  - [Usage Metrics](#usage-metrics)
  - [Raw Metrics](#raw-metrics)
- [Health Check Endpoints](#health-check-endpoints)
- [Request & Response Examples](#request--response-examples)
- [Using with OpenAI SDK](#using-with-openai-sdk)

## Authentication

All API endpoints require Bearer token authentication.

**Headers:**
```
Authorization: Bearer YOUR_API_TOKEN
```

The API token is configured via the `API_TOKEN` environment variable.

**Error Responses:**

- **401 Unauthorized**: Missing or invalid Bearer token
  ```json
  {
    "detail": "Invalid or missing Bearer token"
  }
  ```

- **503 Service Unavailable**: Service misconfiguration (missing API_TOKEN in environment)
  ```json
  {
    "detail": "Service misconfigured: No API token configured"
  }
  ```

## OpenAI-Compatible Endpoints

### List Models

List all available MoM models defined in your configuration.

**Endpoint:** `GET /v1/models`

**Headers:**
- `Authorization: Bearer YOUR_API_TOKEN`

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "mom-creative",
      "object": "model",
      "created": 1234567890,
      "owned_by": "MoM-Service"
    },
    {
      "id": "mom-fast",
      "object": "model",
      "created": 1234567890,
      "owned_by": "MoM-Service"
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-secret-bearer-token"
```

### Chat Completions

Send chat completion requests to your MoM models.

**Endpoint:** `POST /v1/chat/completions`

**Headers:**
- `Authorization: Bearer YOUR_API_TOKEN`
- `Content-Type: application/json`

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Name of the MoM model to use (e.g., "mom-creative") |
| `messages` | array | Yes | Array of message objects with `role` and `content` |
| `stream` | boolean | No | Whether to stream the response (default: false) |
| `temperature` | float | No | Sampling temperature (0.0 - 2.0) |
| `max_tokens` | integer | No | Maximum tokens to generate |
| `top_p` | float | No | Nucleus sampling parameter |
| `frequency_penalty` | float | No | Frequency penalty (-2.0 - 2.0) |
| `presence_penalty` | float | No | Presence penalty (-2.0 - 2.0) |

**Text-Only Request Example:**
```json
{
  "model": "mom-creative",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Explain quantum computing in simple terms"
    }
  ],
  "stream": true,
  "temperature": 0.7
}
```

**Multimodal Request Example (with images):**
```json
{
  "model": "mom-creative",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What's in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/image.jpg",
            "detail": "high"
          }
        }
      ]
    }
  ],
  "stream": false
}
```

#### Special Instructions for Concluding Agent

You can provide special instructions directly to the concluding agent by embedding a special block in the last user message. This is useful for guiding the final synthesis of the fan-out responses.

The instruction block must be in the following format:

`<<CONCLUDING-INSTRUCTION>>`
Your detailed instructions for the concluding agent go here.
`<</CONCLUDING-INSTRUCTION>>`

This block will be extracted and sent only to the concluding agent. It will be removed from the prompt sent to the fan-out models.

**Example:**

```json
{
  "model": "mom-creative",
  "messages": [
    {
      "role": "user",
      "content": "What are the pros and cons of React vs Vue? <<CONCLUDING-INSTRUCTION>> Please synthesize the results into a markdown table. <</CONCLUDING-INSTRUCTION>>"
    }
  ]
}
```

**Response (Non-Streaming):**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "mom-creative",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing is a revolutionary approach..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 150,
    "total_tokens": 165
  }
}
```

**Response (Streaming):**

Streaming responses use Server-Sent Events (SSE) format:

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"mom-creative","choices":[{"index":0,"delta":{"role":"assistant","content":"Quantum"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"mom-creative","choices":[{"index":0,"delta":{"content":" computing"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"mom-creative","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**cURL Examples:**

**Non-streaming request:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-bearer-token" \
  -d '{
    "model": "mom",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    "stream": false
  }'
```

**Streaming request:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-bearer-token" \
  -d '{
    "model": "mom",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    "stream": true
  }'
```

**Multimodal vision request:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-bearer-token" \
  -d '{
    "model": "mom",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What'\''s in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/image.jpg",
              "detail": "high"
            }
          }
        ]
      }
    ],
    "stream": false
  }'
```

**Important Notes:**

- **Multimodal Support**: Vision requests automatically filter to multimodal-capable models (GPT-4o, Claude 3.5 Sonnet, Gemini 2.5 Pro, etc.). Non-capable models are skipped automatically.
- **Message Sanitization**: Messages are automatically sanitized for each provider to ensure compatibility with strict LLM providers like Mistral.
- **Thinking Context**: If `include_thinking_context: true` is set in your model configuration, intermediate LLM responses will be included in the output wrapped in `<think>` tags.

## Metrics & Observability Endpoints

### Usage Metrics

Get aggregated usage metrics with cost tracking.

**Endpoint:** `GET /v1/metrics/usage`

**Headers:**
- `Authorization: Bearer YOUR_API_TOKEN`

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | string | No | Filter by MoM model name |
| `start_time` | float | No | Unix timestamp for time range filtering |
| `end_time` | float | No | Unix timestamp for time range filtering |
| `call_type` | string | No | Filter by call type (fanout, concluding) |

**Response:**
```json
{
  "status": "ok",
  "filters": {
    "model_name": "mom-creative",
    "start_time": 1640000000.0,
    "end_time": null,
    "call_type": null
  },
  "metrics": {
    "total_requests": 150,
    "total_cost": 12.45,
    "total_tokens": 125000,
    "cache_hit_rate": 0.35,
    "by_model": {
      "mom-creative": {
        "requests": 150,
        "cost": 12.45,
        "tokens": 125000,
        "by_llm": {
          "gpt4": {
            "requests": 50,
            "cost": 5.20,
            "tokens": 45000
          },
          "claude": {
            "requests": 50,
            "cost": 4.15,
            "tokens": 40000
          },
          "gemini": {
            "requests": 50,
            "cost": 3.10,
            "tokens": 40000
          }
        }
      }
    },
    "by_call_type": {
      "fanout": {
        "requests": 450,
        "cost": 9.45,
        "tokens": 95000
      },
      "concluding": {
        "requests": 150,
        "cost": 3.00,
        "tokens": 30000
      }
    }
  }
}
```

**Example:**
```bash
# Get all metrics
curl http://localhost:8000/v1/metrics/usage \
  -H "Authorization: Bearer your-token"

# Get metrics for a specific model
curl "http://localhost:8000/v1/metrics/usage?model_name=mom-creative" \
  -H "Authorization: Bearer your-token"

# Get metrics for a time range
curl "http://localhost:8000/v1/metrics/usage?start_time=1640000000&end_time=1650000000" \
  -H "Authorization: Bearer your-token"
```

### Raw Metrics

Get raw metric records for detailed analysis.

**Endpoint:** `GET /v1/metrics/usage/raw`

**Headers:**
- `Authorization: Bearer YOUR_API_TOKEN`

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | string | No | Filter by MoM model name |
| `start_time` | float | No | Unix timestamp for time range filtering |
| `end_time` | float | No | Unix timestamp for time range filtering |
| `call_type` | string | No | Filter by call type (fanout, concluding) |
| `limit` | integer | No | Maximum records to return (default: 100) |

**Response:**
```json
{
  "status": "ok",
  "count": 50,
  "filters": {
    "model_name": null,
    "start_time": null,
    "end_time": null,
    "call_type": null,
    "limit": 100
  },
  "records": [
    {
      "timestamp": 1640000000.0,
      "request_id": "550e8400-e29b-41d4-a716-446655440000",
      "mom_model_name": "mom-creative",
      "llm_name": "gpt4",
      "call_type": "fanout",
      "prompt_tokens": 100,
      "completion_tokens": 200,
      "reasoning_tokens": 0,
      "total_tokens": 300,
      "cost": 0.015,
      "duration_ms": 1234.5,
      "status": "SUCCESS",
      "cache_hit": false,
      "error_message": null
    }
  ]
}
```

**Example:**
```bash
# Get recent raw metrics
curl http://localhost:8000/v1/metrics/usage/raw \
  -H "Authorization: Bearer your-token"

# Get raw metrics with limit
curl "http://localhost:8000/v1/metrics/usage/raw?limit=50" \
  -H "Authorization: Bearer your-token"
```

## Health Check Endpoints

### Basic Health Check

Fast health check to verify the service is running.

**Endpoint:** `GET /health`

**No authentication required.**

**Response:**
```json
{
  "status": "ok"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

### Detailed Health Check

Comprehensive health check including component validation.

**Endpoint:** `GET /health/detailed`

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `check_llm` | boolean | No | Test LLM connectivity (default: false) |

**No authentication required.**

**Response:**
```json
{
  "status": "ok",
  "components": {
    "cache_db": "ok",
    "metrics_db": "ok",
    "config": "ok",
    "llm_connectivity": "ok"
  },
  "timestamp": 1640000000.0
}
```

If any component fails:
```json
{
  "status": "degraded",
  "components": {
    "cache_db": "ok",
    "metrics_db": "error",
    "config": "ok",
    "llm_connectivity": "not_checked"
  },
  "errors": {
    "metrics_db": "Connection failed: database locked"
  },
  "timestamp": 1640000000.0
}
```

**Examples:**
```bash
# Basic detailed health check
curl http://localhost:8000/health/detailed

# Health check with LLM connectivity test
curl "http://localhost:8000/health/detailed?check_llm=true"
```

## Request & Response Examples

### Python with requests

**Non-streaming:**
```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Authorization": "Bearer your-secret-bearer-token",
    "Content-Type": "application/json"
}
data = {
    "model": "mom-creative",
    "messages": [
        {"role": "user", "content": "What is machine learning?"}
    ],
    "stream": False
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

**Streaming:**
```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Authorization": "Bearer your-secret-bearer-token",
    "Content-Type": "application/json"
}
data = {
    "model": "mom-creative",
    "messages": [
        {"role": "user", "content": "What is machine learning?"}
    ],
    "stream": True
}

response = requests.post(url, headers=headers, json=data, stream=True)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            line = line[6:]  # Remove 'data: ' prefix
            if line == '[DONE]':
                break
            try:
                chunk = json.loads(line)
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    print(delta['content'], end='', flush=True)
            except json.JSONDecodeError:
                pass
```

## Using with OpenAI SDK

The MoM Service is fully compatible with the OpenAI Python SDK.

**Installation:**
```bash
pip install openai
```

**Non-streaming example:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-bearer-token"
)

response = client.chat.completions.create(
    model="mom-creative",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the meaning of life?"}
    ]
)

print(response.choices[0].message.content)
```

**Streaming example:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-bearer-token"
)

stream = client.chat.completions.create(
    model="mom-creative",
    messages=[
        {"role": "user", "content": "Tell me a story about a robot"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**Multimodal example:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-bearer-token"
)

response = client.chat.completions.create(
    model="mom-creative",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "high"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

**List models example:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-bearer-token"
)

models = client.models.list()
for model in models.data:
    print(f"Model ID: {model.id}")
```
