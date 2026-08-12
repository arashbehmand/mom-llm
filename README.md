# 🎭 MoM (Mixture of Models) Service

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/arashbehmand/mom-llm)

> **Transform multiple AI perspectives into superior answers through intelligent synthesis**

MoM Service is an OpenAI-compatible API that revolutionizes LLM usage by orchestrating multiple AI models simultaneously. Instead of relying on a single model's perspective, it queries several LLMs in parallel and synthesizes their responses into a single, superior answer using a dedicated "concluding" model.

Think of it as assembling an expert panel: you get the creativity of GPT-5, the reasoning of Claude Sonnet 4.5, and the versatility of Gemini 2.5 Pro—all combined into one comprehensive response that's more reliable and nuanced than any individual model could produce.

## 🌟 Why a Mixture of Models?

In today's AI landscape with hundreds of specialized LLMs, relying on a single model is limiting. A Mixture of Models (MoM) approach delivers compelling advantages:

![Different AI models offer unique perspectives on the same question](docs/neo-fork.png)
*Each AI model brings its own unique perspective and reasoning style. MoM synthesizes these diverse viewpoints into a more comprehensive answer.*

| Benefit | Description |
|---------|-------------|
| **🎯 Superior Quality** | Synthesize multiple perspectives to mitigate individual model weaknesses (hallucinations, biases, knowledge gaps) |
| **🛡️ Enhanced Reliability** | If one LLM fails or underperforms, others compensate to maintain high-quality output |
| **💰 Cost Optimization** | Route queries strategically—use cost-effective models where appropriate, premium ones when needed |
| **🔄 Maximum Flexibility** | Hot-swap models via configuration without code changes. Create specialized "meta-models" for different tasks |

### Real-World Use Cases

- **📝 Content Creation**: Combine creative and factual models for balanced, engaging content
- **💻 Code Generation**: Merge multiple coding assistants for more robust solutions
- **🔍 Research & Analysis**: Get comprehensive answers by consulting multiple AI "experts"
- **🎓 Educational Applications**: Provide students with well-rounded explanations from diverse perspectives

## 🔄 How It Works

MoM Service uses an elegant **fan-out, fan-in architecture** for parallel processing and intelligent synthesis:

```mermaid
graph TD
    A[Client Request via OpenAI-Compatible API] --> B{MoM Service - FastAPI};
    B --> C[Fan-Out to Multiple LLMs];
    subgraph "Parallel LLM Inference"
        C --> D1[GPT-4o];
        C --> D2[Claude 3.5 Sonnet];
        C --> D3[Gemini 1.5 Pro];
        C --> D4[Llama 3.1 405B];
    end
    subgraph "Response Synthesis"
        D1 --> E{Concluding LLM};
        D2 --> E;
        D3 --> E;
        D4 --> E;
    end
    E --> F[Final Response Streamed to User];

    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#ccf,stroke:#333,stroke-width:2px
    style E fill:#cfc,stroke:#333,stroke-width:2px
```

### Processing Flow

1. **📥 Request In**: Client makes request to OpenAI-compatible endpoint (`/v1/chat/completions`)
2. **🎯 Fan-Out**: Service identifies the MoM configuration and forwards request to all configured LLMs
3. **⚡ Concurrent Processing**: All LLMs process the request simultaneously (non-blocking)
4. **🧠 Synthesize**: Responses collected and passed to the "Concluding LLM"
5. **📤 Stream Response**: Final synthesized answer streamed back to client in real-time

## ✨ Features

- **🔌 OpenAI-Compatible API**: Drop-in replacement with `/v1/chat/completions` and `/v1/models` endpoints
- **🎭 Multi-Model Orchestration**: Query multiple LLMs in parallel with intelligent synthesis
- **🖼️ Multimodal Vision Support**: Send images alongside text using OpenAI Vision API format
- **⚡ Real-Time Streaming**: Stream synthesized responses back to clients with low latency
- **⚙️ Configuration-Driven**: Define everything in a single `config.yaml` file—no code changes needed
- **💰 Advanced Pricing & Cost Tracking**:
  - Custom pricing configurations for reasoning tokens
  - Automatic model filtering based on multimodal capabilities
  - Detailed cost breakdowns with normalized token reporting
  - Per-request cost calculation and logging
- **📊 Advanced Observability**:
  - Built-in Langfuse integration for distributed tracing
  - Comprehensive metrics API with cost tracking and usage analytics (reporting service)
  - Live progress page for each request (when Redis/reporting is enabled)
  - Detailed health check endpoints for monitoring system components
- **🔒 Enterprise Security**:
  - Centralized Bearer token authentication with structured error responses
  - Clear distinction between service misconfiguration (503) and auth failures (401)
  - Flexible CORS policies for cross-origin requests
- **🐳 Production Ready**:
  - Multi-stage Docker builds with non-root users
  - Docker Compose for local development
  - Advanced health checks for orchestration
- **💾 Response Caching**: Automatic LLM response caching to reduce costs and latency
- **🧪 Comprehensive Testing**: Full test suite with pytest for reliability

## 📁 Project Structure

```
mom-llm/
├── 📄 Dockerfile              # Multi-stage Docker build for production
├── 🐳 docker-compose.yml      # Docker Compose for local development
├── ⚙️  config.yaml            # Main configuration (gitignored - use template)
├── 📋 config.yaml_template    # Configuration template with examples
├── 📦 requirements.txt        # Python dependencies
├── 📝 LICENSE                 # MIT License
├── 🔒 .env                    # Environment variables (gitignored)
├── 📂 mom_service/
│   ├── 🎯 main.py            # FastAPI application & middleware
│   ├── 🔒 auth.py            # Authentication & token validation
│   ├── ⚙️  config.py         # Configuration loader & models
│   ├── 🧠 core_logic.py      # Fan-out & synthesis engine
│   ├── 📞 llm_calls.py       # LLM communication via LiteLLM
│   ├── 🖼️  multimodal_utils.py # Multimodal content & message sanitization
│   ├── 💰 cost_calculation.py # Cost tracking with reasoning tokens
│   ├── 💵 pricing_utils.py   # Pricing conversions & normalization
│   ├── 📊 metrics_db.py      # Metrics persistence & analytics
│   ├── 📣 events.py          # Redis event publisher (progress reporting)
│   ├── 🏥 health.py          # Health check utilities
│   └── 📂 endpoints/
│       ├── 📋 models.py      # Pydantic request/response models
│       ├── 🔌 openai_v1.py   # OpenAI-compatible endpoints
│       └── 📈 metrics_api.py # Usage metrics API
│   └── 📂 reporting/
│       ├── 🎯 main.py        # Reporting service app
│       ├── 📈 metrics_api.py # Reporting metrics API
│       ├── 📊 metrics_db.py  # Reporting metrics DB helpers
│       └── 📂 templates/
│           └── 📄 progress.html
└── 📂 tests/
    ├── ⚙️  conftest.py       # Pytest fixtures & configuration
    ├── 🧪 test_config.py     # Configuration tests
    ├── 🧪 test_core_logic.py # Core logic tests
    ├── 🧪 test_llm_calls.py  # LLM integration tests
    ├── 🧪 test_endpoints.py  # API endpoint tests
    └── 🧪 test_health.py     # Health check tests
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerized deployment)
- API keys for your chosen LLM providers (OpenAI, Google Gemini, Anthropic, etc.)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/arashbehmand/mom-llm.git
   cd mom-llm
   ```

2. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   # Service Configuration
   API_TOKEN="your-secret-bearer-token"
   ALLOWED_CORS_ORIGINS=""  # Comma-separated origins, or empty for no CORS
   LITELLM_VERBOSE="false"

   # LLM API Keys (add the ones you need)
   OPENAI_API_KEY="sk-..."
   GOOGLE_API_KEY="..."
   ANTHROPIC_API_KEY="..."

   # Optional: Langfuse for observability
   LANGFUSE_PUBLIC_KEY=""
   LANGFUSE_SECRET_KEY=""
   LANGFUSE_HOST="https://cloud.langfuse.com"

   # Optional: progress reporting
   REDIS_URL="redis://localhost:6379"
   REPORTING_SERVICE_URL="http://localhost:8001"
   ```

3. **Configure your models**
   
   Copy the template and customize:
   
   - macOS/Linux:
     ```bash
     cp config.yaml_template config.yaml
     # Edit config.yaml to define your LLMs and MoM configurations
     ```
   - Windows (PowerShell):
     ```powershell
     Copy-Item config.yaml_template config.yaml
     # Then edit config.yaml to define your LLMs and MoM configurations
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the service**
   ```bash
   uvicorn mom_service.main:app --reload --host 0.0.0.0 --port 8000
   ```

### 🐳 Docker Deployment

**Using Docker Compose (Recommended):**

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f mom-service

# Stop the service
docker-compose down
```

**Using Docker directly:**

```bash
# Build the image
docker build -t mom-service .

# Run the container
docker run -d \
  --name mom-service \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/data:/app/data \
  mom-service
```

### 📝 Basic Usage

**Test the service:**
```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-secret-bearer-token"
```

**Make a chat completion request:**
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
Note: Set "stream": false to get a single JSON response instead of an SSE stream.
If `REPORTING_SERVICE_URL` is configured, responses include `X-MoM-Progress-Url` for live status.

**Send an image (multimodal vision request):**
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

**Note**: Vision requests automatically filter to multimodal-capable models. Non-capable models are skipped, and messages are sanitized for each provider to ensure compatibility.

## ⚙️ Configuration

The service is configured through `config.yaml` and environment variables (`.env` file).

### Quick Configuration Overview

**1. Environment Variables** - API keys and service settings:
```bash
# Required
API_TOKEN="your-secret-bearer-token"

# LLM Provider Keys (add the ones you need)
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="..."
ANTHROPIC_API_KEY="..."
```

**2. Configuration File** - Define your LLMs and MoM models:
```yaml
# Define individual LLMs
llm_definitions:
  - base_name: "gpt4"
    model: "openai/gpt-4.1"
    variants:
      - suffix: "h"
        params:
          reasoning_effort: "high"

  - base_name: "claude"
    model: "anthropic/claude-sonnet-4-5-20250929"

  - base_name: "gemini"
    model: "gemini/gemini-2.5-pro"
    # Optional override only when provider default is not what you want:
    # api_key_env: "CUSTOM_GEMINI_KEY_ENV"

# Define synthesis prompts
prompt_definitions:
  - name: "synth_default"
    content: "Synthesize responses into a cohesive answer..."

# Create MoM models
models:
  - name: "mom"
    llms_to_query: ["gpt4:h", "claude", "gemini"]
    concluding_llm: "gpt4:h"
    concluding_prompt: "synth_default"
```

`suffix` entries resolve to final names like `gpt4:h`. If `api_key_env` is omitted, the service infers it from the provider in `model` (e.g., OpenAI -> `OPENAI_API_KEY`).

For detailed configuration options, custom pricing, advanced features, and complete examples, see the **[Configuration Guide](docs/CONFIGURATION.md)**.

## 🔌 API Reference

The MoM service exposes OpenAI-compatible chat endpoints and health checks. The reporting service exposes progress and metrics endpoints.

### Quick API Overview

**Core Endpoints:**
- `GET /v1/models` - List available MoM models (answers Codex CLI's `?client_version=` model-picker refresh in its own `{"models":[...]}` shape)
- `POST /v1/chat/completions` - Chat completions (streaming and non-streaming)
- `GET /health` - Health check
- `GET http://localhost:8001/progress/{request_id}` - Live request progress page
- `GET http://localhost:8001/v1/metrics/usage` - Usage metrics and cost tracking

**Example Request:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mom",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

For complete API documentation including all endpoints, parameters, response formats, and code examples in multiple languages, see the **[API Reference](docs/API.md)**.

### Using with OpenAI SDK

The service is fully compatible with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-bearer-token"
)

response = client.chat.completions.create(
    model="mom",
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

See the [API Reference](docs/API.md#using-with-openai-sdk) for more examples including non-streaming and multimodal requests.

## 🎯 Advanced Features

### Thinking Context

Set `include_thinking_context: true` in your model configuration to see intermediate responses from all LLMs before synthesis:

```
<think>
Model: gpt-4o
Content: [GPT-4o's response]
---
Model: claude-3-5-sonnet
Content: [Claude's response]
---
</think>

[Final synthesized answer]
```

Useful for understanding synthesis logic, debugging, and transparency.

### Message Sanitization

The service automatically sanitizes messages for provider compatibility, removing empty fields and preserving multimodal content appropriately. This ensures reliable operation across all LLM providers without manual adjustments.

### Cost Tracking & Observability

- **Automatic cost calculation** for every request with detailed breakdowns
- **Langfuse integration** for distributed tracing: Add credentials to `.env` and view detailed traces at [Langfuse](https://langfuse.com/)
- **Metrics API** at `/v1/metrics/usage` for usage analytics

## 🛠️ Development

### Running in Development Mode

```bash
uvicorn mom_service.main:app --reload --reload-include "config.yaml"
```

The `--reload-include` flag watches `config.yaml` for changes and automatically reloads the service.

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check with component validation
curl http://localhost:8000/health/detailed

# Include LLM connectivity test
curl http://localhost:8000/health/detailed?check_llm=true
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=mom_service --cov-report=html

# Run specific test file
pytest tests/test_endpoints.py
```

The test suite includes unit tests, integration tests, API tests, and health check validation.

## 📚 Documentation

For more detailed information, check out these guides:

- **[Configuration Guide](docs/CONFIGURATION.md)** - Comprehensive guide to configuring LLMs, MoM models, and service settings
- **[API Reference](docs/API.md)** - Complete API documentation with examples in multiple languages
- **[Contributing Guide](CONTRIBUTING.md)** - Guidelines for contributors

## 🤝 Contributing

Contributions are welcome! Whether you're fixing bugs, improving documentation, or proposing new features, your help is appreciated.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Setting up your development environment
- Code style and standards
- Running tests and quality checks
- Submitting pull requests
- Reporting issues

Quick start for contributors:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest`)
5. Commit your changes
6. Push to your branch
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- This project was developed with the assistance of multiple AI tools, including **Anthropic's Claude**, **GitHub Copilot**, and **Kilo Code**.
- Built with [FastAPI](https://fastapi.tiangolo.com/) and [LiteLLM](https://github.com/BerriAI/litellm)
- Inspired by ensemble learning and multi-agent AI systems
- Observability powered by [Langfuse](https://langfuse.com/)

## 📬 Contact

**Arash Behmand**
- GitHub: [@arashbehmand](https://github.com/arashbehmand)
- LinkedIn: [linkedin.com/in/arashbehmand](https://linkedin.com/in/arashbehmand)

---

⭐ If you find this project useful, please consider giving it a star on GitHub!
