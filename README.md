# 🎭 MoM (Mixture of Models) Service

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> **Transform multiple AI perspectives into superior answers through intelligent synthesis**

MoM Service is an OpenAI-compatible API that revolutionizes LLM usage by orchestrating multiple AI models simultaneously. Instead of relying on a single model's perspective, it queries several LLMs in parallel and synthesizes their responses into a single, superior answer using a dedicated "concluding" model.

Think of it as assembling an expert panel: you get the creativity of GPT-4, the speed of Gemini Flash, and the reasoning of Claude—all combined into one comprehensive response that's more reliable and nuanced than any individual model could produce.

## 🌟 Why a Mixture of Models?

In today's AI landscape with hundreds of specialized LLMs, relying on a single model is limiting. A Mixture of Models (MoM) approach delivers compelling advantages:

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

```
┌─────────────────┐
│  Client Request │
│  (OpenAI API)   │
└────────┬────────┘
         │
         ▼
┌────────────────────┐
│   MoM Service      │
│   (FastAPI)        │
└────────┬───────────┘
         │ Fan-Out
         ├─────────────────┬─────────────────┬─────────────────┐
         ▼                 ▼                 ▼                 ▼
    ┌─────────┐       ┌─────────┐       ┌─────────┐       ┌─────────┐
    │ GPT-4   │       │ Claude  │       │ Gemini  │       │ Llama   │
    │  (LLM1) │       │  (LLM2) │       │  (LLM3) │       │  (LLM4) │
    └────┬────┘       └────┬────┘       └────┬────┘       └────┬────┘
         │                 │                 │                 │
         └─────────────────┴─────────────────┴─────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Concluding LLM      │
                         │  (Synthesizes All)   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Final Response     │
                         │   (Streamed to User) │
                         └──────────────────────┘
```

### Processing Flow

1. **📥 Request In**: Client makes request to OpenAI-compatible endpoint (`/v1/chat/completions`)
2. **🎯 Fan-Out**: Service identifies the MoM configuration and forwards request to all configured LLMs
3. **⚡ Concurrent Processing**: All LLMs process the request simultaneously (non-blocking)
4. **🧠 Synthesize**: Responses collected and passed to the "Concluding LLM"
5. **📤 Stream Response**: Final synthesized answer streamed back to client in real-time

## ✨ Features

- **🔌 OpenAI-Compatible API**: Drop-in replacement with `/v1/chat/completions` and `/v1/models` endpoints
- **🦙 Ollama-Compatible API**: Full compatibility with Ollama clients via `/ollama/api/chat`
- **🎭 Multi-Model Orchestration**: Query multiple LLMs in parallel with intelligent synthesis
- **⚡ Real-Time Streaming**: Stream synthesized responses back to clients with low latency
- **⚙️ Configuration-Driven**: Define everything in a single `config.yaml` file—no code changes needed
- **📊 Langfuse Integration**: Built-in observability and tracing for production monitoring
- **🔒 Enterprise Security**: Bearer token authentication and flexible CORS policies
- **🐳 Docker Ready**: One-command deployment with included `Dockerfile`
- **💾 Response Caching**: Automatic LLM response caching to reduce costs and latency

## 📁 Project Structure

```
mom-llm/
├── 📄 Dockerfile              # Container configuration for deployment
├── ⚙️  config.yaml            # Main configuration (gitignored - use template)
├── 📋 config.yaml_template    # Configuration template with examples
├── 📦 requirements.txt        # Python dependencies
├── 📝 LICENSE                 # MIT License
├── 🔒 .env                    # Environment variables (gitignored)
└── 📂 mom_service/
    ├── 🎯 main.py            # FastAPI application & middleware
    ├── ⚙️  config.py         # Configuration loader & models
    ├── 🧠 core_logic.py      # Fan-out & synthesis engine
    ├── 📞 llm_calls.py       # LLM communication via LiteLLM
    └── 📂 endpoints/
        ├── 📊 models.py      # Pydantic request/response models
        ├── 🔌 openai_v1.py   # OpenAI-compatible endpoints
        └── 🦙 ollama_api.py  # Ollama-compatible endpoints
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

```bash
# Build the image
docker build -t mom-service .

# Run the container
docker run -d \
  --name mom-service \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/config.yaml:/app/config.yaml \
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
## ⚙️ Configuration

The entire service is configured through `config.yaml`. Here's a breakdown of the key sections:

### LLM Definitions

Define the individual LLMs you want to use:

```yaml
llm_definitions:
  - name: "gpt4"
    model: "openai/gpt-4"
    api_key_env: "OPENAI_API_KEY"
    params:
      temperature: 0.7
  
  - name: "gemini-flash"
    model: "gemini/gemini-2.5-flash-preview-04-17"
    api_key_env: "GOOGLE_API_KEY"
```

### Synthesis Prompts

Customize how the concluding LLM synthesizes responses:

```yaml
prompt_definitions:
  - name: "synth_default"
    content: |
      Review all expert responses and synthesize a single, cohesive answer that:
      - Integrates the strongest insights from each response
      - Resolves any disagreements between models
      - Provides a balanced, comprehensive answer
```

### MoM Models

Create your "meta-models" that define which LLMs to query and how to synthesize:

```yaml
models:
  - name: "mom-creative"
    llms_to_query:
      - "gpt4"
      - "claude"
      - "gemini-flash"
    concluding_llm: "gpt4"
    concluding_prompt: "synth_default"
    include_thinking_context: true  # Show intermediate responses
  
  - name: "mom-fast"
    llms_to_query:
      - "gemini-flash"
      - "gpt-3.5-turbo"
    concluding_llm: "gemini-flash"
    concluding_prompt: "synth_default"
    include_thinking_context: false
```

### Service Settings

```yaml
service:
  timeout_seconds: 30

langfuse:  # Optional observability
  public_key_env: "LANGFUSE_PUBLIC_KEY"
  secret_key_env: "LANGFUSE_SECRET_KEY"
  host_env: "LANGFUSE_HOST"
```

## 🔌 API Reference

### OpenAI-Compatible Endpoints

#### `GET /v1/models`
List all available MoM models defined in your configuration.

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
    }
  ]
}
```

#### `POST /v1/chat/completions`
Send chat completion requests to your MoM models.

**Headers:**
- `Authorization: Bearer YOUR_API_TOKEN`
- `Content-Type: application/json`

**Request:**
```json
{
  "model": "mom-creative",
  "messages": [
    {"role": "user", "content": "Explain machine learning"}
  ],
  "stream": true,
  "temperature": 0.7
}
```

### Ollama-Compatible Endpoint

#### `POST /ollama/api/chat`
Compatible with Ollama client applications.

**Request:**
```json
{
  "model": "mom-creative",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false
}
```

## 🎯 Advanced Features

### Thinking Context

When `include_thinking_context: true`, the service includes intermediate LLM responses in the output, wrapped in `<think>` tags:

```
<think>
Model: gpt-4
Content: [GPT-4's response]
---
Model: claude
Content: [Claude's response]
---
</think>

[Final synthesized answer]
```

This is useful for:
- Understanding how the synthesis was formed
- Debugging model behavior
- Transparency in decision-making

### Cost Tracking

The service automatically tracks and logs API costs for each request using LiteLLM's cost calculation.

### Langfuse Observability

Enable tracing and monitoring:

1. Sign up at [Langfuse](https://langfuse.com/)
2. Add credentials to `.env`
3. View detailed traces for every request in Langfuse dashboard

## 🛠️ Development

### Running in Development Mode

```bash
uvicorn mom_service.main:app --reload --reload-include "config.yaml"
```

The `--reload-include` flag watches `config.yaml` for changes and automatically reloads the service.

### Health Check

Verify the service is up:

```bash
curl http://localhost:8000/health
```

### Testing

Test with curl:
```bash
# Quick health check via models endpoint
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-token"

# Test a simple completion
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mom",
    "messages": [{"role": "user", "content": "Hi!"}],
    "stream": false
  }'
```

### Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-bearer-token"
)

response = client.chat.completions.create(
    model="mom-creative",
    messages=[
        {"role": "user", "content": "What is the meaning of life?"}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## 🤝 Contributing

Contributions are welcome! This project is part of my portfolio, but I'm happy to accept:

- Bug fixes
- Performance improvements
- Documentation enhancements
- New feature suggestions

Please open an issue first to discuss major changes.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) and [LiteLLM](https://github.com/BerriAI/litellm)
- Inspired by ensemble learning and multi-agent AI systems
- Observability powered by [Langfuse](https://langfuse.com/)

## 📬 Contact

**Arash Behmand**
- GitHub: [@arashbehmand](https://github.com/arashbehmand)
- LinkedIn: [linkedin.com/in/arashbehmand](https://linkedin.com/in/arashbehmand)

---

⭐ If you find this project useful, please consider giving it a star on GitHub!
