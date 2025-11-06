# MoM Service Architecture

This document provides a detailed technical overview of the MoM (Mixture of Models) Service architecture, including system components, data flow, and design decisions.

## 📊 High-Level Architecture

```mermaid
graph TB
    Client[Client Application<br/>OpenAI SDK / curl / etc.]

    subgraph "MoM Service Container"
        API[FastAPI Application]
        Auth[Authentication Middleware]
        ReqID[Request ID Middleware]

        subgraph "Endpoints"
            OpenAI[OpenAI v1 API]
            Metrics[Metrics API]
            Health[Health Checks]
        end

        subgraph "Core Logic"
            Config[Configuration Loader]
            Fanout[Fanout Engine]
            Concluding[Concluding Engine]
        end

        subgraph "Data Layer"
            Cache[(SQLite Cache DB)]
            MetricsDB[(SQLite Metrics DB)]
        end

        LLMCalls[LLM Call Handler<br/>with Retry & Caching]
    end

    subgraph "External Services"
        OpenAIAPI[OpenAI API]
        AnthropicAPI[Anthropic API]
        GoogleAPI[Google Gemini API]
        OtherLLMs[Other LLM Providers]
        Langfuse[Langfuse<br/>Observability]
    end

    Client -->|HTTP Request| Auth
    Auth -->|Authenticated| ReqID
    ReqID --> API

    API --> OpenAI
    API --> Metrics
    API --> Health

    OpenAI --> Config
    Config --> Fanout
    Fanout --> LLMCalls
    LLMCalls -->|Parallel Calls| Concluding
    Concluding --> LLMCalls

    LLMCalls -.->|Check Cache| Cache
    LLMCalls -.->|Store Response| Cache
    LLMCalls -.->|Record Metrics| MetricsDB

    LLMCalls -->|API Calls| OpenAIAPI
    LLMCalls -->|API Calls| AnthropicAPI
    LLMCalls -->|API Calls| GoogleAPI
    LLMCalls -->|API Calls| OtherLLMs

    LLMCalls -.->|Trace Events| Langfuse

    Metrics -->|Query| MetricsDB
    Health -->|Check| Cache
    Health -->|Check| MetricsDB
    Health -.->|Test| LLMCalls
```

## 🔄 Request Flow Diagram

### Chat Completion Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Auth
    participant Fanout
    participant LLMCalls
    participant LLM1 as GPT-4
    participant LLM2 as Claude
    participant LLM3 as Gemini
    participant Concluding as Concluding LLM
    participant Cache
    participant Metrics
    participant Langfuse

    Client->>FastAPI: POST /v1/chat/completions
    FastAPI->>Auth: Validate Bearer Token
    Auth->>FastAPI: Token Valid

    FastAPI->>Langfuse: Create Trace

    FastAPI->>Fanout: Process Request

    par Parallel LLM Calls
        Fanout->>LLMCalls: Call GPT-4
        LLMCalls->>Cache: Check Cache
        Cache-->>LLMCalls: Cache Miss
        LLMCalls->>LLM1: API Request
        LLM1-->>LLMCalls: Response
        LLMCalls->>Cache: Store Response
        LLMCalls->>Metrics: Record Usage
        LLMCalls->>Langfuse: Log Generation
    and
        Fanout->>LLMCalls: Call Claude
        LLMCalls->>Cache: Check Cache
        Cache-->>LLMCalls: Cache Miss
        LLMCalls->>LLM2: API Request
        LLM2-->>LLMCalls: Response
        LLMCalls->>Cache: Store Response
        LLMCalls->>Metrics: Record Usage
        LLMCalls->>Langfuse: Log Generation
    and
        Fanout->>LLMCalls: Call Gemini
        LLMCalls->>Cache: Check Cache
        Cache-->>LLMCalls: Cache Hit!
        LLMCalls->>Metrics: Record Cache Hit
        LLMCalls->>Langfuse: Log Generation
    end

    Fanout->>Concluding: Synthesize Responses
    Concluding->>LLMCalls: Call Concluding LLM
    LLMCalls->>LLM1: API Request (Synthesis)
    LLM1-->>LLMCalls: Final Response
    LLMCalls->>Metrics: Record Usage
    LLMCalls->>Langfuse: Log Generation

    Concluding->>FastAPI: Stream Response
    FastAPI->>Langfuse: Complete Trace
    FastAPI->>Client: Stream SSE Response
```

## 🏗️ Component Architecture

### 1. API Layer

#### FastAPI Application (`main.py`)
- **Responsibility**: HTTP server, routing, middleware orchestration
- **Key Features**:
  - Request ID generation and tracking (contextvars)
  - CORS configuration
  - Exception handling with detailed logging
  - Health check endpoints
  - Startup configuration loading

#### Authentication (`auth.py`)
- **Responsibility**: Bearer token validation for API endpoints
- **Key Features**:
  - Reusable FastAPI dependency function (`verify_bearer_token`)
  - Validates API_TOKEN environment variable configuration
  - Checks Authorization header for Bearer scheme
  - Returns structured error responses (503 for misconfiguration, 401 for auth failures)
  - Applied to protected endpoints via `dependencies=[Depends(verify_bearer_token)]`

#### Middleware Stack
```
┌────────────────────────────┐
│   RequestIDMiddleware      │  Generate & attach unique request IDs
├────────────────────────────┤
│   CORSMiddleware          │  Handle cross-origin requests
├────────────────────────────┤
│   Authentication          │  Bearer token validation (auth.py)
├────────────────────────────┤
│   Route Handlers          │  Endpoint logic
└────────────────────────────┘
```

### 2. Endpoints Layer

#### OpenAI v1 API (`endpoints/openai_v1.py`)
- `GET /v1/models` - List available MoM models
- `POST /v1/chat/completions` - Chat completion with streaming
  - Supports multimodal content (OpenAI Vision API format)
  - Automatically filters models based on content type

#### Metrics API (`endpoints/metrics_api.py`)
- `GET /v1/metrics/usage` - Aggregated usage statistics
- `GET /v1/metrics/usage/raw` - Raw metric records

#### Request/Response Models (`endpoints/models.py`)

**Multimodal Message Support**:
```python
ChatMessage:
  role: "user" | "assistant" | "system"
  content: Union[str, List[ContentPart]]  # Supports multimodal
  images: List[str] = Field(default_factory=list)  # Alternative multimodal format

ContentPart = Union[TextContentPart, ImageContentPart]

TextContentPart:
  type: "text"
  text: str

ImageContentPart:
  type: "image_url"
  image_url:
    url: str
    detail: "auto" | "low" | "high"  # Optional detail level
```

**Pydantic Model Improvements**:
- `images` field uses `Field(default_factory=list)` instead of `Optional[list]`
- Ensures empty list default instead of `None` for better validation
- Prevents issues with strict providers that reject `None` values

**Backward Compatibility**: Simple string content still supported for text-only messages.

#### Health Checks (`health.py`)
- `GET /health` - Basic liveness check
- `GET /health/detailed` - Component health validation

### 3. Core Logic Layer

#### Multimodal Support (`multimodal_utils.py`)

**Capabilities**:
- Detects multimodal content in messages (images, files, etc.)
- Identifies models with vision/multimodal capabilities
- Automatically filters LLMs based on request type
- Supports OpenAI Vision API format
- **NEW**: Sanitizes messages for strict LLM providers (e.g., Mistral)

**Content Detection**:
```python
# Detects these multimodal formats:
- content: [{"type": "text"}, {"type": "image_url", "image_url": {...}}]
- images: ["url1", "url2"]
- attachments: [...]
- files: [...]
```

**Message Sanitization** (`sanitize_messages_for_provider`):
```
Purpose: Remove provider-specific fields that cause errors with strict providers
Process:
1. Create clean copy of messages with only core fields (role, content)
2. For multimodal-capable models: Include 'images' field if non-empty
3. Preserve standard fields: name, function_call, tool_calls, tool_call_id
4. Return sanitized messages suitable for target provider

Example: Mistral rejects requests with empty 'images' field → sanitizer removes it
```

**Model Filtering Flow**:
```
1. Check if request has multimodal content
2. If yes, filter llms_to_query to only multimodal-capable models
3. Skip non-capable models (e.g., GPT-3.5, text-only models)
4. Sanitize messages for each target provider
5. Log which models are used/skipped
```

**Supported Multimodal Models**:
- OpenAI: gpt-4o, gpt-4-turbo, gpt-4-vision
- Anthropic: claude-3-* (opus, sonnet, haiku)
- Google: gemini-1.5-*, gemini-2.*
- Automatic detection via LiteLLM's supports_vision() when available

#### Pricing & Cost Tracking

**Pricing Utilities (`pricing_utils.py`)**:
- Converts actual costs to normalized tokens (unified pricing model)
- Unified rates: $1/1M input tokens, $10/1M output tokens
- Preserves actual costs while providing consistent token reporting
- Handles rounding from integer token counts

**Cost Calculation (`cost_calculation.py`)**:
- Detailed cost breakdown with reasoning token support
- Custom pricing configs for models with differential pricing
- Fallback to LiteLLM pricing when custom pricing unavailable
- Separate tracking for input, text output, and reasoning output

**Reasoning Token Support**:
```yaml
Models with reasoning tokens:
- Gemini 2.5 Flash/Pro: reasoning_tokens + text_tokens
- Claude Sonnet 4.5: reasoning_tokens in completion_details
- OpenAI o1/o4: reasoning_tokens in token details

Custom pricing example:
pricing:
  prompt_cost_per_token: 0.00000015     # Input tokens
  completion_cost_per_token: 0.00000060 # Text output
  reasoning_cost_per_token: 0.00000350  # Reasoning output (higher cost)
```

**Cost Tracking Flow**:
```
1. Extract token counts from LLM response
2. Check for custom pricing config
3. If present: calculate with reasoning token breakdown
4. If absent: fall back to LiteLLM pricing
5. Record detailed cost breakdown in metrics
6. Update Langfuse with cost details
```

#### Configuration System (`config.py`)

**Purpose**: Defines Pydantic models for all configuration entities and loads config.yaml

**Configuration Structure**:
```python
MoMConfig
├── llm_definitions: List[LLMDefinition]
│   ├── name: str
│   ├── model: str
│   ├── api_key_env: str
│   ├── params: Dict
│   └── pricing: Optional[PricingConfig]  # Custom pricing config
│       ├── prompt_cost_per_token: Optional[float]
│       ├── completion_cost_per_token: Optional[float]
│       └── reasoning_cost_per_token: Optional[float]
├── prompt_definitions: List[PromptDefinition]
├── models: List[ModelConfig]
│   ├── name: str
│   ├── llms_to_query: List[str]
│   ├── concluding_llm: str
│   ├── concluding_prompt: Optional[str]
│   └── include_thinking_context: bool
└── service: ServiceConfig
    ├── timeout_seconds: int
    ├── exposed_apis: List[str]  # Which APIs to expose (e.g., ["openai"])
    ├── cache_enabled: bool
    ├── max_llm_retries: int  # Retry configuration for LLM calls
    └── llm_retry_delay_seconds: int
```

**Config Loading Priority**:
1. Path explicitly provided to `load_config()`
2. `MOM_CONFIG_PATH` environment variable
3. `./config.yaml` (current directory)
4. `../config.yaml` (parent directory)

**Module Documentation**: Comprehensive docstrings explaining all configuration entities and their relationships

#### Fan-Out/Fan-In Orchestration (`core_logic.py`)

**Purpose**: Implements the core MoM pattern for parallel LLM orchestration with aggregation and synthesis

**Module Components**:
- `_perform_fanout_calls`: Execute parallel LLM queries
- `_prepare_concluding_messages`: Prepare synthesis prompt with intermediate results
- `_execute_concluding_call`: Run final synthesis LLM
- `_calculate_and_log_costs`: Aggregate and log costs from all calls

**Fan-Out Engine** (`_perform_fanout_calls`):
```
Input: Request messages, List of LLM names, Config

Process:
1. Create async tasks for each LLM
2. Execute all tasks concurrently (asyncio.gather)
3. Handle individual failures gracefully (partial failure resilience)
4. Collect responses into ThinkingContextItem[]

Output: List[ThinkingContextItem]
```

**Concluding Engine** (`_execute_concluding_call`):
```
Input: Original messages, Intermediate responses, Concluding LLM, Config

Process:
1. Prepare synthesis prompt with all intermediate responses
2. Call concluding LLM with aggregated context
3. Stream response back to client (SSE format)
4. Track usage and costs for final call

Output: Final synthesized response (streaming)
```

**Error Handling**: Gracefully handles partial failures where some fan-out calls succeed while others fail

### 4. LLM Communication Layer

#### LLM Call Handler (`llm_calls.py`)

**Purpose**: Low-level interface for making calls to LLM providers through LiteLLM with caching, retries, and metrics

**Key Functions**:
- `_call_lite_llm`: Main function for LLM calls with caching/retries
- `_generate_cache_key`: Creates deterministic SHA256 cache keys
- `_get_cached_response`: Retrieves responses from SQLite cache
- `_cache_response`: Stores successful responses in cache

**Features**:
- Caching with SHA256 key generation
- Automatic retry with exponential backoff (configurable)
- Cost calculation and metrics recording
- Langfuse tracing integration
- Streaming and non-streaming support
- **NEW**: Message sanitization before LLM calls

**Flow**:
```
1. Sanitize messages for target provider (remove incompatible fields)
2. Generate cache key from (model, sanitized_messages, params)
3. Check cache → Return if hit
4. Call LiteLLM with retry logic (num_retries from config)
5. Record metrics (tokens, cost, duration, status)
6. Store in cache if successful (using sanitized messages)
7. Update Langfuse trace
8. Return response
```

**Message Sanitization Integration**:
- Applied before cache key generation for consistency
- Prevents errors with strict providers (e.g., Mistral)
- Cached using sanitized messages to ensure cache hits work correctly

### 5. Data Persistence Layer

#### Cache Database (`llm_cache.db`)
```sql
Table: cache
├── key: TEXT PRIMARY KEY
├── request_messages: TEXT
├── response_json: TEXT
└── timestamp: REAL
```

**Purpose**: Store LLM responses to reduce API costs and latency

#### Metrics Database (`metrics.db`)
```sql
Table: metrics
├── id: INTEGER PRIMARY KEY
├── request_id: TEXT
├── timestamp: REAL
├── mom_model_name: TEXT
├── llm_name: TEXT
├── call_type: TEXT (fanout/concluding)
├── prompt_tokens: INTEGER
├── completion_tokens: INTEGER
├── total_tokens: INTEGER
├── cost: REAL
├── duration_ms: REAL
├── status: TEXT (SUCCESS/FAILED/CACHED)
├── error_message: TEXT
└── cache_hit: BOOLEAN

Indexes:
- idx_timestamp
- idx_request_id
- idx_mom_model_name
- idx_status
```

**Purpose**: Track usage, costs, and performance metrics

## 🔐 Security Architecture

### Authentication Flow
```
1. Client includes: Authorization: Bearer <token>
2. FastAPI dependency (verify_bearer_token) extracts token
3. Validate API_TOKEN is configured (503 if missing)
4. Check Authorization header format (401 if invalid scheme)
5. Compare token with API_TOKEN environment variable
6. Reject with structured error (401) if tokens don't match
7. Allow request to proceed if valid
```

**Authentication Module** (`auth.py`):
- Centralized authentication logic in reusable dependency
- Structured error responses with error types
- Clear separation of concerns (503 for misconfiguration vs 401 for auth failure)
- Applied via `dependencies=[Depends(verify_bearer_token)]` pattern

### Security Features
- **Non-root Docker user**: Container runs as `appuser`
- **No secrets in code**: All keys via environment variables
- **CORS configuration**: Restrict origins via ALLOWED_CORS_ORIGINS
- **Token-based auth**: Simple bearer token for API access with structured error handling
- **Input validation**: Pydantic models for all requests
- **Message sanitization**: Prevents injection of malicious fields to strict providers

## 📈 Observability Architecture

### Logging Strategy
```
Format: timestamp - module - level - [request_id] - message

Levels:
- INFO: Normal operations, request lifecycle
- WARNING: Recoverable errors, cache misses
- ERROR: Failures, exceptions with stack traces
```

### Request Tracking
- Unique UUID per request
- Propagated via contextvars (thread-safe)
- Included in all log messages
- Returned in X-Request-ID header

### Metrics Collection
```
Per LLM Call:
├── Tokens (prompt, completion, total)
├── Cost (calculated via LiteLLM)
├── Duration (milliseconds)
├── Status (SUCCESS/FAILED/CACHED)
├── Cache hit/miss
└── Request correlation (request_id)
```

### Langfuse Integration
```
Trace Hierarchy:
└── Trace (request_id)
    ├── Generation (fanout: gpt-4)
    ├── Generation (fanout: claude)
    ├── Generation (fanout: gemini)
    └── Generation (concluding: gpt-4)

Each Generation includes:
- Input messages
- Output content
- Token usage
- Model parameters
- Latency
- Status
```

## 🚀 Deployment Architecture

### Docker Container Structure
```
Multi-Stage Build:
┌─────────────────────┐
│  Stage 1: Builder   │
│  - Install deps     │
│  - Compile packages │
└──────────┬──────────┘
           │ Copy artifacts
           ▼
┌─────────────────────┐
│ Stage 2: Production │
│  - Minimal base     │
│  - Runtime deps     │
│  - Non-root user    │
│  - Health checks    │
└─────────────────────┘
```

### Volume Mounts
- `/app/config.yaml` - Configuration (read-only)
- `/app/data` - Databases and persistent storage

### Environment Variables
- `API_TOKEN` - Authentication token
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. - LLM provider keys
- `LANGFUSE_*` - Observability configuration
- `ALLOWED_CORS_ORIGINS` - CORS policy

## 📝 Code Documentation

### Module-Level Documentation
All core modules now include comprehensive docstrings with:
- Purpose and responsibility of the module
- Key functions and their roles
- Data flow explanations
- Usage examples where applicable

**Documented Modules**:
- `auth.py`: Authentication dependency and Bearer token validation
- `config.py`: Configuration entities and loading logic
- `core_logic.py`: Fan-out/fan-in orchestration pattern
- `llm_calls.py`: LLM call handling with caching and retries
- `multimodal_utils.py`: Multimodal content detection and filtering

### Function-Level Documentation
Key functions include detailed docstrings with:
- Parameter descriptions and types
- Return value specifications
- Raised exceptions
- Usage examples and patterns

## 🎯 Design Decisions

### Why Fan-Out Pattern?
- **Parallelism**: Multiple LLMs called simultaneously
- **Resilience**: Single LLM failure doesn't fail request
- **Quality**: Diverse perspectives improve synthesis

### Why SQLite?
- **Simplicity**: No external database server needed
- **Performance**: Fast for read-heavy caching workload
- **Portability**: Single file, easy backups
- **Sufficient**: Adequate for single-instance deployments

### Why LiteLLM?
- **Unified Interface**: Single API for 100+ LLM providers
- **Cost Tracking**: Built-in cost calculation
- **Reliability**: Automatic retries and fallbacks
- **Flexibility**: Easy to add new providers

### Why FastAPI?
- **Performance**: Async/await for concurrent operations
- **Type Safety**: Pydantic models for validation
- **OpenAPI**: Automatic API documentation
- **Ecosystem**: Rich middleware and tooling

## 🔄 Data Flow Examples

### Example 1: Cache Hit Scenario
```
Client Request → Auth → Fanout Engine
                          ↓
                    LLM Call Handler
                          ↓
                    Check Cache → HIT!
                          ↓
                    Return Cached Response (cost: $0)
                          ↓
                    Record Metric (cache_hit: true)
```

### Example 2: Streaming Response
```
Client → POST /v1/chat/completions (stream: true)
           ↓
      Fanout Calls (collect all)
           ↓
      Concluding LLM Call
           ↓
      Yield chunks as they arrive
           ↓
      Client receives SSE stream
```

### Example 3: Metrics Query
```
Client → GET /v1/metrics/usage?model_name=mom-creative&start_time=...
           ↓
      Metrics API Handler
           ↓
      Query SQLite with filters
           ↓
      Aggregate: sum costs, count requests, calculate cache hit rate
           ↓
      Return JSON response
```

## 📊 Performance Characteristics

### Latency Components
```
Total Latency = Fanout (parallel) + Concluding (serial) + Overhead

Fanout Time = max(LLM1, LLM2, LLM3)  # Parallel
Concluding Time = LLM_concluding      # Serial
Overhead = ~50-100ms (validation, routing, serialization)
```

### Caching Impact
- **Cache Hit**: <10ms response time, $0 cost
- **Cache Miss**: Full LLM call latency + cost
- **Typical Hit Rate**: 30-50% (varies by use case)

### Scalability Considerations
- **Current**: Single-instance, SQLite-based
- **Bottleneck**: Concluding LLM call (serial)
- **Scaling Options**:
  - Horizontal: Multiple instances (need shared cache)
  - Vertical: More workers per instance
  - Database: Move to PostgreSQL for multi-instance

## 🧪 Testing Strategy

### Test Pyramid
```
        ┌─────────────┐
        │  E2E Tests  │  ← Full API integration
        ├─────────────┤
        │Integration  │  ← Component interactions
        │    Tests    │
        ├─────────────┤
        │   Unit      │  ← Individual functions
        │   Tests     │
        └─────────────┘
```

### Test Coverage by Layer
- **Config**: Unit tests for validation, loading
- **Core Logic**: Unit tests for cost calculations, message preparation
- **LLM Calls**: Integration tests with mocked HTTP responses
- **Endpoints**: Integration tests with TestClient
- **Health**: Unit + integration tests for all checks

## 📚 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Pydantic Documentation](https://docs.pydantic.dev/)
