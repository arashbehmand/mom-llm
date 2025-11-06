# Configuration Guide

This guide provides detailed information on configuring the MoM Service through the `config.yaml` file and environment variables.

## Table of Contents

- [Overview](#overview)
- [Environment Variables](#environment-variables)
- [Configuration File Structure](#configuration-file-structure)
- [LLM Definitions](#llm-definitions)
- [Synthesis Prompts](#synthesis-prompts)
- [MoM Models](#mom-models)
- [Service Settings](#service-settings)
- [Langfuse Integration](#langfuse-integration)
- [Complete Configuration Example](#complete-configuration-example)
- [Best Practices](#best-practices)

## Overview

The MoM Service is configured through two main sources:

1. **Environment Variables** (`.env` file): API keys, tokens, and sensitive configuration
2. **Configuration File** (`config.yaml`): LLM definitions, model orchestration, and service settings

## Environment Variables

Create a `.env` file in the project root with the following variables:

### Required Variables

```bash
# Service Configuration
API_TOKEN="your-secret-bearer-token"  # Required for API authentication
```

### Optional Service Variables

```bash
# CORS Configuration
ALLOWED_CORS_ORIGINS=""  # Comma-separated origins, or empty for no CORS
                         # Example: "https://example.com,https://app.example.com"

# LiteLLM Logging
LITELLM_VERBOSE="false"  # Set to "true" for detailed LLM call logging
```

### LLM Provider API Keys

Add the API keys for the LLM providers you plan to use:

```bash
# OpenAI
OPENAI_API_KEY="sk-..."

# Google Gemini
GOOGLE_API_KEY="..."

# Anthropic Claude
ANTHROPIC_API_KEY="..."

# Mistral
MISTRAL_API_KEY="..."

# Cohere
COHERE_API_KEY="..."

# Groq
GROQ_API_KEY="..."

# Together AI
TOGETHER_API_KEY="..."

# Replicate
REPLICATE_API_KEY="..."
```

### Observability (Optional)

```bash
# Langfuse for tracing and monitoring
LANGFUSE_PUBLIC_KEY=""
LANGFUSE_SECRET_KEY=""
LANGFUSE_HOST="https://cloud.langfuse.com"
```

## Configuration File Structure

The `config.yaml` file has four main sections:

```yaml
llm_definitions:   # Define individual LLMs
  - name: "..."
    model: "..."
    # ... configuration

prompt_definitions:  # Define synthesis prompts
  - name: "..."
    content: "..."

models:  # Define MoM models (meta-models)
  - name: "..."
    llms_to_query: [...]
    # ... configuration

service:  # Service-level settings
  timeout_seconds: 30
  # ... other settings

langfuse:  # Optional observability
  # ... configuration
```

## LLM Definitions

Define individual LLMs that can be used in your MoM models.

### Basic LLM Definition

```yaml
llm_definitions:
  - name: "gpt4"  # Internal name for referencing
    model: "openai/gpt-4"  # LiteLLM model identifier
    api_key_env: "OPENAI_API_KEY"  # Environment variable containing API key
```

### LLM with Custom Parameters

```yaml
llm_definitions:
  - name: "gpt5"
    model: "openai/gpt-5"
    api_key_env: "OPENAI_API_KEY"
    params:
      reasoning_effort: "high"  # Model-specific parameters
      temperature: 0.7
      max_tokens: 2000
```

### LLM with Custom Pricing

For models with special pricing (like reasoning tokens), you can define custom pricing:

```yaml
llm_definitions:
  - name: "gemini-2.5-pro"
    model: "gemini/gemini-2.5-pro"
    api_key_env: "GOOGLE_API_KEY"
    pricing:
      prompt_cost_per_token: 0.00000015      # $0.15 per 1M tokens
      completion_cost_per_token: 0.00000060  # $0.60 per 1M text tokens
      reasoning_cost_per_token: 0.00000350   # $3.50 per 1M reasoning tokens

  - name: "claude4.5"
    model: "anthropic/claude-4.5"
    api_key_env: "ANTHROPIC_API_KEY"
    pricing:
      prompt_cost_per_token: 0.00000300      # $3.00 per 1M tokens
      completion_cost_per_token: 0.00001500  # $15.00 per 1M text tokens
      reasoning_cost_per_token: 0.00008000   # $80.00 per 1M reasoning tokens
```

**Note**: Custom pricing is optional. If not specified, LiteLLM's default pricing is used. For models with reasoning tokens (Gemini 2.5 Pro, Claude 4.5), custom pricing enables accurate cost tracking.

### Multimodal LLMs

LLMs automatically support multimodal requests if the underlying model supports vision:

```yaml
llm_definitions:
  - name: "gpt4-vision"
    model: "openai/gpt-4-vision-preview"
    api_key_env: "OPENAI_API_KEY"

  - name: "claude-sonnet"
    model: "anthropic/claude-3-5-sonnet-20241022"
    api_key_env: "ANTHROPIC_API_KEY"

  - name: "gemini-pro-vision"
    model: "gemini/gemini-2.5-pro"
    api_key_env: "GOOGLE_API_KEY"
```

The service automatically filters LLMs based on multimodal capability when images are included in requests.

### LiteLLM Model Identifiers

The `model` field uses LiteLLM's model identifier format: `provider/model-name`

Common examples:

| Provider | Format | Example |
|----------|--------|---------|
| OpenAI | `openai/model-name` | `openai/gpt-4o` |
| Anthropic | `anthropic/model-name` | `anthropic/claude-3-5-sonnet-20241022` |
| Google | `gemini/model-name` | `gemini/gemini-2.5-pro` |
| Mistral | `mistral/model-name` | `mistral/mistral-large-latest` |
| Cohere | `cohere/model-name` | `cohere/command-r-plus` |
| Groq | `groq/model-name` | `groq/llama-3.1-70b-versatile` |

See [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for complete list of supported providers.

## Synthesis Prompts

Define prompts that instruct the concluding LLM on how to synthesize multiple responses.

### Basic Synthesis Prompt

```yaml
prompt_definitions:
  - name: "synth_default"
    content: |
      Review all expert responses and synthesize a single, cohesive answer that:
      - Integrates the strongest insights from each response
      - Resolves any disagreements between models
      - Provides a balanced, comprehensive answer
```

### Task-Specific Prompts

```yaml
prompt_definitions:
  - name: "synth_creative"
    content: |
      Synthesize the responses with emphasis on creativity and engagement:
      - Combine the most innovative ideas from each response
      - Create a narrative that flows naturally
      - Balance creativity with factual accuracy

  - name: "synth_technical"
    content: |
      Synthesize the responses focusing on technical accuracy:
      - Verify technical claims across all responses
      - Highlight consensus on technical details
      - Note any technical disagreements and resolve them
      - Provide the most technically sound answer

  - name: "synth_concise"
    content: |
      Create a brief synthesis that:
      - Captures only the most essential points
      - Removes redundancy across responses
      - Provides a clear, concise answer
```

## MoM Models

Define "meta-models" that orchestrate multiple LLMs and synthesize their responses.

### Basic MoM Model

```yaml
models:
  - name: "mom"  # Model name used in API requests
    llms_to_query:  # LLMs to query in parallel
      - "gpt4"
      - "claude"
      - "gemini"
    concluding_llm: "gpt4"  # LLM that synthesizes responses
    concluding_prompt: "synth_default"  # Prompt for synthesis
```

### MoM Model with Thinking Context

Show intermediate responses to users:

```yaml
models:
  - name: "mom-transparent"
    llms_to_query:
      - "gpt4"
      - "claude"
      - "gemini"
    concluding_llm: "gpt4"
    concluding_prompt: "synth_default"
    include_thinking_context: true  # Include intermediate responses
```

When enabled, the output includes intermediate responses wrapped in `<think>` tags:

```
<think>
Model: gpt-4
Content: [GPT-4's response]
---
Model: claude
Content: [Claude's response]
---
Model: gemini
Content: [Gemini's response]
---
</think>

[Final synthesized answer]
```

### Specialized MoM Models

Create different models for different use cases:

```yaml
models:
  # Creative content generation
  - name: "mom-creative"
    llms_to_query:
      - "gpt5"
      - "claude4.5"
      - "gemini-2.5-pro"
    concluding_llm: "gpt5"
    concluding_prompt: "synth_creative"
    include_thinking_context: false

  # Fast responses with cost-effective models
  - name: "mom-fast"
    llms_to_query:
      - "gemini-2.5-pro"
      - "groq-llama"
    concluding_llm: "gemini-2.5-pro"
    concluding_prompt: "synth_concise"
    include_thinking_context: false

  # Technical accuracy focus
  - name: "mom-technical"
    llms_to_query:
      - "gpt5"
      - "claude4.5"
      - "mistral-large"
    concluding_llm: "claude4.5"
    concluding_prompt: "synth_technical"
    include_thinking_context: true

  # Code generation
  - name: "mom-code"
    llms_to_query:
      - "gpt4"
      - "claude-sonnet"
      - "codellama"
    concluding_llm: "claude-sonnet"
    concluding_prompt: "synth_technical"
    include_thinking_context: false
```

## Service Settings

Configure service-level behavior:

```yaml
service:
  # Request timeout
  timeout_seconds: 30  # Maximum time for LLM requests

  # API exposure
  exposed_apis: ["openai"]  # Currently only "openai" is supported

  # Response caching
  cache_enabled: true  # Enable automatic response caching

  # Retry configuration
  max_llm_retries: 3  # Number of retries for failed LLM calls
  llm_retry_delay_seconds: 2  # Delay between retries
```

### Configuration Options Explained

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `timeout_seconds` | integer | 30 | Maximum time (in seconds) to wait for LLM responses |
| `exposed_apis` | array | `["openai"]` | API formats to expose (currently only OpenAI-compatible) |
| `cache_enabled` | boolean | true | Enable automatic caching of LLM responses to reduce costs |
| `max_llm_retries` | integer | 3 | Number of retry attempts for failed LLM calls |
| `llm_retry_delay_seconds` | integer | 2 | Delay (in seconds) between retry attempts |

## Langfuse Integration

Enable distributed tracing and observability with Langfuse:

```yaml
langfuse:
  public_key_env: "LANGFUSE_PUBLIC_KEY"
  secret_key_env: "LANGFUSE_SECRET_KEY"
  host_env: "LANGFUSE_HOST"
```

**Setup Steps:**

1. Sign up at [Langfuse](https://langfuse.com/)
2. Create a new project
3. Copy your API keys
4. Add them to your `.env` file:
   ```bash
   LANGFUSE_PUBLIC_KEY="pk-lf-..."
   LANGFUSE_SECRET_KEY="sk-lf-..."
   LANGFUSE_HOST="https://cloud.langfuse.com"
   ```

Once configured, all requests will be traced in your Langfuse dashboard, providing:
- Request/response traces
- Token usage tracking
- Cost analysis
- Performance metrics
- Error monitoring

## Complete Configuration Example

Here's a comprehensive example combining all features:

```yaml
# LLM Definitions
llm_definitions:
  # OpenAI Models
  - name: "gpt5"
    model: "openai/gpt-5"
    api_key_env: "OPENAI_API_KEY"
    params:
      reasoning_effort: "high"

  - name: "gpt4o"
    model: "openai/gpt-4o"
    api_key_env: "OPENAI_API_KEY"

  # Anthropic Models
  - name: "claude4.5"
    model: "anthropic/claude-4.5"
    api_key_env: "ANTHROPIC_API_KEY"
    pricing:
      prompt_cost_per_token: 0.00000300
      completion_cost_per_token: 0.00001500
      reasoning_cost_per_token: 0.00008000

  - name: "claude-sonnet"
    model: "anthropic/claude-3-5-sonnet-20241022"
    api_key_env: "ANTHROPIC_API_KEY"

  # Google Models
  - name: "gemini-2.5-pro"
    model: "gemini/gemini-2.5-pro"
    api_key_env: "GOOGLE_API_KEY"
    pricing:
      prompt_cost_per_token: 0.00000015
      completion_cost_per_token: 0.00000060
      reasoning_cost_per_token: 0.00000350

  - name: "gemini-flash"
    model: "gemini/gemini-2.5-flash"
    api_key_env: "GOOGLE_API_KEY"

  # Other Providers
  - name: "mistral-large"
    model: "mistral/mistral-large-latest"
    api_key_env: "MISTRAL_API_KEY"

  - name: "groq-llama"
    model: "groq/llama-3.1-70b-versatile"
    api_key_env: "GROQ_API_KEY"

# Synthesis Prompts
prompt_definitions:
  - name: "synth_default"
    content: |
      Review all expert responses and synthesize a single, cohesive answer that:
      - Integrates the strongest insights from each response
      - Resolves any disagreements between models
      - Provides a balanced, comprehensive answer

  - name: "synth_creative"
    content: |
      Synthesize the responses with emphasis on creativity and engagement:
      - Combine the most innovative ideas from each response
      - Create a narrative that flows naturally
      - Balance creativity with factual accuracy

  - name: "synth_technical"
    content: |
      Synthesize the responses focusing on technical accuracy:
      - Verify technical claims across all responses
      - Highlight consensus on technical details
      - Provide the most technically sound answer

# MoM Models
models:
  # Default balanced model
  - name: "mom"
    llms_to_query:
      - "gpt4o"
      - "claude-sonnet"
      - "gemini-2.5-pro"
    concluding_llm: "gpt4o"
    concluding_prompt: "synth_default"
    include_thinking_context: false

  # Premium model with reasoning
  - name: "mom-premium"
    llms_to_query:
      - "gpt5"
      - "claude4.5"
      - "gemini-2.5-pro"
    concluding_llm: "gpt5"
    concluding_prompt: "synth_default"
    include_thinking_context: true

  # Fast and cost-effective
  - name: "mom-fast"
    llms_to_query:
      - "gemini-flash"
      - "groq-llama"
    concluding_llm: "gemini-flash"
    concluding_prompt: "synth_default"
    include_thinking_context: false

  # Creative content
  - name: "mom-creative"
    llms_to_query:
      - "gpt4o"
      - "claude-sonnet"
      - "gemini-2.5-pro"
    concluding_llm: "claude-sonnet"
    concluding_prompt: "synth_creative"
    include_thinking_context: false

  # Technical tasks
  - name: "mom-technical"
    llms_to_query:
      - "gpt5"
      - "claude4.5"
      - "mistral-large"
    concluding_llm: "claude4.5"
    concluding_prompt: "synth_technical"
    include_thinking_context: true

# Service Settings
service:
  timeout_seconds: 30
  exposed_apis: ["openai"]
  cache_enabled: true
  max_llm_retries: 3
  llm_retry_delay_seconds: 2

# Langfuse Integration (Optional)
langfuse:
  public_key_env: "LANGFUSE_PUBLIC_KEY"
  secret_key_env: "LANGFUSE_SECRET_KEY"
  host_env: "LANGFUSE_HOST"
```

## Best Practices

### Model Selection

1. **Diversity**: Choose LLMs with different strengths (creativity, accuracy, speed)
2. **Cost Balance**: Mix premium and cost-effective models
3. **Multimodal**: Ensure all LLMs in a MoM model support vision if you need image processing
4. **Redundancy**: Use at least 3 LLMs for robust synthesis

### Synthesis Configuration

1. **Concluding LLM**: Choose your most capable model for synthesis
2. **Thinking Context**: Enable for debugging and transparency, disable for production speed
3. **Custom Prompts**: Tailor synthesis prompts to your specific use case

### Performance Optimization

1. **Timeouts**: Adjust `timeout_seconds` based on your model selection (slower models need more time)
2. **Caching**: Keep `cache_enabled: true` to reduce costs for repeated queries
3. **Retries**: Configure retries based on your providers' reliability

### Cost Management

1. **Custom Pricing**: Define accurate pricing for models with reasoning tokens
2. **Model Tiers**: Create different MoM models for different priority levels
3. **Monitoring**: Use Langfuse or metrics API to track costs
4. **Fast Models**: Use `mom-fast` style configurations for less critical tasks

### Security

1. **Environment Variables**: Never commit `.env` file to version control
2. **API Token**: Use a strong, unique `API_TOKEN`
3. **CORS**: Restrict `ALLOWED_CORS_ORIGINS` in production
4. **API Keys**: Rotate API keys regularly

### Configuration Management

1. **Version Control**: Keep `config.yaml_template` in git as a reference
2. **Environment-Specific**: Maintain separate configs for dev/staging/prod
3. **Documentation**: Comment your config with explanations for future reference
4. **Validation**: Test configuration changes with health checks before deploying
