"""
Utilities for detecting and handling multimodal content in LLM requests.
"""

import logging
from typing import Any

import litellm

logger = logging.getLogger(__name__)


def has_multimodal_content(messages: list[dict[str, Any]]) -> bool:
    """
    Check if messages contain multimodal content (images, files, etc.).

    Args:
        messages: List of message dictionaries

    Returns:
        True if any message contains multimodal content, False otherwise
    """
    for message in messages:
        # Check for content array format (OpenAI vision API style)
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and (
                    "image_url" in item
                    or "type" in item
                    and item["type"]
                    in [
                        "image",
                        "image_url",
                        "file",
                    ]
                ):
                    return True

        # Check for images field (some formats use this)
        if "images" in message and message["images"]:
            return True

        # Check for attachments or files
        if "attachments" in message and message["attachments"]:
            return True
        if "files" in message and message["files"]:
            return True

    return False


def is_model_multimodal_capable(model_name: str) -> bool:
    """
    Check if a model supports multimodal input (vision/images).

    Uses LiteLLM's model info to determine capability.

    Args:
        model_name: The LiteLLM model identifier (e.g., "openai/gpt-4-vision")

    Returns:
        True if the model supports multimodal input, False otherwise
    """
    try:
        # LiteLLM tracks vision-capable models
        # Check if model name contains vision indicators
        vision_keywords = ["vision", "gpt-4o", "gpt-4-turbo", "claude-3", "gemini-1.5", "gemini-2"]

        model_lower = model_name.lower()
        for keyword in vision_keywords:
            if keyword in model_lower:
                return True

        # Check litellm.supports_vision function if available
        if hasattr(litellm, "supports_vision") and litellm.supports_vision(model_name):
            return True

        # Fallback check against litellm's model list
        if hasattr(litellm, "model_list"):
            for model_info in litellm.model_list:
                if (
                    "model_name" in model_info
                    and model_info["model_name"] == model_name
                    and model_info.get("litellm_supports_vision")
                ):
                    return True

        return False
    except Exception as e:
        logger.warning(f"Error checking multimodal capability for {model_name}: {e}")
        # If we can't determine, assume it doesn't support multimodal to be safe
        return False


def sanitize_messages_for_provider(
    messages: list[dict[str, Any]], model_name: str
) -> list[dict[str, Any]]:
    """
    Remove provider-specific fields from messages that may cause errors with certain LLM providers.

    Some providers (like Mistral) are strict about the message schema and reject requests
    with extra fields like 'images' even if they're empty. This function creates a clean
    copy of messages suitable for the target provider.

    Args:
        messages: List of message dictionaries to sanitize
        model_name: The target LLM model name (used to determine which fields to keep)

    Returns:
        A sanitized copy of messages without unsupported fields
    """
    sanitized = []
    multimodal_capable = is_model_multimodal_capable(model_name)

    for msg in messages:
        # Create a copy to avoid modifying the original
        clean_msg = {"role": msg["role"], "content": msg["content"]}

        # Only include 'images' field if:
        # 1. The model supports multimodal content
        # 2. The field exists and has actual content (not empty list)
        if multimodal_capable and "images" in msg and msg["images"]:
            clean_msg["images"] = msg["images"]

        # Copy other standard fields if present
        for field in ["name", "function_call", "tool_calls", "tool_call_id"]:
            if field in msg:
                clean_msg[field] = msg[field]

        sanitized.append(clean_msg)

    return sanitized


def filter_multimodal_capable_models(
    llm_names: list[str], llm_map: dict[str, Any], messages: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """
    Filter LLM list to only include models capable of handling the request content.

    If messages contain multimodal content, only multimodal-capable models are included.
    Returns both the filtered list and the list of skipped models.

    Args:
        llm_names: List of LLM names to query
        llm_map: Dictionary mapping LLM names to LLMDefinition objects
        messages: Request messages to check for multimodal content

    Returns:
        Tuple of (filtered_llm_names, skipped_llm_names)
    """
    if not has_multimodal_content(messages):
        # No multimodal content, all models can be used
        return llm_names, []

    logger.info("Detected multimodal content in request, filtering for capable models...")

    capable_models = []
    skipped_models = []

    for llm_name in llm_names:
        llm_def = llm_map.get(llm_name)
        if not llm_def:
            logger.warning(f"LLM definition '{llm_name}' not found in map")
            skipped_models.append(llm_name)
            continue

        if is_model_multimodal_capable(llm_def.model):
            capable_models.append(llm_name)
            logger.info(f"Model {llm_name} ({llm_def.model}) supports multimodal content")
        else:
            skipped_models.append(llm_name)
            logger.info(f"Skipping model {llm_name} ({llm_def.model}) - no multimodal support")

    if not capable_models:
        logger.warning(
            "No multimodal-capable models found in configuration. Request may fail or return errors."
        )

    return capable_models, skipped_models
