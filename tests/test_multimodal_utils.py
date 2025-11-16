"""
Unit tests for mom_service.multimodal_utils module

Tests functionality for detecting and handling multimodal content (images, files)
and filtering models based on multimodal capabilities.
"""

from unittest.mock import MagicMock, patch

import pytest

from mom_service.config import LLMDefinition
from mom_service.multimodal_utils import (
    filter_multimodal_capable_models,
    has_multimodal_content,
    is_model_multimodal_capable,
    sanitize_messages_for_provider,
)


class TestHasMultimodalContent:
    """Tests for has_multimodal_content function"""

    def test_text_only_messages(self):
        """Test that text-only messages return False"""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "What is the weather today?"},
        ]

        assert has_multimodal_content(messages) is False

    def test_image_url_in_content_array(self):
        """Test detection of image_url in content array (OpenAI format)"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                ],
            }
        ]

        assert has_multimodal_content(messages) is True

    def test_image_type_in_content_array(self):
        """Test detection of image type in content array"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image"},
                    {"type": "image", "image": "base64encodedimage..."},
                ],
            }
        ]

        assert has_multimodal_content(messages) is True

    def test_file_type_in_content_array(self):
        """Test detection of file type in content array"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this document"},
                    {"type": "file", "file": "document.pdf"},
                ],
            }
        ]

        assert has_multimodal_content(messages) is True

    def test_images_field(self):
        """Test detection of images field in message"""
        messages = [
            {
                "role": "user",
                "content": "What's in these images?",
                "images": ["image1.jpg", "image2.jpg"],
            }
        ]

        assert has_multimodal_content(messages) is True

    def test_empty_images_field(self):
        """Test that empty images field returns False"""
        messages = [{"role": "user", "content": "Hello", "images": []}]

        assert has_multimodal_content(messages) is False

    def test_attachments_field(self):
        """Test detection of attachments field"""
        messages = [
            {
                "role": "user",
                "content": "See the attached file",
                "attachments": [{"type": "image", "url": "image.jpg"}],
            }
        ]

        assert has_multimodal_content(messages) is True

    def test_files_field(self):
        """Test detection of files field"""
        messages = [
            {
                "role": "user",
                "content": "Check this file",
                "files": ["document.pdf"],
            }
        ]

        assert has_multimodal_content(messages) is True

    def test_multiple_messages_with_multimodal(self):
        """Test detection across multiple messages"""
        messages = [
            {"role": "user", "content": "First message is text only"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Second has an image"},
                    {"type": "image_url", "image_url": {"url": "image.jpg"}},
                ],
            },
        ]

        assert has_multimodal_content(messages) is True

    def test_empty_messages_list(self):
        """Test empty messages list returns False"""
        assert has_multimodal_content([]) is False

    def test_none_content(self):
        """Test handling of None content"""
        messages = [{"role": "user", "content": None}]

        assert has_multimodal_content(messages) is False

    def test_mixed_content_formats(self):
        """Test various content formats together"""
        messages = [
            {"role": "user", "content": "Text message"},
            {"role": "user", "content": ["Just a list of strings"]},
            {"role": "user", "content": [{"type": "text", "text": "More text"}]},
        ]

        assert has_multimodal_content(messages) is False


class TestIsModelMultimodalCapable:
    """Tests for is_model_multimodal_capable function"""

    def test_gpt4o_is_multimodal(self):
        """Test that GPT-4o is recognized as multimodal"""
        assert is_model_multimodal_capable("gpt-4o") is True
        assert is_model_multimodal_capable("openai/gpt-4o") is True
        assert is_model_multimodal_capable("gpt-4o-mini") is True

    def test_gpt4_turbo_is_multimodal(self):
        """Test that GPT-4 Turbo is recognized as multimodal"""
        assert is_model_multimodal_capable("gpt-4-turbo") is True
        assert is_model_multimodal_capable("openai/gpt-4-turbo") is True

    def test_vision_keyword_models(self):
        """Test models with 'vision' in the name"""
        assert is_model_multimodal_capable("gpt-4-vision") is True
        assert is_model_multimodal_capable("some-model-vision-preview") is True

    def test_claude3_is_multimodal(self):
        """Test that Claude 3 models are recognized as multimodal"""
        assert is_model_multimodal_capable("claude-3-opus") is True
        assert is_model_multimodal_capable("claude-3-sonnet") is True
        assert is_model_multimodal_capable("anthropic/claude-3-haiku") is True

    def test_gemini_15_is_multimodal(self):
        """Test that Gemini 1.5 models are multimodal"""
        assert is_model_multimodal_capable("gemini-1.5-pro") is True
        assert is_model_multimodal_capable("gemini/gemini-1.5-flash") is True

    def test_gemini_2_is_multimodal(self):
        """Test that Gemini 2 models are multimodal"""
        assert is_model_multimodal_capable("gemini-2.0-flash") is True
        assert is_model_multimodal_capable("gemini/gemini-2.5-flash") is True

    def test_gpt35_not_multimodal(self):
        """Test that GPT-3.5 is not multimodal"""
        assert is_model_multimodal_capable("gpt-3.5-turbo") is False

    def test_text_only_models(self):
        """Test that text-only models are not detected as multimodal"""
        assert is_model_multimodal_capable("text-davinci-003") is False
        assert is_model_multimodal_capable("claude-2") is False
        assert is_model_multimodal_capable("llama-2-70b") is False

    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive"""
        assert is_model_multimodal_capable("GPT-4O") is True
        assert is_model_multimodal_capable("CLAUDE-3-OPUS") is True
        assert is_model_multimodal_capable("GEMINI-1.5-PRO") is True

    def test_litellm_supports_vision_function(self):
        """Test fallback to litellm.supports_vision if available"""
        with patch("mom_service.multimodal_utils.litellm") as mock_litellm:
            mock_litellm.supports_vision = MagicMock(return_value=True)

            # Test with a model that doesn't match keywords (avoid 'vision', 'gpt-4o', 'claude-3', 'gemini')
            assert is_model_multimodal_capable("custom-model-x") is True
            mock_litellm.supports_vision.assert_called()

    def test_litellm_model_list_fallback(self):
        """Test fallback to litellm.model_list"""
        with patch("mom_service.multimodal_utils.litellm") as mock_litellm:
            mock_litellm.model_list = [
                {
                    "model_name": "custom-model-x",
                    "litellm_supports_vision": True,
                },
                {
                    "model_name": "custom-model-y",
                    "litellm_supports_vision": False,
                },
            ]
            # Remove supports_vision function
            delattr(mock_litellm, "supports_vision")

            assert is_model_multimodal_capable("custom-model-x") is True
            assert is_model_multimodal_capable("custom-model-y") is False

    def test_error_handling(self):
        """Test that errors are caught and default to False"""
        with patch("mom_service.multimodal_utils.litellm") as mock_litellm:
            mock_litellm.supports_vision = MagicMock(side_effect=Exception("Error"))

            # Should default to False on error (for models without keywords)
            with patch("mom_service.multimodal_utils.logger.warning") as mock_warning:
                result = is_model_multimodal_capable("unknown-model")
                assert result is False
                mock_warning.assert_called()


class TestFilterMultimodalCapableModels:
    """Tests for filter_multimodal_capable_models function"""

    @pytest.fixture
    def llm_map(self):
        """Fixture providing a sample LLM map"""
        return {
            "gpt4o": LLMDefinition(
                name="gpt4o",
                model="gpt-4o",
                api_key_env="OPENAI_API_KEY",
            ),
            "gpt35": LLMDefinition(
                name="gpt35",
                model="gpt-3.5-turbo",
                api_key_env="OPENAI_API_KEY",
            ),
            "claude3": LLMDefinition(
                name="claude3",
                model="claude-3-opus",
                api_key_env="ANTHROPIC_API_KEY",
            ),
            "gemini": LLMDefinition(
                name="gemini",
                model="gemini-1.5-pro",
                api_key_env="GOOGLE_API_KEY",
            ),
        }

    def test_text_only_messages_all_models_included(self, llm_map):
        """Test that all models are included for text-only messages"""
        messages = [{"role": "user", "content": "Hello"}]
        llm_names = ["gpt4o", "gpt35", "claude3", "gemini"]

        capable, skipped = filter_multimodal_capable_models(llm_names, llm_map, messages)

        assert capable == llm_names
        assert not skipped

    def test_multimodal_content_filters_models(self, llm_map):
        """Test that multimodal content filters out non-capable models"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "image.jpg"}},
                ],
            }
        ]
        llm_names = ["gpt4o", "gpt35", "claude3", "gemini"]

        capable, skipped = filter_multimodal_capable_models(llm_names, llm_map, messages)

        # gpt-3.5-turbo should be filtered out
        assert "gpt4o" in capable
        assert "claude3" in capable
        assert "gemini" in capable
        assert "gpt35" in skipped

    def test_all_models_capable(self, llm_map):
        """Test when all models are multimodal-capable"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this"},
                    {"type": "image", "image": "base64..."},
                ],
            }
        ]
        llm_names = ["gpt4o", "claude3", "gemini"]

        capable, skipped = filter_multimodal_capable_models(llm_names, llm_map, messages)

        assert set(capable) == set(llm_names)
        assert not skipped

    def test_no_models_capable(self, llm_map):
        """Test when no models are multimodal-capable"""
        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "image.jpg"}}],
            }
        ]
        llm_names = ["gpt35"]  # Only non-capable model

        with patch("mom_service.multimodal_utils.logger.warning") as mock_warning:
            capable, skipped = filter_multimodal_capable_models(llm_names, llm_map, messages)

            assert capable == []
            assert "gpt35" in skipped

            # Should log warning about no capable models
            assert mock_warning.called
            warning_msg = mock_warning.call_args[0][0]
            assert "no multimodal-capable" in warning_msg.lower()

    def test_llm_not_in_map(self, llm_map):
        """Test handling when LLM is not in the map"""
        messages = [{"role": "user", "content": [{"type": "image", "image": "..."}]}]
        llm_names = ["gpt4o", "nonexistent-model"]

        with patch("mom_service.multimodal_utils.logger.warning") as mock_warning:
            capable, skipped = filter_multimodal_capable_models(llm_names, llm_map, messages)

            assert "gpt4o" in capable
            assert "nonexistent-model" in skipped

            # Should log warning about missing LLM definition
            assert mock_warning.called

    def test_empty_llm_list(self, llm_map):
        """Test with empty LLM list"""
        messages = [{"role": "user", "content": [{"type": "image", "image": "..."}]}]

        capable, skipped = filter_multimodal_capable_models([], llm_map, messages)

        assert capable == []
        assert not skipped

    def test_logs_info_for_multimodal_detection(self, llm_map):
        """Test that info logs are generated during filtering"""
        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "image.jpg"}}],
            }
        ]
        llm_names = ["gpt4o", "gpt35"]

        with patch("mom_service.multimodal_utils.logger.info") as mock_info:
            filter_multimodal_capable_models(llm_names, llm_map, messages)

            # Should log detection of multimodal content
            assert mock_info.called
            log_messages = [call[0][0] for call in mock_info.call_args_list]
            assert any("multimodal content" in msg.lower() for msg in log_messages)

    def test_preserves_llm_order(self, llm_map):
        """Test that the order of LLMs is preserved"""
        messages = [{"role": "user", "content": "Text only"}]
        llm_names = ["gemini", "gpt4o", "claude3", "gpt35"]

        capable, _ = filter_multimodal_capable_models(llm_names, llm_map, messages)

        # Order should be preserved for text-only
        assert capable == llm_names

    def test_mixed_multimodal_detection(self, llm_map):
        """Test with various multimodal indicators"""
        messages = [
            {"role": "user", "content": "First is text"},
            {
                "role": "user",
                "content": "Has image",
                "images": ["img.jpg"],
            },
        ]
        llm_names = ["gpt4o", "gpt35", "claude3"]

        capable, skipped = filter_multimodal_capable_models(llm_names, llm_map, messages)

        assert "gpt4o" in capable
        assert "claude3" in capable
        assert "gpt35" in skipped


class TestSanitizeMessagesForProvider:
    """Tests for sanitize_messages_for_provider function"""

    def test_removes_empty_images_field_for_non_multimodal(self):
        """Test that empty images field is removed for non-multimodal models"""
        messages = [
            {"role": "user", "content": "Hello", "images": []},
            {"role": "assistant", "content": "Hi there", "images": []},
        ]

        sanitized = sanitize_messages_for_provider(messages, "mistral-medium-latest")

        # images field should be removed
        assert all("images" not in msg for msg in sanitized)
        assert sanitized[0]["content"] == "Hello"
        assert sanitized[1]["content"] == "Hi there"

    def test_removes_empty_images_field_for_mistral(self):
        """Test that empty images field is removed for Mistral models"""
        messages = [
            {"role": "user", "content": "What is 2+2?", "images": []},
        ]

        sanitized = sanitize_messages_for_provider(messages, "mistral-medium-latest")

        assert "images" not in sanitized[0]
        assert sanitized[0]["role"] == "user"
        assert sanitized[0]["content"] == "What is 2+2?"

    def test_keeps_images_field_for_multimodal_with_content(self):
        """Test that images field with content is kept for multimodal models"""
        messages = [
            {
                "role": "user",
                "content": "Analyze this",
                "images": ["https://example.com/image.jpg"],
            },
        ]

        sanitized = sanitize_messages_for_provider(messages, "gpt-4o")

        # images field should be kept for multimodal models with content
        assert "images" in sanitized[0]
        assert sanitized[0]["images"] == ["https://example.com/image.jpg"]

    def test_removes_empty_images_even_for_multimodal(self):
        """Test that empty images field is removed even for multimodal models"""
        messages = [
            {"role": "user", "content": "Hello", "images": []},
        ]

        sanitized = sanitize_messages_for_provider(messages, "gpt-4o")

        # Empty images field should be removed even for multimodal models
        assert "images" not in sanitized[0]

    def test_preserves_other_standard_fields(self):
        """Test that other standard fields are preserved"""
        messages = [
            {
                "role": "user",
                "content": "Test",
                "images": [],
                "name": "user1",
                "tool_call_id": "call_123",
            },
        ]

        sanitized = sanitize_messages_for_provider(messages, "mistral-medium-latest")

        assert sanitized[0]["role"] == "user"
        assert sanitized[0]["content"] == "Test"
        assert sanitized[0]["name"] == "user1"
        assert sanitized[0]["tool_call_id"] == "call_123"
        assert "images" not in sanitized[0]

    def test_preserves_function_call_and_tool_calls(self):
        """Test that function_call and tool_calls are preserved"""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "images": [],
                "function_call": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                "tool_calls": [{"id": "1", "type": "function"}],
            },
        ]

        sanitized = sanitize_messages_for_provider(messages, "gpt-3.5-turbo")

        assert "function_call" in sanitized[0]
        assert "tool_calls" in sanitized[0]
        assert "images" not in sanitized[0]

    def test_does_not_modify_original_messages(self):
        """Test that original messages are not modified"""
        original_messages = [
            {"role": "user", "content": "Hello", "images": []},
        ]

        sanitize_messages_for_provider(original_messages, "mistral-medium-latest")

        # Original should still have images field
        assert "images" in original_messages[0]

    def test_handles_multiple_messages(self):
        """Test sanitization of multiple messages"""
        messages = [
            {"role": "system", "content": "You are helpful", "images": []},
            {"role": "user", "content": "Question 1", "images": []},
            {"role": "assistant", "content": "Answer 1", "images": []},
            {"role": "user", "content": "Question 2", "images": []},
        ]

        sanitized = sanitize_messages_for_provider(messages, "mistral-medium-latest")

        assert len(sanitized) == 4
        assert all("images" not in msg for msg in sanitized)
        assert [msg["content"] for msg in sanitized] == [
            "You are helpful",
            "Question 1",
            "Answer 1",
            "Question 2",
        ]

    def test_handles_complex_content_array(self):
        """Test sanitization with complex content array (multimodal format)"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "image.jpg"}},
                ],
                "images": [],
            },
        ]

        sanitized = sanitize_messages_for_provider(messages, "gpt-4o")

        # Complex content should be preserved
        assert isinstance(sanitized[0]["content"], list)
        assert len(sanitized[0]["content"]) == 2
        # Empty images field should be removed
        assert "images" not in sanitized[0]

    def test_empty_messages_list(self):
        """Test sanitization with empty messages list"""
        sanitized = sanitize_messages_for_provider([], "gpt-4o")
        assert not sanitized

    def test_mistral_models_have_strict_schema(self):
        """Test that various Mistral model names trigger sanitization"""
        messages = [{"role": "user", "content": "Test", "images": []}]

        for model in ["mistral-medium-latest", "mistral-large", "mistral-small"]:
            sanitized = sanitize_messages_for_provider(messages, model)
            assert "images" not in sanitized[0], f"images should be removed for {model}"

    def test_openai_models_keep_images_with_content(self):
        """Test that OpenAI models keep images when present"""
        messages = [
            {
                "role": "user",
                "content": "Analyze",
                "images": ["img1.jpg", "img2.jpg"],
            },
        ]

        for model in ["gpt-4o", "gpt-4-turbo", "gpt-4-vision-preview"]:
            sanitized = sanitize_messages_for_provider(messages, model)
            assert "images" in sanitized[0], f"images should be kept for {model}"
            assert len(sanitized[0]["images"]) == 2
