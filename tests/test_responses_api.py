"""
Tests for mom_service.responses_api – content format normalization and
Responses API helpers.
"""

from mom_service.responses_api import _build_responses_kwargs, _normalize_message_content_format


class TestNormalizeMessageContentFormat:
    """Tests for _normalize_message_content_format workaround."""

    def test_all_plain_strings_unchanged(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = _normalize_message_content_format(messages)
        assert result is messages  # same object, no copy needed

    def test_all_list_content_unchanged(self):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are helpful."}]},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        result = _normalize_message_content_format(messages)
        assert result == messages
        # Strings should NOT have been touched
        for msg in result:
            assert isinstance(msg["content"], list)

    def test_mixed_formats_promotes_strings_to_list(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = _normalize_message_content_format(messages)

        assert result[0]["content"] == [{"type": "text", "text": "You are helpful."}]
        assert result[1]["content"] == [{"type": "text", "text": "Hello"}]
        assert result[2]["content"] == [{"type": "text", "text": "Hi there!"}]

    def test_mixed_formats_preserves_roles(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "usr"}]},
        ]
        result = _normalize_message_content_format(messages)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_does_not_mutate_original_messages(self):
        original_msg = {"role": "user", "content": "Hello"}
        messages = [
            original_msg,
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        ]
        _normalize_message_content_format(messages)
        # Original dict should be untouched
        assert original_msg["content"] == "Hello"

    def test_empty_list(self):
        result = _normalize_message_content_format([])
        assert not result

    def test_none_content_left_alone(self):
        messages = [
            {"role": "assistant", "content": None},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
        result = _normalize_message_content_format(messages)
        assert result[0]["content"] is None
        assert result[1]["content"] == [{"type": "text", "text": "hi"}]

    def test_multimodal_content_preserved(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            },
        ]
        result = _normalize_message_content_format(messages)
        assert result[0]["content"] == [{"type": "text", "text": "You are helpful."}]
        assert result[1]["content"] == messages[1]["content"]

    def test_extra_message_keys_preserved(self):
        messages = [
            {"role": "user", "content": "hi", "name": "alice"},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ]
        result = _normalize_message_content_format(messages)
        assert result[0]["name"] == "alice"


class TestBuildResponsesKwargsNormalization:
    """Verify _build_responses_kwargs applies content normalization."""

    def test_mixed_content_normalized_in_input(self):
        params = {
            "model": "xai/grok-test",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ],
            "timeout": 60,
        }
        kwargs = _build_responses_kwargs(params)
        # Both messages should now be list-of-parts
        assert kwargs["input"][0]["content"] == [{"type": "text", "text": "Be helpful."}]
        assert kwargs["input"][1]["content"] == [{"type": "text", "text": "Hello"}]

    def test_plain_string_messages_left_as_strings(self):
        params = {
            "model": "xai/grok-test",
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "timeout": 60,
        }
        kwargs = _build_responses_kwargs(params)
        assert kwargs["input"][0]["content"] == "Hello"
