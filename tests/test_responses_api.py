"""
Tests for mom_service.responses_api – content format normalization and
Responses API helpers.
"""

from mom_service.responses_api import _build_responses_kwargs, _flatten_message_content_to_strings


class TestFlattenMessageContentToStrings:
    """Tests for _flatten_message_content_to_strings workaround."""

    def test_all_plain_strings_unchanged(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = _flatten_message_content_to_strings(messages)
        assert result is messages  # same object, no copy needed

    def test_text_only_list_flattened_to_string(self):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are helpful."}]},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        result = _flatten_message_content_to_strings(messages)
        assert result[0]["content"] == "You are helpful."
        assert result[1]["content"] == "Hello"

    def test_multi_text_parts_concatenated(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part one. "},
                    {"type": "text", "text": "Part two."},
                ],
            },
        ]
        result = _flatten_message_content_to_strings(messages)
        assert result[0]["content"] == "Part one. Part two."

    def test_mixed_formats_flattens_lists_keeps_strings(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = _flatten_message_content_to_strings(messages)

        assert result[0]["content"] == "You are helpful."
        assert result[1]["content"] == "Hello"
        assert result[2]["content"] == "Hi there!"

    def test_preserves_roles(self):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "usr"}]},
        ]
        result = _flatten_message_content_to_strings(messages)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_does_not_mutate_original_messages(self):
        original_content = [{"type": "text", "text": "Hello"}]
        original_msg = {"role": "user", "content": original_content}
        messages = [original_msg]
        _flatten_message_content_to_strings(messages)
        # Original dict should be untouched
        assert original_msg["content"] is original_content

    def test_empty_list(self):
        result = _flatten_message_content_to_strings([])
        assert not result

    def test_none_content_left_alone(self):
        messages = [
            {"role": "assistant", "content": None},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
        result = _flatten_message_content_to_strings(messages)
        assert result[0]["content"] is None
        assert result[1]["content"] == "hi"

    def test_multimodal_content_not_flattened(self):
        """Messages with non-text parts (e.g. images) must be left as-is."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are helpful."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            },
        ]
        result = _flatten_message_content_to_strings(messages)
        # Text-only system message gets flattened
        assert result[0]["content"] == "You are helpful."
        # Multimodal user message stays as list
        assert isinstance(result[1]["content"], list)
        assert len(result[1]["content"]) == 2

    def test_extra_message_keys_preserved(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}], "name": "alice"},
        ]
        result = _flatten_message_content_to_strings(messages)
        assert result[0]["name"] == "alice"
        assert result[0]["content"] == "hi"


class TestBuildResponsesKwargsFlattening:
    """Verify _build_responses_kwargs applies content flattening."""

    def test_list_content_flattened_in_input(self):
        params = {
            "model": "xai/grok-test",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ],
            "timeout": 60,
        }
        kwargs = _build_responses_kwargs(params)
        # Both should be plain strings
        assert kwargs["input"][0]["content"] == "Be helpful."
        assert kwargs["input"][1]["content"] == "Hello"

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

    def test_reasoning_effort_is_forwarded(self):
        params = {
            "model": "xai/grok-test",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "high",
        }

        kwargs = _build_responses_kwargs(params)

        assert kwargs["reasoning_effort"] == "high"

    def test_reasoning_config_is_forwarded(self):
        params = {
            "model": "openai/gpt-5.6-sol",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning": {"effort": "max", "mode": "pro"},
        }

        kwargs = _build_responses_kwargs(params)

        assert kwargs["reasoning"] == {"effort": "max", "mode": "pro"}

    def test_explicit_http_client_is_forwarded(self):
        client = object()
        params = {
            "model": "openai/gpt-5.6-sol",
            "messages": [{"role": "user", "content": "Hello"}],
            "client": client,
        }

        kwargs = _build_responses_kwargs(params)

        assert kwargs["client"] is client
