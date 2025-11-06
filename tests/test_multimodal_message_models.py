"""
Tests for multimodal message models (OpenAI Vision API format)
"""

# Allow small test classes with a single test method
# pylint: disable=too-few-public-methods

import pytest
from pydantic import ValidationError

from mom_service.endpoints.models import (
    ChatMessage,
    ImageContentPart,
    ImageUrlDetail,
    TextContentPart,
)


class TestTextContentPart:
    """Tests for TextContentPart model"""

    def test_valid_text_content_part(self):
        """Test creating a valid text content part"""
        part = TextContentPart(type="text", text="Hello world")
        assert part.type == "text"
        assert part.text == "Hello world"

    def test_text_content_part_dict(self):
        """Test serialization to dict"""
        part = TextContentPart(type="text", text="Hello")
        data = part.model_dump()
        assert data == {"type": "text", "text": "Hello"}


class TestImageContentPart:
    """Tests for ImageContentPart model"""

    def test_valid_image_content_part(self):
        """Test creating a valid image content part"""
        part = ImageContentPart(
            type="image_url", image_url=ImageUrlDetail(url="https://example.com/image.jpg")
        )
        assert part.type == "image_url"
        assert part.image_url.url == "https://example.com/image.jpg"
        assert part.image_url.detail == "auto"

    def test_image_content_part_with_detail(self):
        """Test image content part with detail level"""
        part = ImageContentPart(
            type="image_url",
            image_url=ImageUrlDetail(url="https://example.com/image.jpg", detail="high"),
        )
        assert part.image_url.detail == "high"

    def test_image_content_part_dict(self):
        """Test serialization to dict"""
        part = ImageContentPart(
            type="image_url",
            image_url=ImageUrlDetail(url="https://example.com/image.jpg", detail="low"),
        )
        data = part.model_dump()
        assert data == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg", "detail": "low"},
        }


class TestChatMessageSimpleContent:
    """Tests for ChatMessage with simple string content"""

    def test_simple_text_message(self):
        """Test chat message with simple text content"""
        msg = ChatMessage(role="user", content="Hello, how are you?")
        assert msg.role == "user"
        assert msg.content == "Hello, how are you?"
        assert isinstance(msg.content, str)

    def test_simple_text_message_dict(self):
        """Test serialization of simple text message"""
        msg = ChatMessage(role="assistant", content="I'm doing well!")
        data = msg.model_dump(exclude_none=True)
        # After changing images from Optional[list] to list with default [],
        # the images field is now always present (as an empty list)
        assert data == {"role": "assistant", "content": "I'm doing well!", "images": []}


class TestChatMessageMultimodalContent:
    """Tests for ChatMessage with multimodal content (list of parts)"""

    def test_multimodal_message_text_and_image(self):
        """Test chat message with text and image content"""
        msg = ChatMessage(
            role="user",
            content=[
                TextContentPart(type="text", text="What's in this image?"),
                ImageContentPart(
                    type="image_url",
                    image_url=ImageUrlDetail(url="https://example.com/image.jpg"),
                ),
            ],
        )
        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextContentPart)
        assert isinstance(msg.content[1], ImageContentPart)

    def test_multimodal_message_dict_serialization(self):
        """Test that multimodal message serializes correctly to dict"""
        msg = ChatMessage(
            role="user",
            content=[
                TextContentPart(type="text", text="Analyze this"),
                ImageContentPart(
                    type="image_url",
                    image_url=ImageUrlDetail(url="https://example.com/image.jpg", detail="high"),
                ),
            ],
        )
        data = msg.model_dump(exclude_none=True)

        assert data["role"] == "user"
        assert isinstance(data["content"], list)
        assert len(data["content"]) == 2
        assert data["content"][0] == {"type": "text", "text": "Analyze this"}
        assert data["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg", "detail": "high"},
        }

    def test_multimodal_message_from_dict(self):
        """Test creating multimodal message from dict (request parsing)"""
        data = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://capsulos.example.com/image.jpg",
                        "detail": "auto",
                    },
                },
            ],
        }

        msg = ChatMessage(**data)
        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_multimodal_message_multiple_images(self):
        """Test message with multiple images"""
        msg = ChatMessage(
            role="user",
            content=[
                TextContentPart(type="text", text="Compare these images"),
                ImageContentPart(
                    type="image_url",
                    image_url=ImageUrlDetail(url="https://example.com/img1.jpg"),
                ),
                ImageContentPart(
                    type="image_url",
                    image_url=ImageUrlDetail(url="https://example.com/img2.jpg"),
                ),
            ],
        )
        assert len(msg.content) == 3
        assert all(isinstance(part, (TextContentPart, ImageContentPart)) for part in msg.content)

    def test_multimodal_message_only_images(self):
        """Test message with only images (no text)"""
        msg = ChatMessage(
            role="user",
            content=[
                ImageContentPart(
                    type="image_url",
                    image_url=ImageUrlDetail(url="https://example.com/image.jpg"),
                ),
            ],
        )
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ImageContentPart)


class TestChatMessageValidation:
    """Tests for validation of ChatMessage"""

    def test_invalid_role(self):
        """Test that invalid role raises validation error"""
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid", content="Hello")

    def test_empty_content_list(self):
        """Test that empty content list is allowed"""
        msg = ChatMessage(role="user", content=[])
        assert msg.content == []

    def test_mixed_content_types(self):
        """Test that we can mix text and image content parts"""
        msg = ChatMessage(
            role="user",
            content=[
                TextContentPart(type="text", text="First text"),
                ImageContentPart(
                    type="image_url",
                    image_url=ImageUrlDetail(url="https://example.com/img.jpg"),
                ),
                TextContentPart(type="text", text="Second text"),
            ],
        )
        assert len(msg.content) == 3


class TestRealWorldMultimodalRequest:
    """Test with actual multimodal request format"""

    def test_lobe_chat_format(self):
        """Test the exact format from the user's request"""
        request_data = {
            "role": "user",
            "content": [
                {
                    "text": "what do you see?",
                    "type": "text",
                },
                {
                    "image_url": {
                        "detail": "auto",
                        "url": "https://capsulos.wand-magic.stream/lobe/files/489564/c1cce594-31dd-476e-9c1f-64fb58390b04.jpg",
                    },
                    "type": "image_url",
                },
            ],
        }

        # This should parse without errors
        msg = ChatMessage(**request_data)
        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

        # Verify serialization works
        data = msg.model_dump(exclude_none=True)
        assert data["role"] == "user"
        assert len(data["content"]) == 2
        assert data["content"][0]["type"] == "text"
        assert data["content"][1]["type"] == "image_url"
