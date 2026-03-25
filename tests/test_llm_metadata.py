"""Tests for LLM response metadata (F-10).

Validates that get_response_from_llm() returns an
info dict with expected keys: finish_reason, usage,
model.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest

# conftest.py mocks backoff, litellm, dotenv, etc.
from agent.llm import get_response_from_llm


class TestLlmMetadataKeys:
    """F-10: info dict contains expected keys."""

    def test_return_annotation_is_tuple(self):
        """get_response_from_llm returns a 3-tuple
        (text, history, info)."""
        sig = inspect.signature(
            get_response_from_llm
        )
        ret = sig.return_annotation
        assert "Tuple" in str(ret)

    def test_info_dict_constructed_in_source(self):
        """Source constructs info with finish_reason,
        usage, model keys."""
        src = inspect.getsource(
            get_response_from_llm
        )
        assert '"finish_reason"' in src
        assert '"usage"' in src
        assert '"model"' in src

    def test_info_dict_is_returned(self):
        """The function returns (response_text,
        new_msg_history, info)."""
        src = inspect.getsource(
            get_response_from_llm
        )
        assert (
            "return response_text, "
            "new_msg_history, info"
        ) in src

    def test_info_structure_via_mock(self):
        """Mock litellm.completion and verify the
        returned info dict shape."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "hello"
        )
        mock_response.choices[0].finish_reason = (
            "stop"
        )
        mock_response.usage = MagicMock()
        mock_response.model = "test-model"

        # Make response subscriptable for the
        # response['choices'][0]['message']['content']
        # pattern used in the source.
        choice_msg = {"content": "hello"}
        choice = {"message": choice_msg}

        def getitem(self, key):
            if key == "choices":
                return [choice]
            return None

        type(mock_response).__getitem__ = getitem

        with patch(
            "agent.llm.litellm.completion",
            return_value=mock_response,
        ):
            text, history, info = (
                get_response_from_llm(
                    msg="test", model="test-model"
                )
            )

        assert isinstance(info, dict)
        assert "finish_reason" in info
        assert "usage" in info
        assert "model" in info
        assert info["finish_reason"] == "stop"
        assert info["model"] == "test-model"
        assert text == "hello"
