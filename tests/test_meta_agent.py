"""
Tests for meta_agent.py — verifies that the documented parameters
`eval_path` and `iterations_left` are surfaced to the agent in the
generated instruction string.

These tests intentionally do NOT exercise the underlying chat_with_agent
implementation — they only check that MetaAgent.forward composes the
correct instruction and passes it through. Mocking chat_with_agent
keeps the tests fast and free of external dependencies.
"""

from unittest.mock import patch

import pytest

from meta_agent import MetaAgent


@pytest.fixture
def meta_agent(tmp_path):
    """Construct a MetaAgent with a temporary chat history file."""
    return MetaAgent(
        model="mock-model",
        chat_history_file=str(tmp_path / "chat_history.md"),
    )


@patch("meta_agent.chat_with_agent")
def test_instruction_includes_repo_path(mock_chat, meta_agent):
    """The instruction always references the target repo_path."""
    meta_agent.forward(
        repo_path="/path/to/repo",
        eval_path="/evals",
        iterations_left=5,
    )
    instruction = mock_chat.call_args[0][0]
    assert "/path/to/repo" in instruction


@patch("meta_agent.chat_with_agent")
def test_instruction_mentions_eval_path_when_provided(mock_chat, meta_agent):
    """When eval_path is provided, the instruction tells the agent where prior results live."""
    meta_agent.forward(
        repo_path="/repo",
        eval_path="/path/to/evals",
        iterations_left=None,
    )
    instruction = mock_chat.call_args[0][0]
    assert "/path/to/evals" in instruction
    assert "evaluation results" in instruction.lower()


@patch("meta_agent.chat_with_agent")
def test_instruction_omits_eval_section_when_eval_path_falsy(mock_chat, meta_agent):
    """When eval_path is empty (falsy), the eval-results section is skipped."""
    meta_agent.forward(
        repo_path="/repo",
        eval_path="",
        iterations_left=None,
    )
    instruction = mock_chat.call_args[0][0]
    assert "evaluation results" not in instruction.lower()


@patch("meta_agent.chat_with_agent")
def test_instruction_includes_iterations_left_when_provided(mock_chat, meta_agent):
    """When iterations_left is set, the instruction surfaces the numeric budget."""
    meta_agent.forward(
        repo_path="/repo",
        eval_path="/evals",
        iterations_left=7,
    )
    instruction = mock_chat.call_args[0][0]
    assert "7" in instruction
    assert "remaining" in instruction.lower()


@patch("meta_agent.chat_with_agent")
def test_instruction_omits_iteration_hint_when_none(mock_chat, meta_agent):
    """When iterations_left=None (the documented default), no budget hint appears."""
    meta_agent.forward(
        repo_path="/repo",
        eval_path="/evals",
        iterations_left=None,
    )
    instruction = mock_chat.call_args[0][0]
    assert "remaining" not in instruction.lower()
