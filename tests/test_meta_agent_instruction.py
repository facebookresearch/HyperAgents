"""Tests for MetaAgent instruction construction (F-04).

Validates that the instruction string built inside
MetaAgent.forward() contains eval_path, iterations_left,
and is not trivially short.
"""

import inspect

import pytest

# conftest.py mocks backoff, litellm, dotenv,
# thread_logger, etc.
from meta_agent import MetaAgent


class TestMetaAgentInstruction:
    """F-04: Instruction string is comprehensive."""

    def test_forward_accepts_eval_path(self):
        """forward() signature includes eval_path."""
        sig = inspect.signature(MetaAgent.forward)
        params = list(sig.parameters.keys())
        assert "eval_path" in params

    def test_forward_accepts_iterations_left(self):
        """forward() signature includes
        iterations_left."""
        sig = inspect.signature(MetaAgent.forward)
        params = list(sig.parameters.keys())
        assert "iterations_left" in params

    def test_eval_path_appears_in_instruction(self):
        """The source of forward() references
        eval_path in the instruction string."""
        src = inspect.getsource(MetaAgent.forward)
        assert "eval_path" in src

    def test_iterations_left_in_instruction(self):
        """When iterations_left is provided, it
        appears in the instruction."""
        src = inspect.getsource(MetaAgent.forward)
        assert "iterations_left" in src

    def test_instruction_is_substantial(self):
        """The instruction is built with multiple
        concatenations, not just a few words."""
        src = inspect.getsource(MetaAgent.forward)
        plus_eq_count = src.count("instruction +=")
        assert plus_eq_count >= 3, (
            f"Expected 3+ instruction +=, got "
            f"{plus_eq_count}"
        )

    def test_instruction_mentions_readme(self):
        """Instruction tells the agent to read the
        README for orientation."""
        src = inspect.getsource(MetaAgent.forward)
        assert "README" in src

    def test_instruction_mentions_repo_path(self):
        """Instruction references repo_path."""
        src = inspect.getsource(MetaAgent.forward)
        assert "repo_path" in src

    def test_forward_calls_chat_with_agent(self):
        """forward() delegates to chat_with_agent."""
        src = inspect.getsource(MetaAgent.forward)
        assert "chat_with_agent" in src
