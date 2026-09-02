"""Tests for meta_agent.py fix (Issue #30).

Verifies that MetaAgent.forward wires eval_path and iterations_left
into the instruction instead of dropping them.
"""

import unittest.mock as mock

from meta_agent import MetaAgent


def _make_meta_agent():
    """Create a MetaAgent with mocked base dependencies."""
    with mock.patch("meta_agent.AgentSystem.__init__", lambda self, **kw: None):
        agent = MetaAgent.__new__(MetaAgent)
        agent.model = "test-model"
        agent.log = lambda *_a, **_kw: None
        return agent


def test_forward_includes_repo_path():
    agent = _make_meta_agent()
    with mock.patch("meta_agent.chat_with_agent") as m:
        m.return_value = []
        agent.forward(repo_path="/tmp/repo", eval_path="", iterations_left=None)
        instruction = m.call_args[0][0]
        assert "/tmp/repo" in instruction


def test_forward_includes_eval_path_when_truthy():
    agent = _make_meta_agent()
    with mock.patch("meta_agent.chat_with_agent") as m:
        m.return_value = []
        agent.forward(
            repo_path="/tmp/repo",
            eval_path="/tmp/evals",
            iterations_left=None,
        )
        instruction = m.call_args[0][0].lower()
        assert "/tmp/evals" in instruction
        assert "evaluation results" in instruction


def test_forward_omits_eval_path_when_falsy():
    agent = _make_meta_agent()
    with mock.patch("meta_agent.chat_with_agent") as m:
        m.return_value = []
        agent.forward(repo_path="/tmp/repo", eval_path="", iterations_left=None)
        instruction = m.call_args[0][0].lower()
        # Should not mention evaluation results when eval_path is ""
        assert "evaluation results" not in instruction


def test_forward_includes_iterations_left_when_not_none():
    agent = _make_meta_agent()
    with mock.patch("meta_agent.chat_with_agent") as m:
        m.return_value = []
        agent.forward(
            repo_path="/tmp/repo", eval_path="", iterations_left=5
        )
        instruction = m.call_args[0][0].lower()
        assert "5" in instruction
        assert "remaining" in instruction


def test_forward_omits_iterations_left_when_none():
    agent = _make_meta_agent()
    with mock.patch("meta_agent.chat_with_agent") as m:
        m.return_value = []
        agent.forward(
            repo_path="/tmp/repo", eval_path="", iterations_left=None
        )
        instruction = m.call_args[0][0].lower()
        assert "remaining" not in instruction


def test_forward_uses_chat_with_agent_tools_all():
    agent = _make_meta_agent()
    with mock.patch("meta_agent.chat_with_agent") as m:
        m.return_value = []
        agent.forward(
            repo_path="/tmp/repo",
            eval_path="/tmp/evals",
            iterations_left=2,
        )
        assert m.call_args[1]["tools_available"] == "all"
