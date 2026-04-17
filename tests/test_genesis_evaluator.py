"""Tests for genesis Evaluator (F-01, F-02).

F-01: run_episode() must return a dict (not None).
F-02: The parallel worker call signature must match
      the run_episode method signature.
"""

import importlib.util
import inspect
import os
import sys

import pytest

_PROJ = os.path.normpath(
    "C:/Users/ryuke/Desktop/Projects/Hyperagents"
)


def _load_module_from_file(module_name, rel_path):
    """Load a module directly from file, bypassing
    package __init__.py (avoids torch import)."""
    abs_path = os.path.join(_PROJ, rel_path)
    spec = importlib.util.spec_from_file_location(
        module_name, abs_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# conftest.py already mocked tqdm at import time.
_mod = _load_module_from_file(
    "domains.genesis.evaluator",
    "domains/genesis/evaluator.py",
)
Evaluator = _mod.Evaluator
EvaluatorManager = _mod.EvaluatorManager


class TestRunEpisodeReturnType:
    """F-01: run_episode must return a dict."""

    def test_run_episode_returns_dict(self):
        """run_episode should always return a dict
        containing episode log data, never None."""
        src = inspect.getsource(
            Evaluator.run_episode
        )
        assert "return episode_log" in src, (
            "run_episode must return episode_log dict"
        )
        lines = src.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped == "return":
                pytest.fail(
                    "Bare return found -- would yield "
                    "None"
                )
            if stripped == "return None":
                pytest.fail(
                    "Explicit return None found"
                )

    def test_episode_log_initialized_as_dict(self):
        """episode_log is created as a dict literal
        in run_episode."""
        src = inspect.getsource(
            Evaluator.run_episode
        )
        assert "episode_log" in src
        assert "episode_log = {" in src or (
            "episode_log = {**" in src
        )


class TestWorkerCallSignature:
    """F-02: _worker call args must match
    run_episode params."""

    def test_worker_calls_run_episode_correctly(
        self,
    ):
        """The _worker method must pass the right
        keyword args to run_episode."""
        worker_src = inspect.getsource(
            EvaluatorManager._worker
        )
        assert "run_episode(" in worker_src
        assert "process_num=" in worker_src
        assert "position=" in worker_src
        assert "episode_idx=" in worker_src

    def test_run_episode_signature_params(self):
        """run_episode accepts task, agent,
        process_num, position, episode_idx."""
        sig = inspect.signature(
            Evaluator.run_episode
        )
        params = list(sig.parameters.keys())
        assert params[0] == "self"
        assert "task" in params
        assert "agent" in params
        assert "process_num" in params
        assert "position" in params
        assert "episode_idx" in params

    def test_run_parallel_passes_position(self):
        """_run_parallel must pass a position arg
        to each _worker call."""
        src = inspect.getsource(
            EvaluatorManager._run_parallel
        )
        assert "target=self._worker" in src
        assert "position" in src

    def test_sequential_calls_run_episode(self):
        """_run_sequential must call run_episode
        with episode_idx."""
        src = inspect.getsource(
            EvaluatorManager._run_sequential
        )
        assert "run_episode(" in src
        assert "episode_idx=" in src

    def test_worker_signature_has_4_args(self):
        """_worker takes (self, task_queue,
        results_queue, agent_factory, position)."""
        sig = inspect.signature(
            EvaluatorManager._worker
        )
        params = list(sig.parameters.keys())
        assert "task_queue" in params
        assert "results_queue" in params
        assert "agent_factory" in params
        assert "position" in params
