"""Tests for run_smoke_test() (F-15).

Validates that run_smoke_test correctly interprets
container.exec_run results: True on success,
False on non-zero exit code or missing sentinel.
"""

from unittest.mock import MagicMock, patch

import pytest

# conftest.py mocks docker, etc.
from utils.gl_utils import run_smoke_test


def _make_mock_container(exit_code, output_text):
    """Create a mock Docker container whose exec_run
    returns the given exit code and output."""
    container = MagicMock()
    exec_result = MagicMock()
    exec_result.exit_code = exit_code
    exec_result.output = output_text.encode("utf-8")
    container.exec_run.return_value = exec_result
    return container


class TestRunSmokeTest:
    """F-15: run_smoke_test container validation."""

    def _run(self, exit_code, output):
        """Helper: run smoke test with mock."""
        container = _make_mock_container(
            exit_code, output
        )
        with patch(
            "utils.gl_utils.log_container_output",
            lambda *a, **kw: None,
        ):
            return run_smoke_test(container)

    def test_success_returns_true(self):
        """Exit code 0 + sentinel present -> True."""
        result = self._run(
            0, "some output\nsmoke_test_passed\n"
        )
        assert result is True

    def test_nonzero_exit_returns_false(self):
        """Non-zero exit code -> False."""
        result = self._run(
            1, "smoke_test_passed\n"
        )
        assert result is False

    def test_missing_sentinel_returns_false(self):
        """Exit code 0 but no sentinel string ->
        False."""
        result = self._run(
            0, "import succeeded\n"
        )
        assert result is False

    def test_empty_output_returns_false(self):
        """Empty output -> False."""
        result = self._run(0, "")
        assert result is False

    def test_exception_returns_false(self):
        """If exec_run raises, returns False."""
        container = MagicMock()
        container.exec_run.side_effect = (
            RuntimeError("docker error")
        )
        with patch(
            "utils.gl_utils.log_container_output",
            lambda *a, **kw: None,
        ):
            result = run_smoke_test(container)
        assert result is False

    def test_calls_exec_run_with_command(self):
        """exec_run is called with a command list
        including 'python' and '-c'."""
        container = _make_mock_container(
            0, "smoke_test_passed"
        )
        with patch(
            "utils.gl_utils.log_container_output",
            lambda *a, **kw: None,
        ):
            run_smoke_test(container)

        call_args = container.exec_run.call_args
        cmd = call_args.kwargs.get(
            "cmd",
            call_args.args[0]
            if call_args.args
            else None,
        )
        assert "python" in cmd
        assert "-c" in cmd
