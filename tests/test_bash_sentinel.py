"""Tests for BashSession random sentinel (F-14).

Validates that each BashSession instance gets a
unique, random sentinel instead of the static
'<<exit>>' string, preventing command output from
accidentally matching the sentinel.
"""

import re
import sys

import pytest

from agent.tools.bash import BashSession


class TestBashSentinelUniqueness:
    """F-14: Each BashSession gets a unique
    random sentinel."""

    def test_sentinel_is_not_static(self):
        """Sentinel must not be the old static
        '<<exit>>' string."""
        session = BashSession()
        assert session._sentinel != "<<exit>>"
        assert session._sentinel != "<<EXIT>>"

    def test_sentinel_matches_expected_pattern(self):
        """Sentinel matches <<EXIT_[hex]>> format."""
        session = BashSession()
        pattern = r"^<<EXIT_[0-9a-f]+>>$"
        assert re.match(pattern, session._sentinel), (
            f"Sentinel {session._sentinel!r} does not "
            f"match pattern {pattern}"
        )

    def test_each_instance_gets_unique_sentinel(self):
        """Two separate BashSession instances must have
        different sentinels."""
        s1 = BashSession()
        s2 = BashSession()
        assert s1._sentinel != s2._sentinel, (
            "Two sessions should not share a sentinel"
        )

    def test_many_instances_all_unique(self):
        """Creating 50 sessions yields 50 distinct
        sentinels."""
        sentinels = {
            BashSession()._sentinel for _ in range(50)
        }
        assert len(sentinels) == 50

    def test_sentinel_hex_length(self):
        """The hex portion has 32 chars (uuid4.hex)."""
        session = BashSession()
        # <<EXIT_{32 hex chars}>>
        inner = session._sentinel[7:-2]  # strip <<EXIT_ and >>
        assert len(inner) == 32
        assert all(c in "0123456789abcdef" for c in inner)

    def test_sentinel_used_in_run_command(self):
        """The run() method references self._sentinel,
        not a hardcoded string."""
        import inspect
        src = inspect.getsource(BashSession.run)
        assert "self._sentinel" in src
        assert "<<exit>>" not in src.lower()
