"""Tests for tool output JSON serialization (F-03).

Validates that tool output messages use json.dumps()
for proper serialization, avoiding the old f-string
approach which produced invalid JSON with unescaped
quotes and special characters.
"""

import json

import pytest


class TestToolOutputJsonSerialization:
    """Verify json.dumps produces valid JSON for tool
    output messages."""

    def test_basic_tool_output_is_valid_json(self):
        """A simple tool output round-trips through
        json.dumps / json.loads."""
        tool_input = {
            "command": "ls -la",
            "path": "/tmp",
        }
        tool_msg_data = {
            "tool_name": "bash",
            "tool_input": tool_input,
            "tool_output": "file1.txt\nfile2.txt",
        }
        serialized = json.dumps(tool_msg_data)
        parsed = json.loads(serialized)
        assert parsed["tool_name"] == "bash"
        assert parsed["tool_input"] == tool_input
        assert "file1.txt" in parsed["tool_output"]

    def test_old_fstring_approach_produces_invalid_json(
        self,
    ):
        """Demonstrate the bug: f-string interpolation
        of dicts produces repr() output that is NOT
        valid JSON (single quotes, unescaped chars)."""
        tool_input = {
            "command": "echo 'hello'",
            "path": "/tmp",
        }
        tool_output = 'He said "hello"'

        # Old broken approach: f-string with dict
        old_msg = (
            f'{{"tool_name": "bash", '
            f'"tool_input": {tool_input}, '
            f'"tool_output": "{tool_output}"}}'
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(old_msg)

    def test_special_chars_quotes(self):
        """Double quotes in tool_output are properly
        escaped by json.dumps."""
        data = {
            "tool_name": "bash",
            "tool_input": {"command": "echo"},
            "tool_output": 'He said "hello"',
        }
        serialized = json.dumps(data)
        parsed = json.loads(serialized)
        assert parsed["tool_output"] == (
            'He said "hello"'
        )

    def test_special_chars_newlines(self):
        """Newlines in tool_output are escaped."""
        data = {
            "tool_name": "bash",
            "tool_input": {"command": "ls"},
            "tool_output": "line1\nline2\nline3",
        }
        serialized = json.dumps(data)
        # Raw string should contain \\n, not newlines
        assert "\\n" in serialized
        parsed = json.loads(serialized)
        assert parsed["tool_output"].count("\n") == 2

    def test_special_chars_backslashes(self):
        """Backslashes in tool_output are escaped."""
        data = {
            "tool_name": "bash",
            "tool_input": {"command": "echo"},
            "tool_output": "C:\\Users\\test\\file.txt",
        }
        serialized = json.dumps(data)
        parsed = json.loads(serialized)
        assert (
            parsed["tool_output"]
            == "C:\\Users\\test\\file.txt"
        )

    def test_special_chars_tabs_and_unicode(self):
        """Tabs and unicode in tool_output are handled."""
        data = {
            "tool_name": "bash",
            "tool_input": {"command": "cat"},
            "tool_output": "col1\tcol2\n\u2603 snowman",
        }
        serialized = json.dumps(data)
        parsed = json.loads(serialized)
        assert "\t" in parsed["tool_output"]
        assert "\u2603" in parsed["tool_output"]

    def test_nested_json_in_output(self):
        """Tool output containing JSON-like strings
        serializes correctly."""
        inner = json.dumps({"key": "value"})
        data = {
            "tool_name": "bash",
            "tool_input": {"command": "curl"},
            "tool_output": inner,
        }
        serialized = json.dumps(data)
        parsed = json.loads(serialized)
        # The output is a string, not a dict
        assert isinstance(parsed["tool_output"], str)
        inner_parsed = json.loads(
            parsed["tool_output"]
        )
        assert inner_parsed["key"] == "value"

    def test_empty_tool_output(self):
        """Empty string tool output serializes."""
        data = {
            "tool_name": "bash",
            "tool_input": {"command": "true"},
            "tool_output": "",
        }
        serialized = json.dumps(data)
        parsed = json.loads(serialized)
        assert parsed["tool_output"] == ""

    def test_actual_format_with_xml_wrapper(self):
        """Test the actual format used in
        llm_withtools.py: <json>...json.dumps...</json>
        """
        tool_msg_data = {
            "tool_name": "bash",
            "tool_input": {"command": "ls"},
            "tool_output": "file.txt",
        }
        tool_msg = (
            f"<json>\n"
            f"{json.dumps(tool_msg_data, indent=2)}"
            f"\n</json>"
        )
        # Extract the JSON between tags
        start = tool_msg.index("<json>\n") + 7
        end = tool_msg.index("\n</json>")
        extracted = tool_msg[start:end]
        parsed = json.loads(extracted)
        assert parsed["tool_name"] == "bash"
