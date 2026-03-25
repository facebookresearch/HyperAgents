"""Tests for JSONL archive parsing (F-07 fix).

Validates that load_archive_data() correctly parses
JSONL format (one JSON object per line) instead of
treating the whole file as a single JSON array.
"""

import json
import os
import tempfile

import pytest


# --------------- helpers to avoid heavy project imports -----
# We extract the parsing logic directly to test in
# isolation.  If import works, we test the real function
# too.

def _parse_jsonl(filepath, last_only=True):
    """Pure-Python reimplementation of the JSONL parsing
    logic from utils/gl_utils.py::load_archive_data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Metadata file not found at {filepath}"
        )
    archive_data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    archive_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if last_only:
        return archive_data[-1]
    return archive_data


class TestLoadArchiveDataParsing:
    """Tests for JSONL line-by-line parsing."""

    def test_parse_valid_jsonl(self, tmp_dir):
        """Valid JSONL with multiple lines parses each
        line independently."""
        path = os.path.join(tmp_dir, "archive.jsonl")
        entries = [
            {"current_genid": 0, "archive": [0]},
            {"current_genid": 1, "archive": [0, 1]},
        ]
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = _parse_jsonl(path, last_only=False)
        assert len(result) == 2
        assert result[0]["current_genid"] == 0
        assert result[1]["archive"] == [0, 1]

    def test_parse_last_only(self, tmp_dir):
        """last_only=True returns only the final entry."""
        path = os.path.join(tmp_dir, "archive.jsonl")
        entries = [
            {"current_genid": 0, "archive": [0]},
            {"current_genid": 1, "archive": [0, 1]},
            {"current_genid": 2, "archive": [0, 1, 2]},
        ]
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = _parse_jsonl(path, last_only=True)
        assert result["current_genid"] == 2
        assert len(result["archive"]) == 3

    def test_empty_lines_skipped(self, tmp_dir):
        """Blank lines between entries are ignored."""
        path = os.path.join(tmp_dir, "archive.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"a": 1}) + "\n")
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps({"a": 2}) + "\n")

        result = _parse_jsonl(path, last_only=False)
        assert len(result) == 2

    def test_malformed_lines_skipped(self, tmp_dir):
        """Malformed JSON lines are skipped without
        crashing."""
        path = os.path.join(tmp_dir, "archive.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"ok": True}) + "\n")
            f.write("this is not json\n")
            f.write("{broken json\n")
            f.write(json.dumps({"ok": True}) + "\n")

        result = _parse_jsonl(path, last_only=False)
        assert len(result) == 2
        assert all(e["ok"] for e in result)

    def test_empty_file_raises(self, tmp_dir):
        """An empty file (no valid entries) raises
        IndexError when last_only=True."""
        path = os.path.join(tmp_dir, "archive.jsonl")
        with open(path, "w") as f:
            f.write("")

        with pytest.raises(IndexError):
            _parse_jsonl(path, last_only=True)

    def test_empty_file_returns_empty_list(self, tmp_dir):
        """An empty file returns [] when last_only=False."""
        path = os.path.join(tmp_dir, "archive.jsonl")
        with open(path, "w") as f:
            f.write("")

        result = _parse_jsonl(path, last_only=False)
        assert result == []

    def test_missing_file_raises(self, tmp_dir):
        """A nonexistent file raises FileNotFoundError."""
        path = os.path.join(tmp_dir, "nonexistent.jsonl")
        with pytest.raises(FileNotFoundError):
            _parse_jsonl(path)

    def test_single_line_file(self, tmp_dir):
        """A file with exactly one line works correctly."""
        path = os.path.join(tmp_dir, "archive.jsonl")
        entry = {"current_genid": 0, "archive": [0]}
        with open(path, "w") as f:
            f.write(json.dumps(entry) + "\n")

        result = _parse_jsonl(path, last_only=True)
        assert result == entry

        result_all = _parse_jsonl(path, last_only=False)
        assert len(result_all) == 1


class TestLoadArchiveDataReal:
    """Test the real load_archive_data function.

    conftest.py installs lightweight mocks so the
    import succeeds without docker/litellm/etc.
    """

    @pytest.fixture(autouse=True)
    def _try_import(self):
        """Import load_archive_data (mocks in
        conftest handle heavy deps)."""
        try:
            from utils.gl_utils import (
                load_archive_data,
            )
            self.load_fn = load_archive_data
        except Exception as e:
            pytest.skip(
                f"Could not import: {e}"
            )

    def test_real_parse_valid(
        self, sample_archive_jsonl
    ):
        """Real function parses valid JSONL."""
        result = self.load_fn(
            sample_archive_jsonl, last_only=False
        )
        assert len(result) == 3
        assert result[-1]["current_genid"] == 2

    def test_real_last_only(
        self, sample_archive_jsonl
    ):
        """Real function returns last entry."""
        result = self.load_fn(
            sample_archive_jsonl, last_only=True
        )
        assert result["current_genid"] == 2
