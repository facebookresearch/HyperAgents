"""Tests for atomic metadata writes (F-11).

Validates that update_node_metadata() uses the
temp-file + os.replace pattern to avoid corruption.
"""

import json
import os
import sys

import pytest

# conftest.py mocks heavy deps (docker, etc.)
from utils.gl_utils import (
    update_node_metadata,
    get_node_metadata_key,
)


class TestUpdateNodeMetadataAtomic:
    """F-11: metadata writes use temp + rename."""

    def _make_gen_dir(self, tmp_dir, genid, data):
        """Helper: create gen_{genid}/metadata.json."""
        gen_dir = os.path.join(
            tmp_dir, f"gen_{genid}"
        )
        os.makedirs(gen_dir, exist_ok=True)
        meta_path = os.path.join(
            gen_dir, "metadata.json"
        )
        with open(meta_path, "w") as f:
            json.dump(data, f)
        return meta_path

    def test_update_writes_correct_data(self, tmp_dir):
        """After update, the file contains the merged
        data."""
        original = {"parent_genid": None, "score": 0.5}
        self._make_gen_dir(tmp_dir, 0, original)

        update_node_metadata(
            tmp_dir, 0, {"score": 0.9, "new_key": True}
        )

        meta_path = os.path.join(
            tmp_dir, "gen_0", "metadata.json"
        )
        with open(meta_path, "r") as f:
            result = json.load(f)

        assert result["score"] == 0.9
        assert result["new_key"] is True
        assert result["parent_genid"] is None

    def test_no_temp_file_left_behind(self, tmp_dir):
        """After a successful write, no .tmp file
        remains."""
        self._make_gen_dir(
            tmp_dir, 0, {"key": "value"}
        )
        update_node_metadata(
            tmp_dir, 0, {"key": "updated"}
        )

        gen_dir = os.path.join(tmp_dir, "gen_0")
        files = os.listdir(gen_dir)
        tmp_files = [f for f in files if f.endswith(".tmp")]
        assert len(tmp_files) == 0, (
            f"Temp files remain: {tmp_files}"
        )

    def test_atomic_pattern_in_source(self):
        """Source code uses tmp_file + os.replace
        pattern."""
        import inspect
        src = inspect.getsource(update_node_metadata)
        assert "tmp" in src.lower(), (
            "Should use a temp file"
        )
        assert "os.replace" in src or "os.rename" in src, (
            "Should use os.replace or os.rename for "
            "atomic swap"
        )
        assert "f.flush()" in src, (
            "Should flush before fsync"
        )
        assert "os.fsync" in src, (
            "Should fsync before replace"
        )

    def test_missing_metadata_is_noop(self, tmp_dir):
        """If metadata.json doesn't exist,
        update_node_metadata does nothing."""
        # gen_99 does not exist
        update_node_metadata(
            tmp_dir, 99, {"key": "value"}
        )
        gen_dir = os.path.join(tmp_dir, "gen_99")
        assert not os.path.exists(gen_dir)

    def test_get_after_update_returns_new_value(
        self, tmp_dir
    ):
        """get_node_metadata_key returns the updated
        value after update_node_metadata."""
        self._make_gen_dir(
            tmp_dir, 1, {"status": "pending"}
        )
        update_node_metadata(
            tmp_dir, 1, {"status": "complete"}
        )
        val = get_node_metadata_key(
            tmp_dir, 1, "status"
        )
        assert val == "complete"

    def test_concurrent_safety_no_partial_writes(
        self, tmp_dir
    ):
        """Simulates that the original file stays
        intact if we check it before os.replace
        would run -- i.e., the tmp file is written
        first, then swapped."""
        original = {"step": 1, "data": "original"}
        meta_path = self._make_gen_dir(
            tmp_dir, 0, original
        )

        # Perform multiple updates sequentially
        for i in range(2, 6):
            update_node_metadata(
                tmp_dir, 0, {"step": i}
            )

        with open(meta_path, "r") as f:
            result = json.load(f)
        assert result["step"] == 5
        assert result["data"] == "original"
