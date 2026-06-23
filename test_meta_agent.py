# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import tempfile

import pytest

from meta_agent import _read_eval_summaries


class TestReadEvalSummaries:
    """Tests for _read_eval_summaries helper."""

    def test_returns_none_for_none_path(self):
        assert _read_eval_summaries(None) is None

    def test_returns_none_for_empty_string(self):
        assert _read_eval_summaries("") is None

    def test_returns_none_for_nonexistent_directory(self):
        assert _read_eval_summaries("/tmp/this_dir_does_not_exist_12345") is None

    def test_returns_none_for_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert _read_eval_summaries(tmpdir) is None

    def test_extracts_score_fields_from_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_file = os.path.join(tmpdir, "result.json")
            with open(result_file, "w") as f:
                json.dump(
                    {"pass_rate": 0.73, "avg_score": 12.5, "model_name": "test"},
                    f,
                )
            summary = _read_eval_summaries(tmpdir)
            assert summary is not None
            assert "pass_rate" in summary
            assert "0.73" in summary
            assert "avg_score" in summary
            assert "model_name" not in summary  # not a score-like field

    def test_extracts_md_preview(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_file = os.path.join(tmpdir, "eval_report.md")
            with open(md_file, "w") as f:
                f.write("# Evaluation Results\n\nScore: 85/100\n\nDetails follow.")
            summary = _read_eval_summaries(tmpdir)
            assert summary is not None
            assert "Evaluation Results" in summary
            assert "Score: 85/100" in summary

    def test_ignores_irrelevant_json_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_file = os.path.join(tmpdir, "metadata.json")
            with open(result_file, "w") as f:
                json.dump({"timestamp": "2026-01-01", "version": "1.0"}, f)
            # No score-like keys → summary should be None
            assert _read_eval_summaries(tmpdir) is None

    def test_mixed_json_and_md_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "scores.json"), "w") as f:
                json.dump({"accuracy": 0.92}, f)
            with open(os.path.join(tmpdir, "notes.md"), "w") as f:
                f.write("# Notes\nSome observations.")
            summary = _read_eval_summaries(tmpdir)
            assert "accuracy" in summary
            assert "Notes" in summary

    def test_handles_corrupt_json_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bad.json"), "w") as f:
                f.write("{not valid json")
            # Corrupt file is skipped, no crash
            assert _read_eval_summaries(tmpdir) is None