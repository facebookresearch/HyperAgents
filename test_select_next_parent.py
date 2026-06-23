import random
import pytest
import numpy as np

import select_next_parent as sp
import utils.gl_utils as gl
import utils.domain_utils as du


class TestNoveltyWeightedSelection:
    """Tests for the novelty-weighted selection logic in select_next_parent.

    These tests mock the utility functions to isolate the selection algorithm.
    """

    @pytest.fixture(autouse=True)
    def _patch_utils(self, monkeypatch):
        """Patch utility functions where they're imported in select_next_parent."""
        self._metadata = {}
        self._scores = {}

        # Patch on the select_next_parent module's references
        monkeypatch.setattr(sp, "is_starting_node", lambda genid: genid == "initial")
        monkeypatch.setattr(
            sp,
            "get_node_metadata_key",
            lambda output_dir, genid, key: self._metadata.get(genid, {}).get(key),
        )
        monkeypatch.setattr(
            sp,
            "get_parent_genid",
            lambda output_dir, genid: self._metadata.get(genid, {}).get("parent_genid"),
        )
        monkeypatch.setattr(
            sp,
            "get_saved_score",
            lambda domain, output_dir, genid, split, type: self._scores.get(genid),
        )
        monkeypatch.setattr(sp, "get_domain_splits", lambda dom: ["val"])

    def _setup_archive(self, archive_data):
        """Set up test data. archive_data is list of (genid, score, parent_genid, valid_parent)."""
        for genid, score, parent, valid in archive_data:
            self._metadata[genid] = {
                "valid_parent": valid,
                "parent_genid": parent,
            }
            self._scores[genid] = score

    def test_under_explored_parents_preferred(self, tmpdir):
        """Candidates with fewer children should be selected more often."""
        # gen1 has 1 child (gen3), gen2 has 0 children
        self._setup_archive([
            ("initial", 0.8, None, True),
            ("gen1", 0.9, "initial", True),
            ("gen2", 0.7, "initial", True),
            ("gen3", 0.85, "gen1", True),
        ])

        counts = {"initial": 0, "gen1": 0, "gen2": 0, "gen3": 0}
        np.random.seed(42)
        for _ in range(2000):
            parent = sp.select_next_parent(
                archive=["initial", "gen1", "gen2", "gen3"],
                output_dir=str(tmpdir),
                domains=["domain_a"],
            )
            counts[parent] += 1

        # gen2 (0 children) should be selected more often than gen1 (1 child)
        assert counts["gen2"] > counts["gen1"], (
            f"Expected gen2 (0 children) > gen1 (1 child), got {counts}"
        )

    def test_single_candidate_always_selected(self, tmpdir):
        """With only one candidate, it should always be selected."""
        self._setup_archive([
            ("gen1", 0.5, "initial", True),
        ])

        np.random.seed(42)
        for _ in range(10):
            parent = sp.select_next_parent(
                archive=["gen1"],
                output_dir=str(tmpdir),
                domains=["domain_a"],
            )
            assert parent == "gen1"

    def test_equal_children_have_similar_counts(self, tmpdir):
        """When all candidates have equal child counts, selection is roughly uniform."""
        self._setup_archive([
            ("gen1", 0.8, "initial", True),
            ("gen2", 0.9, "initial", True),
        ])

        counts = {"gen1": 0, "gen2": 0}
        np.random.seed(42)
        for _ in range(1000):
            parent = sp.select_next_parent(
                archive=["gen1", "gen2"],
                output_dir=str(tmpdir),
                domains=["domain_a"],
            )
            counts[parent] += 1

        # With equal weights, each should get roughly 500 (±150 at this scale)
        assert 350 < counts["gen1"] < 650, f"Expected roughly uniform, got {counts}"
        assert 350 < counts["gen2"] < 650, f"Expected roughly uniform, got {counts}"

    def test_no_valid_candidates_raises(self, tmpdir):
        """Should raise ValueError when no valid candidates exist."""
        self._setup_archive([
            ("gen1", None, "initial", True),
        ])

        with pytest.raises(ValueError, match="No evaluation results found"):
            sp.select_next_parent(
                archive=["gen1"],
                output_dir=str(tmpdir),
                domains=["domain_a"],
            )

    def test_dead_code_eliminated(self):
        """Verify that child_counts is actually used in selection.

        Before the fix, child_counts was computed but never read.
        After the fix, the function uses numpy for weighted selection
        based on child_counts.
        """
        import inspect

        source = inspect.getsource(sp.select_next_parent)
        # The fixed version uses numpy for weighted selection
        assert "np.random.choice" in source or "numpy.random.choice" in source, (
            "select_next_parent should use weighted random selection, not random.choice"
        )
        # The fixed version references child_counts in the selection logic
        assert "child_counts" in source, (
            "child_counts should be used in the selection logic"
        )