"""
Tests for select_next_parent.py — novelty-weighted parent selection.
"""

from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest

from select_next_parent import select_next_parent


@patch("select_next_parent.get_parent_genid")
@patch("select_next_parent.get_saved_score")
@patch("select_next_parent.get_node_metadata_key")
@patch("select_next_parent.is_starting_node")
@patch("select_next_parent.get_domain_splits")
def test_novelty_weighted_selection_favors_under_explored(
    mock_splits, mock_starting, mock_meta, mock_score, mock_parent
):
    """
    Setup: two valid parent candidates with equal scores —
        - 'A' has 9 children
        - 'B' has 0 children

    Expected weights:
        weight(A) = 1 / (1 + 9) = 0.1
        weight(B) = 1 / (1 + 0) = 1.0
        P(A) = 0.1 / 1.1 ≈ 0.0909
        P(B) = 1.0 / 1.1 ≈ 0.9091

    Over 10,000 trials, the empirical frequencies should match within ±2%.
    """
    archive = ["A", "B"] + [f"child_{i}" for i in range(9)]
    parent_map = {f"child_{i}": "A" for i in range(9)}
    parent_map["A"] = None
    parent_map["B"] = None
    valid = {"A": True, "B": True, **{f"child_{i}": False for i in range(9)}}

    mock_starting.return_value = False
    mock_splits.return_value = ["val"]
    mock_meta.side_effect = lambda od, gid, key: valid.get(gid, False)
    mock_score.side_effect = (
        lambda dom, od, gid, split, type: 1.0 if gid in ("A", "B") else None
    )
    mock_parent.side_effect = lambda od, gid: parent_map.get(gid)

    np.random.seed(42)
    counts = Counter(
        select_next_parent(archive, "/tmp/fake", ["dom"]) for _ in range(10_000)
    )

    p_a = counts["A"] / 10_000
    p_b = counts["B"] / 10_000

    assert 0.07 <= p_a <= 0.11, f"P(A) = {p_a}, expected ~0.0909 ± 0.02"
    assert 0.89 <= p_b <= 0.93, f"P(B) = {p_b}, expected ~0.9091 ± 0.02"


@patch("select_next_parent.get_parent_genid")
@patch("select_next_parent.get_saved_score")
@patch("select_next_parent.get_node_metadata_key")
@patch("select_next_parent.is_starting_node")
@patch("select_next_parent.get_domain_splits")
def test_single_candidate_returns_that_candidate(
    mock_splits, mock_starting, mock_meta, mock_score, mock_parent
):
    """When only one valid candidate exists, it must always be selected."""
    mock_starting.return_value = False
    mock_splits.return_value = ["val"]
    mock_meta.return_value = True
    mock_score.return_value = 1.0
    mock_parent.return_value = None

    np.random.seed(0)
    for _ in range(100):
        assert (
            select_next_parent(["only_one"], "/tmp/fake", ["dom"]) == "only_one"
        )


@patch("select_next_parent.get_parent_genid")
@patch("select_next_parent.get_saved_score")
@patch("select_next_parent.get_node_metadata_key")
@patch("select_next_parent.is_starting_node")
@patch("select_next_parent.get_domain_splits")
def test_no_valid_candidates_raises(
    mock_splits, mock_starting, mock_meta, mock_score, mock_parent
):
    """When archive contains no valid parent candidates, raise ValueError."""
    mock_starting.return_value = False
    mock_splits.return_value = ["val"]
    mock_meta.return_value = False  # nothing is a valid parent
    mock_score.return_value = None
    mock_parent.return_value = None

    with pytest.raises(ValueError, match="No evaluation results"):
        select_next_parent(["x", "y"], "/tmp/fake", ["dom"])
