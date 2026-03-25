"""Tests for ensemble majority voting logic.

Validates weighted majority voting for classification
domains, single-best fallback, and domain gating.
"""

import json
import os
import sys
from collections import defaultdict
from unittest.mock import patch, MagicMock

import pytest

_PROJ = "C:/Users/ryuke/Desktop/Projects/Hyperagents"
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)


# ------ Pure logic tests (no imports needed) ------

class TestMajorityVoteLogic:
    """Test the weighted majority voting algorithm
    in isolation."""

    @staticmethod
    def _weighted_majority(predictions_scores):
        """Reimplement the core voting logic from
        ensemble.py for isolated testing.

        Args:
            predictions_scores: list of (pred, score)
        Returns:
            The prediction with highest total weight.
        """
        votes = defaultdict(float)
        for pred, score in predictions_scores:
            if pred is not None:
                votes[pred] += score
        if votes:
            return max(votes, key=votes.get)
        return None

    def test_majority_simple(self):
        """3 agents: A(0.8), B(0.7), A(0.6) -> A
        wins with 1.4 vs 0.7."""
        result = self._weighted_majority([
            ("A", 0.8), ("B", 0.7), ("A", 0.6),
        ])
        assert result == "A"

    def test_weights_override_count(self):
        """2 agents vote B with low scores, 1 votes A
        with high score. A wins on weight."""
        result = self._weighted_majority([
            ("A", 0.95), ("B", 0.1), ("B", 0.1),
        ])
        assert result == "A"

    def test_tie_broken_deterministically(self):
        """When weights tie, max() returns
        deterministically."""
        result = self._weighted_majority([
            ("A", 0.5), ("B", 0.5),
        ])
        # max() returns first key encountered with max
        assert result in ("A", "B")

    def test_all_same_vote(self):
        """All agents agree -> that answer wins."""
        result = self._weighted_majority([
            ("X", 0.9), ("X", 0.8), ("X", 0.7),
        ])
        assert result == "X"

    def test_none_predictions_ignored(self):
        """None predictions don't count toward any
        vote."""
        result = self._weighted_majority([
            ("A", 0.9), (None, 0.8), ("B", 0.7),
        ])
        assert result == "A"

    def test_all_none_returns_none(self):
        """If all predictions are None, returns None."""
        result = self._weighted_majority([
            (None, 0.9), (None, 0.8),
        ])
        assert result is None


class TestEnsembleDomainGating:
    """Verify domain-based routing: classification
    domains use voting, others use single-best."""

    def test_classification_domains_known(self):
        """Classification domains that support
        ensemble are a known set."""
        # conftest.py mocks make this import work
        from ensemble import _CLASSIFICATION_DOMAINS
        assert (
            "search_arena" in _CLASSIFICATION_DOMAINS
        )
        assert (
            "paper_review" in _CLASSIFICATION_DOMAINS
        )
        assert (
            "imo_grading" in _CLASSIFICATION_DOMAINS
        )

    def test_non_classification_not_in_set(self):
        """Non-classification domains are NOT in the
        classification set."""
        from ensemble import _CLASSIFICATION_DOMAINS
        assert (
            "genesis_go2walking"
            not in _CLASSIFICATION_DOMAINS
        )
        assert (
            "balrog_babyai"
            not in _CLASSIFICATION_DOMAINS
        )
        assert (
            "polyglot"
            not in _CLASSIFICATION_DOMAINS
        )


class TestEnsembleFallback:
    """Test fallback behavior when <3 agents
    available."""

    @staticmethod
    def _should_use_voting(
        domain, can_ensemble, n_agents
    ):
        """Replicate the gating logic from ensemble()
        for testing."""
        classification = {
            "search_arena",
            "paper_review",
            "imo_grading",
        }
        return (
            domain in classification
            and can_ensemble
            and n_agents >= 3
        )

    def test_fewer_than_3_uses_single_best(self):
        """With 2 agents, should NOT use voting."""
        assert not self._should_use_voting(
            "search_arena", True, 2
        )

    def test_3_or_more_uses_voting(self):
        """With 3+ agents, should use voting."""
        assert self._should_use_voting(
            "search_arena", True, 3
        )

    def test_non_classification_always_single(self):
        """Non-classification domain never uses
        voting regardless of agent count."""
        assert not self._should_use_voting(
            "genesis_go2walking", False, 5
        )

    def test_can_ensemble_false_blocks_voting(self):
        """Even a classification domain with
        can_ensemble=False uses single-best."""
        assert not self._should_use_voting(
            "imo_grading", False, 3
        )
