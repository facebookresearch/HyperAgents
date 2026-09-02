"""Tests for select_next_parent.py fix (Issue #29).

Verifies novelty-weighted parent selection: P(c) proportional to
1/(1+child_counts[c]) — under-explored candidates preferred.
"""

import unittest.mock as mock

import numpy as np

import select_next_parent as snp_mod


def _run_with_archive(archive, candidates, child_counts_map):
    """Helper that patches get_saved_score / get_parent etc. and runs."""
    # archive is list of genids in order inserted
    # candidates is dict genid->score (only valid parents)
    # child_counts_map is dict parent->count after building (for direct test)
    pass


def test_novelty_weighted_distribution():
    """With A having 9 children and B having 0, B should be ~90%."""
    candidates = {"gen_A": 0.8, "gen_B": 0.79}
    # archive with 9 children of A, 0 of B, plus both candidates themselves
    archive = ["gen_A", "gen_B"] + ["gen_A"] * 9

    # Patch helpers to return valid_parent=True and appropriate scores
    def fake_get_node_metadata_key(output_dir, genid, key):
        if key == "valid_parent":
            return True
        return None

    def fake_is_starting_node(genid):
        return False

    def fake_get_saved_score(domain, output_dir, genid, split="train", type="max"):
        # Only gen_A/gen_B are valid candidates; children have no score
        if genid in candidates:
            return candidates[genid]
        return None

    def fake_get_parent_genid(output_dir, genid):
        # For the 9 extra entries, parent is gen_A; otherwise None
        # We encode that archive[2:] are children of gen_A
        # But to keep mapping simple, return gen_A for those 9
        # Archive iteration in function loops over archive to build counts:
        # it checks each genid's parent.
        # Our archive list above has gen_A appearing as genid itself,
        # so we need to distinguish: we set parent for each archive entry index.
        # Simpler: mock to return gen_A for 9 of them tracked via call count.
        return None  # overridden per-call below

    # Manual counted simulation instead of mocking archive loop internals:
    # directly test the weight formula
    call_counts = {"gen_A": 9, "gen_B": 0}
    candidate_ids = list(candidates.keys())
    weights = np.array([1.0 / (1.0 + call_counts[c]) for c in candidate_ids])
    probabilities = weights / weights.sum()
    # Expected: A weight 0.1, B weight 1.0 => total 1.1 => P(A)=0.0909, P(B)=0.909
    assert abs(probabilities[0] - 0.0909) < 0.02  # gen_A
    assert abs(probabilities[1] - 0.9091) < 0.02  # gen_B

    # Also test actual function with mocked data producing same counts
    fake_parents = {
        # archive entries and their parent mapping
        # We'll enumerate 11 archive entries: idx0 gen_A (parent None), idx1 gen_B (None), idx2-10 children of A
        0: None,
        1: None,
        2: "gen_A",
        3: "gen_A",
        4: "gen_A",
        5: "gen_A",
        6: "gen_A",
        7: "gen_A",
        8: "gen_A",
        9: "gen_A",
        10: "gen_A",
    }

    def fake_get_parent_genid_indexed(output_dir, genid):
        # Use a counter to map call order to parent
        # The function iterates archive in order; we map by genid position
        # but genid duplicates make this ambiguous — we use call order
        fake_get_parent_genid_indexed.counter += 1
        idx = fake_get_parent_genid_indexed.counter - 1
        return fake_parents.get(idx, None)

    fake_get_parent_genid_indexed.counter = 0

    archive_for_func = [
        "gen_A",
        "gen_B",
        "child_0",
        "child_1",
        "child_2",
        "child_3",
        "child_4",
        "child_5",
        "child_6",
        "child_7",
        "child_8",
    ]

    with mock.patch.object(
        snp_mod, "get_node_metadata_key", side_effect=fake_get_node_metadata_key
    ), mock.patch.object(
        snp_mod, "is_starting_node", side_effect=fake_is_starting_node
    ), mock.patch.object(
        snp_mod, "get_saved_score", side_effect=fake_get_saved_score
    ), mock.patch.object(
        snp_mod, "get_parent_genid", side_effect=fake_get_parent_genid_indexed
    ), mock.patch.object(
        snp_mod, "get_domain_splits", return_value=["train"]
    ):
        np.random.seed(0)
        counts = {"gen_A": 0, "gen_B": 0}
        trials = 2000
        for _ in range(trials):
            fake_get_parent_genid_indexed.counter = 0
            parent = snp_mod.select_next_parent(
                archive_for_func, "/tmp/out", ["polyglot"]
            )
            counts[parent] += 1
        p_a = counts["gen_A"] / trials
        p_b = counts["gen_B"] / trials
        assert 0.05 < p_a < 0.15, f"P(gen_A)={p_a} expected ~0.09"
        assert 0.85 < p_b < 0.95, f"P(gen_B)={p_b} expected ~0.91"


def test_single_candidate_deterministic():
    """Single candidate should always be selected regardless of child count."""
    candidates = {"only_gen": 0.9}
    archive = ["only_gen"]

    with mock.patch.object(
        snp_mod, "get_node_metadata_key", return_value=True
    ), mock.patch.object(snp_mod, "is_starting_node", return_value=False), mock.patch.object(
        snp_mod, "get_saved_score", return_value=0.9
    ), mock.patch.object(
        snp_mod, "get_parent_genid", return_value=None
    ), mock.patch.object(
        snp_mod, "get_domain_splits", return_value=["train"]
    ):
        for _ in range(5):
            parent = snp_mod.select_next_parent(
                archive, "/tmp/out", ["polyglot"]
            )
            assert parent == "only_gen"


def test_empty_candidates_raises():
    """No valid parent should raise ValueError."""
    with mock.patch.object(
        snp_mod, "get_node_metadata_key", return_value=False
    ), mock.patch.object(
        snp_mod, "is_starting_node", return_value=False
    ), mock.patch.object(
        snp_mod, "get_domain_splits", return_value=["train"]
    ):
        try:
            snp_mod.select_next_parent(["gen_1"], "/tmp/out", ["polyglot"])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No evaluation" in str(e)
