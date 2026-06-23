import random

import numpy as np

from utils.gl_utils import (
    get_node_metadata_key,
    is_starting_node,
    get_saved_score,
    get_parent_genid,
)
from utils.domain_utils import get_domain_splits


def select_next_parent(archive, output_dir, domains):
    """
    Selects the next parent to continue open-ended exploration.

    Uses novelty-weighted selection: candidates with fewer children are
    preferentially picked so the search explores under-explored branches
    rather than repeatedly sampling the same parent. This is the standard
    mechanism used by FunSearch, MAP-Elites, and AlphaEvolve to prevent
    mode collapse during open-ended search.

    Args:
        archive (list): List of generations in the archive.
        output_dir (str): Output directory for the generation.
        domains (list): List of domains to consider.
    
    Returns:
        str: The selected parent.
    """
    # Get candidate scores (averaged across domains)
    candidates = {}
    for genid in archive:
        # Skip non-valid parents
        valid_parent = (
            get_node_metadata_key(output_dir, genid, "valid_parent")
            if not is_starting_node(genid)
            else True
        )
        if not valid_parent:
            continue
        # Get per-domain scores
        per_domain_scores = []
        for dom in domains:
            split = "val" if "val" in get_domain_splits(dom) else "train"
            score = get_saved_score(dom, output_dir, genid, split=split, type="max")
            per_domain_scores.append(score)
        if per_domain_scores and all(score is not None for score in per_domain_scores):
            candidates[genid] = sum(per_domain_scores) / len(per_domain_scores)

    if not candidates:
        raise ValueError("No evaluation results found in archive.")

    # Build child counts from metadata
    child_counts = {genid: 0 for genid in candidates}
    for genid in archive:
        parent = get_parent_genid(output_dir, genid)
        if parent in child_counts:
            child_counts[parent] += 1

    # Novelty-weighted selection: probability inversely proportional to
    # (1 + child_count), so under-explored branches are preferentially picked.
    # This prevents mode collapse where a single high-scoring parent
    # dominates the archive's evolutionary tree.
    weights = np.array([1.0 / (1 + child_counts[genid]) for genid in candidates])
    weights /= weights.sum()

    candidates_list = list(candidates.keys())
    chosen = np.random.choice(candidates_list, p=weights)
    return str(chosen)