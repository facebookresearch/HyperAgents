# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from collections import defaultdict

import pandas as pd

from utils.domain_utils import can_domain_ensembled
from utils.gl_utils import load_archive_data, get_score

# Classification domains that support weighted majority voting
_CLASSIFICATION_DOMAINS = {"search_arena", "paper_review", "imo_grading"}


def _get_top_agents(
    domain, generate_output_dir,
    archive_genids, split, top_k=3
):
    """Return up to top_k (genid, score) pairs sorted by score descending."""
    scored = []
    for genid in archive_genids:
        score = get_score(domain, generate_output_dir, genid, split=split)
        if score is not None:
            scored.append((genid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _get_prediction_for_agent(
    domain, generate_output_dir,
    genid, question_id, split
):
    """Load a single agent's prediction for a given question_id."""
    if split == "train":
        pred_dirname = f"{domain}_eval"
    else:
        pred_dirname = f"{domain}_eval_{split}"
    predictions_path = os.path.join(
        generate_output_dir, f"gen_{genid}/{pred_dirname}/predictions.csv"
    )
    try:
        df = pd.read_csv(predictions_path)
        match = df.loc[df["question_id"] == question_id, "prediction"]
        if match.empty:
            return None
        return match.iloc[0]
    except Exception:
        return None


def ensemble(domain, task, generate_output_dir, split="train"):
    """
    Run ensemble on a single task.

    For classification domains (search_arena, paper_review, imo_grading),
    uses weighted majority voting across the top-3 agents by score.
    For other domains, returns the single best agent's prediction.

    Args:
        domain (str): The domain of the task.
        task (dict): A task dictionary, with keys "question_id", and necessary input keys for the domain.
        generate_output_dir (str): The directory where the generated archive is stored.
        split (str): The data split to use.

    Returns:
        str: The prediction of the ensemble.
    """
    question_id = task["question_id"]

    # Load archive
    archive_path = os.path.join(generate_output_dir, "archive.jsonl")
    archive_data = load_archive_data(archive_path, last_only=True)
    archive_genids = archive_data.get("archive", [])

    # Get top agents by score
    top_agents = _get_top_agents(
        domain, generate_output_dir,
        archive_genids, split
    )

    if not top_agents:
        return None

    # Weighted majority voting for classification domains with 3+ agents
    is_classification = (
        domain in _CLASSIFICATION_DOMAINS
        and can_domain_ensembled(domain)
        and len(top_agents) >= 3
    )
    if is_classification:
        votes = defaultdict(float)
        for genid, score in top_agents[:3]:
            pred = _get_prediction_for_agent(
                domain, generate_output_dir, genid, question_id, split
            )
            if pred is not None:
                votes[pred] += score

        if votes:
            # Pick the prediction with the highest weighted vote
            return max(votes, key=votes.get)

    # Fallback: single best agent prediction
    best_genid, _ = top_agents[0]
    return _get_prediction_for_agent(
        domain, generate_output_dir, best_genid, question_id, split
    )
