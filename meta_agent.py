# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent


def _read_eval_summaries(eval_path):
    """Read evaluation result files from eval_path and return a summary string.

    Looks for common result formats (JSON, markdown) in the eval directory
    and concatenates key findings into a prompt-friendly summary.
    """
    if not eval_path or not os.path.isdir(eval_path):
        return None

    summaries = []

    # Collect JSON result files
    for root, _, files in os.walk(eval_path):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            if fname.endswith(".json"):
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    # Extract the most useful fields — scores, pass rates, metrics
                    score_fields = {
                        k: v
                        for k, v in data.items()
                        if isinstance(v, (int, float, str))
                        and any(
                            kw in k.lower()
                            for kw in ("score", "pass", "reward", "success", "acc", "metric", "result")
                        )
                    }
                    if score_fields:
                        summaries.append(
                            f"  {os.path.relpath(fpath, eval_path)}: {json.dumps(score_fields)}"
                        )
                except (json.JSONDecodeError, OSError):
                    continue
            elif fname.endswith(".md"):
                try:
                    with open(fpath) as f:
                        content = f.read()
                    # Grab the first few lines — enough context without bloating the prompt
                    preview = "\n".join(content.splitlines()[:10]).strip()
                    if preview:
                        summaries.append(
                            f"  {os.path.relpath(fpath, eval_path)}:\n    {preview}"
                        )
                except OSError:
                    continue

    if not summaries:
        return None

    header = f"Previous evaluation results from {eval_path}:"
    return header + "\n" + "\n".join(summaries)


class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None):
        """
        A meta agent that recursively self-improves.

        Args:
            repo_path (str): The path to the repository.
            eval_path (str): The path to previously generated agents and their evaluation results.
            iterations_left (int, optional): The number of remaining iterations in which the meta agent will be invoked in future. Defaults to None.
        """
        instruction = f"Modify any part of the codebase at `{repo_path}`."

        # Surface prior evaluation results so the agent can learn from them
        eval_summary = _read_eval_summaries(eval_path)
        if eval_summary:
            instruction += f"\n\n{eval_summary}"

        # Inform the agent of its iteration budget so it can trade off
        # exploration vs. exploitation across recursive calls
        if iterations_left is not None:
            instruction += (
                f"\n\nYou have {iterations_left} iteration(s) remaining."
                " Prioritize high-impact improvements that address the"
                " weakest areas identified in the evaluation results above."
            )

        new_msg_history = chat_with_agent(instruction, model=self.model, msg_history=[], logging=self.log, tools_available='all')