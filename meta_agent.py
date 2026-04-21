# Copyright (c) Meta Platforms, Inc. and affiliates.

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent

class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None):
        """
        A meta agent that recursively self-improves.

        Args:
            repo_path (str): The path to the repository.
            eval_path (str): The path to previously generated agents and their evaluation results.
            iterations_left (int, optional): Number of
                remaining meta-agent iterations.
                Defaults to None.
        """
        instruction = (
            f"Modify any part of the codebase"
            f" at `{repo_path}`."
        )
        instruction += (
            f"\n\nStart by reading"
            f" `{repo_path}/README.md`"
            f" for orientation on the system"
            f" and file structure."
        )
        instruction += (
            f"\n\nPrevious generations and their"
            f" evaluation results are at"
            f" `{eval_path}`. Analyze these to"
            f" understand current performance."
        )
        if iterations_left is not None:
            instruction += (
                f"\n\nYou have {iterations_left}"
                f" iteration(s) remaining."
                f" Budget your changes accordingly."
            )
        instruction += (
            "\n\nSuggested approach:"
            " 1) Read evaluation results to"
            " identify bottlenecks,"
            " 2) Analyze the relevant code,"
            " 3) Implement targeted"
            " improvements."
        )

        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available='all',
        )
