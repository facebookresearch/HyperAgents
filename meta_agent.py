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
            iterations_left (int, optional): The number of remaining iterations in which the meta agent will be invoked in future. Defaults to None.
        """
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
        ]
        if eval_path:
            instruction_parts.append(
                f"Previously generated agents and their evaluation "
                f"results are stored at `{eval_path}`. Read them "
                f"before proposing modifications so you can learn "
                f"from what has already been tried and avoid "
                f"repeating prior failures."
            )
        if iterations_left is not None:
            instruction_parts.append(
                f"You have {iterations_left} iteration(s) remaining "
                f"after this one. Budget exploration vs. "
                f"exploitation accordingly."
            )
        instruction = "\n\n".join(instruction_parts)

        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available='all',
        )
