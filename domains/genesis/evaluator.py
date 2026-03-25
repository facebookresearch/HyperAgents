import json
import logging
import multiprocessing
import os
import random
import re
import traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm
import hashlib
import time

logger = logging.getLogger(__name__)

def get_unique_seed(process_num=None, episode_idx=0):
    """Generate a unique seed using process number, episode index, and high-resolution time."""
    pid = os.getpid()
    time_ns = time.time_ns()
    unique_str = f"{pid}_{process_num}_{episode_idx}_{time_ns}"
    hashed = hashlib.sha256(unique_str.encode()).hexdigest()
    seed = int(hashed[:8], 16)
    return seed

class EvaluatorManager:
    """Manages evaluation of agents across multiple environments and tasks.

    The EvaluatorManager initializes evaluators for each specified environment and handles the execution
    of evaluation tasks either sequentially or in parallel using multiple workers.
    """

    def __init__(self, config, original_cwd="", output_dir="."):
        """Initialize the EvaluatorManager.

        Args:
            config (omegaconf.DictConfig): Configuration object containing evaluation settings.
            original_cwd (str, optional): Original current working directory. Defaults to "".
            output_dir (str, optional): Directory to save evaluation outputs. Defaults to ".".
        """
        self.config = config
        self.original_cwd = original_cwd
        self.output_dir = output_dir

        self.env_names = config.envs.names.split("-")
        self.env_evaluators = {}
        self.tasks = []
        for env_name in self.env_names:
            evaluator = Evaluator(
                env_name, config, original_cwd=original_cwd, output_dir=self.output_dir
            )
            self.env_evaluators[env_name] = evaluator
            for task in evaluator.tasks:
                for episode_idx in range(evaluator.num_episodes):
                    # Check if task has been completed
                    json_filename = os.path.join(
                        self.output_dir,
                        env_name,
                        task,
                        f"{task}_run_{episode_idx:02d}.json",
                    )
                    if os.path.exists(json_filename):
                        logging.info(
                            f"Skipping completed task: {env_name}, {task}, episode {episode_idx}"
                        )
                    else:
                        self.tasks.append((env_name, task, episode_idx))
        self.num_workers = config.eval.num_workers

    def run(self, agent_factory):
        """Run the evaluation using the specified agent factory.

        Args:
            agent_factory (AgentFactory): Factory object to create agents for evaluation.

        Returns:
            dict: Results of the evaluation aggregated by environment name.
        """
        if self.num_workers > 1:
            results = self._run_parallel(agent_factory)
        else:
            results = self._run_sequential(agent_factory)
        return results

    def _run_sequential(self, agent_factory):
        """Run the evaluation sequentially.

        Args:
            agent_factory (AgentFactory): Factory object to create agents for evaluation.

        Returns:
            dict: Results of the evaluation aggregated by environment name.
        """
        results = defaultdict(list)
        total_episodes = len(self.tasks)
        with tqdm(
            total=total_episodes, desc="Evaluating Episodes", position=0, disable=True
        ) as pbar:
            for env_name, task, episode_idx in self.tasks:

                task_base, variation = task.split("/")
                evaluator = self.env_evaluators[env_name]

                # Create agent
                chat_history_file = os.path.join(
                    self.output_dir,
                    env_name,
                    task,
                    f"{task_base}_run_{episode_idx:02d}_chathistory.md",
                )
                Path(chat_history_file).parent.mkdir(exist_ok=True, parents=True)
                agent = agent_factory.create_agent(chat_history_file=chat_history_file)

                episode_log = evaluator.run_episode(
                    task, agent, position=1, episode_idx=episode_idx
                )
                results[env_name].append(episode_log)
                pbar.update(1)
        return results

    def _run_parallel(self, agent_factory):
        """Run the evaluation in parallel using multiple workers.

        Args:
            agent_factory (AgentFactory): Factory object to create agents for evaluation.

        Returns:
            dict: Results of the evaluation aggregated by environment name.
        """
        task_queue = multiprocessing.Queue()
        results_queue = multiprocessing.Queue()

        ctx = multiprocessing.get_context("fork")

        # Initially fill the task queue with tasks up to the number of workers
        for item in self.tasks[: self.num_workers]:
            task_queue.put(item)

        # Create a master progress bar
        pbar = tqdm(total=len(self.tasks), position=0, leave=True, disable=True)

        # Assign unique positions for progress bars
        positions = list(range(self.num_workers))

        processes = []
        for idx in range(self.num_workers):
            position = positions[idx]
            p = ctx.Process(
                target=self._worker,
                args=(task_queue, results_queue, agent_factory, position),
            )
            processes.append(p)
            p.start()

        results = defaultdict(list)
        tasks_completed = 0
        tasks_queued = self.num_workers

        total_tasks = len(self.tasks)

        while tasks_completed < total_tasks:
            result = results_queue.get()
            if "error" in result:
                logging.error(
                    f"Error in task {result['task']} processed by {result['process_num']}: {result['error']}"
                )
                logging.error(f"Traceback:\n{result['traceback']}")
            else:
                results[result["env_name"]].append(result)
            tasks_completed += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_description(
                f"Last task: {result['task']}, Process: {result.get('process_num', 'N/A')}"
            )

            # Queue another task if there are any left
            if tasks_queued < total_tasks:
                task_queue.put(self.tasks[tasks_queued])
                tasks_queued += 1

        # Signal workers to stop
        for _ in range(self.num_workers):
            task_queue.put(None)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Close the master bar when done
        pbar.close()

        return results

    def _worker(self, task_queue, results_queue, agent_factory, position):
        """Worker process for parallel evaluation.

        Args:
            task_queue (multiprocessing.Queue): Queue containing tasks to process.
            results_queue (multiprocessing.Queue): Queue to put the results.
            agent_factory (AgentFactory): Factory object to create agents.
            position (int): Position index for the progress bar.
        """
        seed = get_unique_seed(process_num=position)
        random.seed(seed)
        np.random.seed(seed)

        process_num = multiprocessing.current_process().name
        while True:
            item = task_queue.get()
            if item is None:
                break
            try:
                env_name, task, episode_idx = item
                evaluator = self.env_evaluators[env_name]

                # Create a fresh agent per episode so files don't collide across parallel runs
                chat_history_file = os.path.join(
                    self.output_dir,
                    env_name,
                    task,
                    f"{task}_run_{episode_idx:02d}_chathistory.md",
                )
                Path(chat_history_file).parent.mkdir(exist_ok=True, parents=True)
                agent = agent_factory.create_agent(chat_history_file=chat_history_file)

                result = evaluator.run_episode(
                    task,
                    agent,
                    process_num=process_num,
                    position=position + 1,
                    episode_idx=episode_idx,
                )
                result["process_num"] = process_num  # Include process number in result
                result["env_name"] = env_name
                results_queue.put(result)
            except Exception as e:
                tb = traceback.format_exc()
                logging.error(
                    f"Error in worker processing task {task}: {e}\n{tb}"
                )  # pyright: ignore
                results_queue.put(
                    {
                        "env_name": env_name,  # pyright: ignore
                        "task": task,
                        "error": str(e),
                        "traceback": tb,
                        "process_num": process_num,
                    }
                )


class Evaluator:
    """Evaluator for a single environment and task.

    The Evaluator handles the execution of evaluation episodes for a specific environment and task,
    including loading in-context learning episodes and running episodes with the agent.
    """

    def __init__(self, env_name, config, original_cwd="", output_dir="."):
        """Initialize the Evaluator.

        Args:
            env_name (str): Name of the environment to evaluate.
            config (omegaconf.DictConfig): Configuration object containing evaluation settings.
            original_cwd (str, optional): Original current working directory. Defaults to "".
            output_dir (str, optional): Directory to save evaluation outputs. Defaults to ".".
        """
        self.env_name = env_name.strip()
        self.config = config
        self.output_dir = output_dir
        self.tasks = config.tasks[f"{self.env_name}_tasks"]

        self.num_episodes = config.eval.num_episodes[self.env_name]
        self.num_workers = config.eval.num_workers
        self.max_steps_per_episode = config.eval.max_steps_per_episode

        # Genesis evaluator doesn't use in-context datasets like Balrog
        # TODO: Implement any genesis-specific dataset loading if needed
        self.dataset = None

    def run_episode(self, task, agent, process_num=None, position=0, episode_idx=0):
        """Run a single evaluation episode.

        Args:
            task (str): Task name.
            agent (Agent): Agent to evaluate.
            process_num (str, optional): Identifier of the process running the episode. Defaults to None.
            position (int, optional): Position index for the progress bar. Defaults to 0.
            episode_idx (int, optional): Index of the episode. Defaults to 0.

        Returns:
            dict: Log of the episode containing statistics and results.
        """
        ### Phase 1: Initialization

        # 1.1 seed
        # TODO: incorporate seed for RL training and eval
        seed = self.config.envs.env_kwargs.seed
        if seed is None:
            seed = get_unique_seed(process_num=process_num, episode_idx=episode_idx)
        random.seed(seed)
        np.random.seed(seed)

        # 1.2 Prepare Inputs to the Task Agent
        from domains.genesis.genesis_utils import file_to_string

        root_dir = self.config.utils.root_dir
        task_description = self.config.task_description.get(task.split("/")[0], "")
        task_name = task.split("-")[0]
        genesis_environment_path = (
            f"{root_dir}/domains/genesis/environments/{task_name}.py"
        )
        default_rewfn_path = f"{root_dir}/domains/genesis/reward/default_function.py"
        default_rewfn_string = file_to_string(default_rewfn_path)

        inputs = {
            # "domain": f"genesis_{self.env_name}",
            "domain": "genesis_go2walking",  # NOTE: the agent should use the same behavior for go2 tasks
            "task_description": task_description,
            "genesis_environment_path": genesis_environment_path,
            "default_reward_function": default_rewfn_string,
        }

        #  Phase 2: Task Agent Generation
        # get task agent response
        response, _ = agent.forward(inputs)
        # for non-initial runs, extract the reward function from the LLM response
        code_str = self.extract_code_str(response)
        # code_str = default_rewfn_string  # NOTE: uncomment this to use default reward function
        # save the reward function to a file
        rwd_func_path = os.path.join(
            self.output_dir,
            f"reward_function_{episode_idx:02d}.py",
        )
        self.save_reward_function(code_str, rwd_func_path)

        #  Phase 3: RL Training and Evaluation
        # 3.1 Launch RL training
        train_log = self.run_rl_train(
            task, episode_idx, seed,
            rwd_func_path=rwd_func_path,
        )
        # 3.2 Launch RL evaluation
        if train_log["training_success"]:
            eval_log = self.run_rl_eval(
                task, episode_idx, seed,
                rwd_func_path=rwd_func_path,
            )
            episode_log = {**train_log, **eval_log}
        else:
            episode_log = {**train_log, "fitness": 0}

        return episode_log

    def run_rl_train(self, task, episode_idx, seed, rwd_func_path=""):
        import subprocess
        import sys

        env_kwargs = self.config.envs.get(f"{self.env_name}_kwargs", {})

        rl_dir = os.path.join(
            self.output_dir,
            self.env_name,
            task,
        )
        episode_log = {
            "task": task,
            "seed": seed,
        }
        exp_name = f"{task}_episode_{episode_idx:02d}"

        training_cmd = [
            sys.executable,
            "-m",
            "domains.genesis.genesis_train.rl_trainer",
            "-e",
            exp_name,
            "--num_envs",
            str(self.config.envs.get("num_envs", 4096)),  # Default to 4096 envs
            "--max_iterations",
            str(
                self.config.rl_trainer.get("max_iterations", 101)
            ),  # Default to 101 iterations
            "--output_dir",
            rl_dir,
            "--rwd_func_path",
            rwd_func_path,
            "--episode_idx",
            str(episode_idx).zfill(2),
        ]

        # Add environment-specific arguments
        if self.env_name in ["go2walking", "go2walkback"]:
            lin_vel_x_range = env_kwargs.get("lin_vel_x_range", [0.5, 0.5])
            training_cmd.extend([
                "--lin_vel_x_range",
                str(lin_vel_x_range[0]),
                str(lin_vel_x_range[1]),
            ])

        try:
            logger.info(f"Starting RL training with command: {' '.join(training_cmd)}")
            # result is a CompletedProcess object with:
            # - result.returncode (0 for success, non-zero for failure)
            # - result.stdout (captured output)
            # - result.stderr (captured errors)
            # - result.args (the command that was run)
            result = subprocess.run(
                training_cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=self.config.utils.root_dir,
            )

            if result.returncode == 0:
                training_logs_dir = f"{rl_dir}/genesis_train_{episode_idx:02d}"
                logger.info(f"RL training completed successfully for {exp_name}")
                training_output = result.stdout
                # log all printed output:
                logger.info(f"Training output:\n{result.stdout}")
            else:
                logger.error(
                    f"RL training failed for {exp_name}: {result.stderr}\n\n{result.stdout}"
                )
                training_output = result.stderr
                training_logs_dir = None

            episode_log["training_output"] = training_output
            episode_log["training_logs_dir"] = training_logs_dir
            episode_log["training_success"] = result.returncode == 0

        except subprocess.TimeoutExpired:
            logger.error(f"RL training timed out for {exp_name}")
            episode_log["training_output"] = "Training timed out"
            episode_log["training_logs_dir"] = None
            episode_log["training_success"] = False
        except Exception as e:
            logger.error(f"Failed to launch RL training for {exp_name}: {str(e)}")
            episode_log["training_output"] = f"Failed to launch: {str(e)}"
            episode_log["training_logs_dir"] = None
            episode_log["training_success"] = False

        # Save the episode_log to a JSON file
        task_base = task.split("/")[0]
        json_filename = os.path.join(
            self.output_dir,
            self.env_name,
            task,
            f"genesis_train_{episode_idx:02d}",
            f"{task_base}_run_{episode_idx:02d}.json",
        )
        Path(json_filename).parent.mkdir(exist_ok=True, parents=True)
        with open(json_filename, "w") as f:
            json.dump(episode_log, f, indent=4)

        return episode_log

    def run_rl_eval(self, task, episode_idx, seed, rwd_func_path=""):
        import subprocess
        import sys

        rl_dir = os.path.join(
            self.output_dir,
            self.env_name,
            task,
        )
        episode_log = {
            "task": task,
            "mission": self.config.task_description.get(task, ""),
            "seed": seed,
        }
        exp_name = f"{task}_episode_{episode_idx:02d}"
        env_kwargs = self.config.envs.get(f"{self.env_name}_kwargs", {})

        eval_cmd = [
            sys.executable,
            "-m",
            "domains.genesis.genesis_eval.rl_eval",
            "-e",
            exp_name,
            "--output_dir",
            rl_dir,
            "--ckpt",
            str(self.config.rl_trainer.get("max_iterations", 101) - 1),
            "--num_envs",
            str(self.config.envs.get("num_envs")),
            "--max_steps",
            str(self.config.rl_eval.max_steps),
            "--rwd_func_path",
            rwd_func_path,
            "--record_video" if self.config.rl_eval.record_video else "--no-record_video",
            "--episode_idx",
            str(episode_idx).zfill(2),
        ]

        # Add environment-specific arguments
        if self.env_name in ["go2walking", "go2walkback"]:
            lin_vel_x_range = env_kwargs.get("lin_vel_x_range", [0.5, 0.5])
            eval_cmd.extend([
                "--lin_vel_x_range",
                str(lin_vel_x_range[0]),
                str(lin_vel_x_range[1]),
            ])

        try:
            logger.info(f"Starting RL eval with command: {' '.join(eval_cmd)}")

            result = subprocess.run(
                eval_cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=self.config.utils.root_dir,
            )

            if result.returncode == 0:
                logger.info(f"RL eval completed successfully for {exp_name}")
                eval_logs_dir = f"{rl_dir}/genesis_eval_{episode_idx:02d}"
                # log all printed output:
                logger.info(f"Eval output:\n{result.stdout}")
            else:
                logger.error(
                    f"RL eval failed for {exp_name}: {result.stderr}\n\n{result.stdout}"
                )
                eval_logs_dir = None
            episode_log["eval_output"] = "Eval succeeded"
            episode_log["eval_logs_dir"] = eval_logs_dir
            episode_log["eval_success"] = result.returncode == 0

        except subprocess.TimeoutExpired:
            logger.error(f"RL eval timed out for {exp_name}")
            episode_log["eval_output"] = "Eval timed out"
            episode_log["eval_logs_dir"] = None
            episode_log["eval_success"] = False
        except Exception as e:
            logger.error(f"Failed to launch RL eval for {exp_name}: {str(e)}")
            episode_log["eval_output"] = f"Failed to launch: {str(e)}"
            episode_log["eval_logs_dir"] = None
            episode_log["eval_success"] = False

        # Save the episode_log to eval.log file
        log_filename = os.path.join(
            self.output_dir,
            self.env_name,
            task,
            f"genesis_eval_{episode_idx:02d}",
            "genesis_eval.log",
        )
        Path(log_filename).parent.mkdir(exist_ok=True, parents=True)
        with open(log_filename, "w") as f:
            for key, value in episode_log.items():
                f.write(f"{key}: {value}\n")

        return episode_log

    def extract_reward_function(self, stripped) -> None:
            # Look for code starting with 'from' or 'import' or 'def' and ending with 'return'
            if "def compute_reward" in stripped:
                # Find the start of the Python code
                start_patterns = [
                    r"(from )",
                    r"(import )",
                    r"(def compute_reward)",
                ]
                start_pos = -1

                for pattern in start_patterns:
                    match = re.search(pattern, stripped)
                    if match:
                        start_pos = match.start()
                        break

                if start_pos == -1:
                    raise ValueError("Could not find start of Python code")

                # Extract from start position to the last 'return' statement
                code_portion = stripped[start_pos:]

                # Find the last return statement and include everything up to it
                # Look for 'return total_reward, reward_components, reward_scales'
                return_match = re.search(
                    r"return\s+total_reward\s*,\s*reward_components\s*,\s*reward_scales",
                    code_portion,
                )
                if return_match:
                    end_pos = return_match.end()
                    return code_portion[:end_pos].strip()
                else:
                    raise ValueError("Could not find valid return statement in Python code")

            else:
                raise ValueError("Could not find compute_reward function in Python code")
    
    def extract_code_str(self, update_reward_str: str) -> str:
        """Extract Python code from various response formats.

        Args:
        Returns:
            The extracted Python code string

        Raises:
            ValueError: If no valid Python code is found
        """
        try:
            stripped = update_reward_str.strip()

            # Handle escaped newlines by replacing them with actual newlines
            if "\\n" in stripped:
                stripped = stripped.replace("\\n", "\n")

            # Try to extract reward function
            code_str = self.extract_reward_function(stripped)
            return code_str
        except Exception as e:
            logger.warning(f"Failed to extract reward function: {e}")
            # Try to return the default reward function content
            try:
                from domains.genesis.genesis_utils import file_to_string
                root_dir = self.config.utils.root_dir
                default_rewfn_path = (
                    f"{root_dir}/domains/genesis"
                    "/reward/default_function.py"
                )
                return file_to_string(default_rewfn_path)
            except Exception:
                return ""
        
        # # Try to extract reward function from markdown code block 
        # try:
        #     code_match = re.search(r"```python\s*(.*?)\s*```", stripped, re.DOTALL)
        #     if code_match:
        #         self.extract_reward_function(code_match.group(1))
        # except:
        #     raise ValueError("Failed to extract reward function from markdown code block")

    def save_reward_function(self, code_str: str, rwd_func_path: str) -> None:
        """Save the reward function code to a Python file.
            Always overwrite if the rwd_func_path already exists.

        Args:
            code_str: The Python code string to save
            rwd_func_path: The file path where the reward function should be saved
        """
        from pathlib import Path

        # Ensure parent directory exists
        Path(rwd_func_path).parent.mkdir(parents=True, exist_ok=True)

        # Write the code to file
        with open(rwd_func_path, "w") as f:
            f.write(code_str)
