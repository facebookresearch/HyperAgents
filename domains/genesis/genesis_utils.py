import ast
import json
import math
import os
from collections import defaultdict

import numpy as np

from hydra import compose, initialize_config_dir


def collect_and_summarize_results(output_dir):
    """Collect and summarize results from JSON files in the output directory for Genesis domain.

    Args:
        output_dir (str): Directory containing per-episode results in JSON format.

    Returns:
        dict: A summary dictionary containing average fitness, standard errors, and task summaries.
    """
    # Load config
    # NOTE: Get the number of tasks that should have been run instead of what was successfully run
    config_dir = os.path.abspath(output_dir)
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="config")
    env_names = cfg["envs"]["names"].split('-')
    general_to_specific_tasks = {
        k: v for k, v in cfg["tasks"].items()
        if any(k.startswith(env) for env in env_names)
    }
    expected_num_episodes = {
        k: v for k, v in cfg["eval"]["num_episodes"].items()
        if any(k.startswith(env) for env in env_names)
    }
    expected_task_counts = {}
    for general_task, count in expected_num_episodes.items():
        specific_tasks = general_to_specific_tasks[f"{general_task}_tasks"]
        for specific_task in specific_tasks:
            expected_task_counts[specific_task.split('-')[0]] = count

    # Find all genesis_eval/ folders
    genesis_eval_dirs = [
        os.path.join(root, dir)
        for root, dirs, files in os.walk(output_dir)
        for dir in dirs
        if dir.startswith("genesis_eval")
    ]

    # From each genesis_eval/ folder, find all .json files
    json_files = []
    for genesis_eval_dir in genesis_eval_dirs:
        for root, dirs, files in os.walk(genesis_eval_dir):
            for filename in files:
                if filename.endswith(".json"):
                    json_filepath = os.path.join(root, filename)
                    json_files.append(json_filepath)

    # Each json file is a separate run, so we need to group them by tasks
    # Each json file has the name format of {task}_run_{run_id}.json
    task_to_runs = {}
    task_to_episodes_played = {}
    for json_file in json_files:
        task_name = os.path.basename(json_file).split("_run_")[0]
        if task_name not in task_to_runs:
            task_to_runs[task_name] = []
            task_to_episodes_played[task_name] = []
        with open(json_file, "r") as f:
            run_data = json.load(f)
            run_fitness_scores = run_data["fitness_score"]
            run_fitness = sum(run_fitness_scores) / len(
                run_fitness_scores
            )  # average over
            task_to_runs[task_name].append(run_fitness)
            task_to_episodes_played[task_name].append(run_data["total_episodes_played"])

    # Fill in failed runs (i.e., genesis_eval_dirs with no json file) with 0 fitness
    for task_name, expected_count in expected_task_counts.items():
        if task_name not in task_to_runs:
            task_to_runs[task_name] = [0] * expected_count
            task_to_episodes_played[task_name] = [0] * expected_count
        elif len(task_to_runs[task_name]) < expected_count:
                task_to_runs[task_name] += [0] * (expected_count - len(task_to_runs[task_name]))
                task_to_episodes_played[task_name] += [0] * (expected_count - len(task_to_episodes_played[task_name]))

    # Summarize results for each task
    task_info = {}
    for task_name, fitness_scores in task_to_runs.items():
        task_info[task_name] = {
            "average_fitness": np.mean(fitness_scores),
            "standard_error": np.std(fitness_scores),
        }
    for task_name, episodes_played in task_to_episodes_played.items():
        task_info.setdefault(task_name, {})["episodes_played"] = episodes_played

    # Create report
    all_fitness_scores = []
    for task_name, fitness_scores in task_to_runs.items():
        all_fitness_scores.extend(fitness_scores)
    report = {
        "average_fitness": np.mean(all_fitness_scores),
        "standard_error": np.std(all_fitness_scores),
        "environments": task_info,
    }

    # Save report
    report_filename = os.path.join(output_dir, "report.json")
    with open(report_filename, "w") as f:
        json.dump(report, f, indent=4)

    # Print report
    print(json.dumps(report, indent=4))


def file_to_string(filename):
    with open(filename, "r") as file:
        return file.read()


def get_function_signature(code_string):
    # Parse the code string into an AST
    module = ast.parse(code_string)

    # Find the function definitions
    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]

    # If there are no function definitions, return None
    if not function_defs:
        return None

    # For simplicity, we'll just return the signature of the first function definition
    function_def = function_defs[0]

    input_lst = []
    # Construct the function signature (within object class)
    signature = (
        function_def.name
        + "(self."
        + ", self.".join(arg.arg for arg in function_def.args.args)
        + ")"
    )
    for arg in function_def.args.args:
        input_lst.append(arg.arg)
    return signature, input_lst
