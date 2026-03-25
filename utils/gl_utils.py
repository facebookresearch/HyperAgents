"""
Utils for generate_loop.py
"""

import fnmatch
import json
import math
import os
import random
import re
import shutil

import numpy as np

from utils.common import read_file
from utils.constants import REPO_NAME
from utils.docker_utils import copy_to_container, log_container_output
from utils.domain_utils import (
    can_domain_ensembled,
    get_domain_score_key,
    get_domain_splits,
    get_domain_stagedeval_frac,
)
from utils.git_utils import commit_repo, get_git_commit_hash


def is_starting_node(genid):
    # Starting nodes are initial or 0
    return genid == "initial" or genid == 0


def get_saved_score(domain, output_dir, genid, split="train", type="agent"):
    # Type can be "agent" or "ensemble" or "max"
    agent_score = get_score(domain, output_dir, genid, split=split)
    ensemble_score = get_saved_ensemble_score(domain, output_dir, genid, split=split)

    # If full eval is not run, adjust score
    run_full_eval = get_node_metadata_key(output_dir, genid, "run_full_eval")
    if genid == "initial" or (run_full_eval is None or not run_full_eval):
        stagedeval_frac = get_domain_stagedeval_frac(domain)
        agent_score = agent_score * stagedeval_frac if agent_score is not None else None
        ensemble_score = (
            ensemble_score * stagedeval_frac if ensemble_score is not None else None
        )

    # Get score based on type
    if type == "agent":
        score = agent_score
    elif type == "ensemble":
        score = ensemble_score
    elif type == "max":
        if agent_score is not None and ensemble_score is not None:
            score = max(agent_score, ensemble_score)
        elif agent_score is not None:
            score = agent_score
        elif ensemble_score is not None:
            score = ensemble_score
        else:
            score = None
    else:
        raise ValueError(f"Unknown type '{type}'")
    return score


def get_score(domain, output_dir, genid, split="train"):
    # Get score from eval file
    eval_dirname = f"{domain}_eval" if split == "train" else f"{domain}_eval_{split}"
    eval_file = os.path.join(output_dir, f"gen_{genid}/{eval_dirname}/report.json")
    score_key = get_domain_score_key(domain)
    try:
        with open(eval_file, "r") as f:
            eval_results = json.load(f)
        score = eval_results[score_key]
        # Filter nan values
        if math.isnan(score):
            score = None

        if "balrog" in domain:
            # Check if evals on environments ran
            if len(eval_results["environments"]) <= 0:
                return None
            # Normalize score
            return score / 100.0

        if "polyglot" in domain:
            # Check if any task was evaled without error
            noerror_tasks = eval_results["total_unresolved_ids"] + eval_results["total_emptypatch_ids"] + eval_results["total_resolved_ids"]
            if len(noerror_tasks) <= 0:
                return None

        return score
    except Exception:
        return None  # If score is missing or file not found


def get_saved_ensemble_score(domain, output_dir, genid, split="train"):
    # Get score from eval file for ensemble
    eval_file = os.path.join(
        output_dir, f"gen_{genid}/report_ensemble_{domain}_{split}.json"
    )
    score_key = get_domain_score_key(domain)
    try:
        with open(eval_file, "r") as f:
            eval_results = json.load(f)
        return eval_results[score_key]
    except Exception:
        return None  # If score is missing or file not found


def get_parent_genid(output_dir, genid):
    folder_prefix = "gen"
    metadata_file = os.path.join(output_dir, f"{folder_prefix}_{genid}/metadata.json")
    if not os.path.exists(metadata_file):
        return None
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata.get("parent_genid", None)


def get_patch_files(output_dir, genid):
    # Get all patch files, prev and curr
    folder_prefix = "gen"
    metadata_file = os.path.join(output_dir, f"{folder_prefix}_{genid}/metadata.json")
    if not os.path.exists(metadata_file):
        return []
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    patch_files = metadata.get("prev_patch_files", []) + metadata.get(
        "curr_patch_files", []
    )
    return patch_files


def update_node_metadata(output_dir, genid, data_update):
    # Load metadata from genid
    folder_prefix = "gen"
    metadata_file = os.path.join(output_dir, f"{folder_prefix}_{genid}/metadata.json")
    if not os.path.exists(metadata_file):
        return
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    # Update metadata
    metadata.update(data_update)
    # Save metadata atomically
    tmp_file = metadata_file + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(metadata, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, metadata_file)


def get_node_metadata_key(output_dir, genid, key):
    # Load metadata from genid
    folder_prefix = "gen"
    metadata_file = os.path.join(output_dir, f"{folder_prefix}_{genid}/metadata.json")
    if not os.path.exists(metadata_file):
        return None
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata.get(key, None)


def update_and_save_archive(output_dir, archive, new_node):
    # Update archive
    archive.append(new_node)
    # Save archive
    archive_file = os.path.join(output_dir, "archive.jsonl")
    with open(archive_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "current_genid": new_node,
                    "archive": archive,
                },
            )
            + "\n"
        )
    # Return updated archive
    return archive


def load_archive_data(filepath, last_only=True):
    # Load all archives from given metadata file
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found at {filepath}")
    # Read all JSON entries from the JSONL file (one JSON object per line)
    archive_data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    archive_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # Return the last entry
    if last_only:
        return archive_data[-1]
    # Return all entries
    return archive_data


def get_archive_len(output_dir):
    # Get number of generations from archive
    archive_file = os.path.join(output_dir, "archive.jsonl")
    if not os.path.exists(archive_file):
        return 0
    archive_data = load_archive_data(archive_file, last_only=True)
    return len(archive_data.get("archive", []))


def setup_initial_gen(
    output_dir,
    domains,
    copy_root_dir=None,
    subsets=[],
    resume=False,
    copy_eval=True,
    optimize_option="only_agent",
    run_baseline=None,
    eval_test=False,
    edit_select_parent=False,
):
    # Resume from previous run
    if resume:
        root_dir = os.path.abspath(os.path.join(output_dir, f"gen_initial/{REPO_NAME}"))
        commit_hash = get_git_commit_hash(root_dir)
        return root_dir, commit_hash

    # Make a copy of the eval folder
    if copy_eval:
        for domain, subset in zip(domains, subsets):
            splits = get_domain_splits(domain, eval_test=eval_test)
            # Eval on training set
            gen_output_dir = os.path.join(output_dir, f"gen_initial/{domain}_eval")
            shutil.copytree(
                f"./outputs/initial_{domain}{subset}_0/",
                gen_output_dir,
                dirs_exist_ok=True,
            )
            # Eval on validation set
            if "val" in splits:  # pyright: ignore
                gen_output_dir = os.path.join(
                    output_dir, f"gen_initial/{domain}_eval_val"
                )
                shutil.copytree(
                    f'./outputs/initial_{domain}{subset.replace("_train", "_val")}_0/',
                    gen_output_dir,
                    dirs_exist_ok=True,
                )
            # Eval on test set
            if "test" in splits:  # pyright: ignore
                gen_output_dir = os.path.join(
                    output_dir, f"gen_initial/{domain}_eval_test"
                )
                shutil.copytree(
                    f'./outputs/initial_{domain}{subset.replace("_train", "_test")}_0/',
                    gen_output_dir,
                    dirs_exist_ok=True,
                )

    # Make a copy of the repo
    root_dir = os.path.abspath(os.path.join(output_dir, f"gen_initial/{REPO_NAME}"))
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir, exist_ok=True)

    # Define exclusion criteria
    excluded_dirs = {
        ".claude",
        "outputs",
        "analysis",
        "misc",
        "baselines",
        "domains",
    }
    excluded_files = {
        "Dockerfile",
        ".dockerignore",
        "setup_initial.sh",
        "LICENSE.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
    }
    if "polyglot" not in domains:
        excluded_files.add("run_task_agent.py")
    excluded_patterns = ["venv*", "__pycache__*", "*.png", "outputs_os*"]
    if "ensemble" not in optimize_option or not any(can_domain_ensembled(d) for d in domains) and not copy_root_dir:
        excluded_patterns.append("*ensemble*")
    if not edit_select_parent and not copy_root_dir:
        excluded_patterns.append("*select_next_parent*")

    # Define exclusion criteria for domains
    excluded_dirs_domains = {
        "search_arena/saved",
        "polyglot/polyglot_benchmark_metadata.json",
        "polyglot/polyglot-benchmark",
        "polyglot/predictions",
        "polyglot/SWE-bench",
        "polyglot/logs",
    }
    excluded_dirs_domains.update({  # exclude domains that are not in the current run
        f"{folder_name}"
        for folder_name in os.listdir("./domains")
        if os.path.isdir(os.path.join("./domains", folder_name)) and not any(folder_name in d for d in domains)
    })
    excluded_files_domains = {}
    excluded_patterns_domains = ["venv*", "__pycache__*", "*.png"]

    # Function to ignore specific files/directories
    source_root = os.path.abspath(copy_root_dir or "./")
    def ignore_function(src, names):
        ignored = []
        for name in names:
            full_path = os.path.join(src, name)
            rel = os.path.relpath(full_path, start=source_root).replace(os.sep, "/")
            if any(rel == d or rel.startswith(d.rstrip("/") + "/") for d in excluded_dirs):
                ignored.append(name)
            elif name in excluded_files and os.path.isfile(full_path):
                ignored.append(name)
            else:
                for pattern in excluded_patterns:
                    if fnmatch.fnmatch(name, pattern):
                        ignored.append(name)
                        break
        return ignored
    source_root_domains = "./domains"
    def ignore_function_domains(src, names):
        ignored = []
        for name in names:
            full_path = os.path.join(src, name)
            rel = os.path.relpath(full_path, start=source_root_domains).replace(os.sep, "/")
            if any(rel == d or rel.startswith(d.rstrip("/") + "/") for d in excluded_dirs_domains):
                ignored.append(name)
            elif name in excluded_files_domains and os.path.isfile(full_path):
                ignored.append(name)
            else:
                for pattern in excluded_patterns_domains:
                    if fnmatch.fnmatch(name, pattern):
                        ignored.append(name)
                        break
        return ignored

    # Copy the repo
    if copy_root_dir is not None:
        shutil.copytree(copy_root_dir, root_dir, dirs_exist_ok=True, ignore=ignore_function)
        shutil.copytree("./domains", os.path.join(root_dir, "domains"), dirs_exist_ok=True, ignore=ignore_function_domains)
    else:
        shutil.copytree("./", root_dir, dirs_exist_ok=True, ignore=ignore_function)
        shutil.copytree("./domains", os.path.join(root_dir, "domains"), dirs_exist_ok=True, ignore=ignore_function_domains)

    # Setup README.md
    readme_path = os.path.join(root_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        readme_desc = get_readme_description(
            ensemble="ensemble" in optimize_option,
            edit_select_parent=edit_select_parent,
            domains=domains,
            generation=0,
            eval_path=output_dir,
        )
        f.write(readme_desc)

    # Setup files for dgm baseline
    if copy_root_dir is None and run_baseline and "dgm" in run_baseline:
        shutil.copyfile("./baselines/dgm/coding_agent.py", os.path.join(root_dir, "coding_agent.py"))
        os.remove(os.path.join(root_dir, "meta_agent.py"))
        os.remove(os.path.join(root_dir, "run_meta_agent.py"))

    # Get commit hash
    commit_hash = commit_repo(root_dir)

    # Return info
    return root_dir, commit_hash


def get_readme_description(
    ensemble=False,
    edit_select_parent=False,
    domains=None,
    generation=None,
    eval_path=None,
):
    desc = """# Self-Improving AI

This system is designed to automatically produce agents for solving downstream tasks. The system iteratively improves the generated agents through code editing. To enable continuous improvement, the system should look at its code repository and the provided path to previously generated agents and their evaluation results, and then edit and enhance its own mechanisms for generating agents. This process creates a recursive loop of self-improvement.
"""

    # Add domain and evaluation info
    if domains:
        domain_metrics = {
            "search_arena": "overall_accuracy",
            "paper_review": "overall_accuracy",
            "imo_grading": "overall_accuracy",
            "imo_proof": "points_percentage",
        }
        desc += "\n## Target Domains\n\n"
        for d in domains:
            if d in domain_metrics:
                metric = domain_metrics[d]
            elif "balrog" in d:
                metric = "average_progress"
            elif "genesis" in d:
                metric = "average_fitness"
            elif "polyglot" in d:
                metric = "accuracy_score"
            else:
                metric = "score"
            desc += f"- **{d}** (metric: `{metric}`)\n"

    if generation is not None:
        desc += (
            f"\n## Current State\n\n"
            f"This is generation **{generation}**"
            f" (initial setup).\n"
        )

    if eval_path:
        desc += (
            f"\nEvaluation results from previous"
            f" generations are stored at"
            f" `{eval_path}`. Each generation"
            f" folder (e.g., `gen_0/`, `gen_1/`)"
            f" contains a"
            f" `<domain>_eval/report.json`"
            f" with scores.\n"
        )

    desc += """
## Key Files

- `meta_agent.py` — The meta agent that drives
  self-improvement. It can modify any file.
- `task_agent.py` — The task agent that solves downstream domain tasks.
- `agent/` — LLM interface, tool definitions, and base agent classes.
- `domains/` — Domain-specific evaluation and task code.
"""

    if ensemble:
        desc += """\n## Optimize the Ensemble of Agents

Given a fixed archive of agents, optimize the performance of the ensemble without modifying the individual agents. The archive of agents is provided in `/tmp/agent_archive/`. You can edit and improve the ensemble logic in `ensemble.py`."""

    if edit_select_parent:
        desc += """\n## Parent Selection Mechanism

The parent selection mechanism should sample agents from the archive using a non-greedy, diversity-preserving strategy that allows weaker, novel, or niche agents to produce offspring. This enables the discovery of interesting stepping stones that can unlock larger future improvements. The goal is to maximize long-term innovation and avoid premature convergence, while still filtering out uninteresting or unproductive exploration paths to use compute efficiently. You can edit and improve the select parent logic in `select_next_parent.py`.

Note that:
- A node with no children does not mean that its path has not been explored. The node's lineage depth indicates the depth of exploration."""

    return desc

def filter_patch_by_files(patch_str, target_files):
    """
    Filters out the diff blocks related to any of the target_files in a patch string.

    Args:
        patch_str (str): The complete patch text.
        target_files (list[str]): A list of filenames for which to extract changes (e.g. ['affine_cipher.py', 'other.py']).

    Returns:
        str: A string containing only the diff blocks for the specified target files.
    """
    lines = patch_str.splitlines()
    filtered_lines = []
    include_block = False

    for line in lines:
        # When we encounter a new diff block header, check if the block is for any of the target files.
        if line.startswith("diff --git"):
            include_block = not any(f"a/{target}" in line and f"b/{target}" in line for target in target_files)
        if include_block:
            filtered_lines.append(line)
    return "\n".join(filtered_lines) + "\n"

def process_meta_patch_files(meta_patch_files, output_dir, reset_task_agent=False, reset_meta_agent=False):
    new_meta_patch_files = []
    meta_patch_dir = os.path.join(output_dir, "meta_patch_files")
    os.makedirs(meta_patch_dir, exist_ok=True)

    # Process each patch file
    for i, patch_file in enumerate(meta_patch_files):
        patch_str = read_file(patch_file)
        # Filter out the diff blocks related to task_agent.py
        if reset_task_agent:
            patch_str = filter_patch_by_files(patch_str, ["task_agent.py"])
        # Filter out the diff blocks related to meta_agent.py
        if reset_meta_agent:
            patch_str = filter_patch_by_files(patch_str, ["meta_agent.py"])
        # Write the patch to new file
        new_meta_patch_file = os.path.join(meta_patch_dir, f"model_patch_{i}.diff")
        with open(new_meta_patch_file, "w") as f:
            f.write(patch_str)
        new_meta_patch_files.append(new_meta_patch_file)

    return new_meta_patch_files

def apply_diffs_container(container, patch_files, repo_name=REPO_NAME, verbose=True):
    # Apply all diffs
    patch_files = patch_files or []
    for patch_file in patch_files:
        # Read and filter the patch to exclude domains/ folder changes
        patch_content = read_file(patch_file)
        filtered_patch = filter_patch_by_files(patch_content, ["domains/"])

        # Write filtered patch to a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(filtered_patch)
            filtered_patch_file = f.name

        try:
            copy_to_container(container, filtered_patch_file, f"/{repo_name}/parent_patch.txt", verbose=verbose)
            exec_result = container.exec_run(
                f"/bin/sh -c 'patch -p1 < /{repo_name}/parent_patch.txt'",
                workdir=f"/{repo_name}",
            )
            log_container_output(exec_result, verbose=verbose)
            exec_result = container.exec_run(
                f"rm /{repo_name}/parent_patch.txt", workdir=f"/{repo_name}"
            )
            log_container_output(exec_result, verbose=verbose)
        finally:
            os.remove(filtered_patch_file)

    # Stage all changes
    exec_result = container.exec_run("git add --all", workdir=f"/{repo_name}")
    log_container_output(exec_result, verbose=verbose)

    # Check if there's anything to commit
    exec_result = container.exec_run("git status --porcelain", workdir=f"/{repo_name}/")
    log_container_output(exec_result, verbose=verbose)
    status_output = exec_result.output.decode("utf-8").strip()

    if status_output:
        # If there are changes, commit them
        exec_result = container.exec_run(
            "git -c user.name='user' -c user.email='you@example.com' commit -m 'a nonsense commit message'",
            workdir=f"/{repo_name}/",
        )
        log_container_output(exec_result, verbose=verbose)
        commit_output = exec_result.output.decode("utf-8")
        commit_hash = commit_output.split()[1].strip("[]")  # Extract the hash part
    else:
        # Otherwise, get the current commit hash
        exec_result = container.exec_run("git rev-parse HEAD", workdir=f"/{repo_name}/")
        log_container_output(exec_result, verbose=verbose)
        commit_hash = exec_result.output.decode("utf-8").strip()

    # # Install requirements again in case of any changes
    # container.exec_run(
    #     cmd=[
    #         "/bin/bash",
    #         "-lc",
    #         "HTTPS_PROXY=http://fwdproxy:8080 "
    #         "HTTP_PROXY=http://fwdproxy:8080 "
    #         "FTP_PROXY=http://fwdproxy:8080 "
    #         "https_proxy=http://fwdproxy:8080 "
    #         "http_proxy=http://fwdproxy:8080 "
    #         "ftp_proxy=http://fwdproxy:8080 "
    #         "http_no_proxy='.facebook.com,.tfbnw.net,.fb.com' "
    #         f"python -m pip install -r '/{REPO_NAME}/requirements.txt'",
    #     ],
    #     workdir="/",
    # )
    # log_container_output(exec_result)

    return commit_hash


def select_parent(archive, output_dir, domains, method="best"):
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
        # Get the first initial node as the only candidate
        candidates[archive[0]] = 0.0
        # raise ValueError("No evaluation results found in archive.")

    # Build child counts from metadata
    child_counts = {genid: 0 for genid in candidates}
    for genid in archive:
        parent = get_parent_genid(output_dir, genid)
        if parent in child_counts:
            child_counts[parent] += 1

    # Select parent randomly
    if method == "random":
        return random.choice(list(candidates.keys()))

    # Select the latest compiled node
    elif method == "latest":
        return list(candidates.keys())[-1]

    # Select the best compiled node
    elif method == "best":
        return max(candidates, key=candidates.get)  # pyright: ignore[reportCallIssue]

    # Select the best compiled node with probability proportional to score
    elif method == "score_prop":
        commits = list(candidates.keys())
        scores = [candidates[commit] for commit in commits]
        mid_point = np.mean(sorted(scores, reverse=True)[:3])
        scores = [1 / (1 + math.exp(-10 * (score - mid_point))) for score in scores]
        total = sum(scores)
        probabilities = (
            [s / total for s in scores]
            if total > 0
            else [1 / len(scores)] * len(scores)
        )
        return random.choices(commits, weights=probabilities)[0]

    # Select the best compiled node with probability proportional to score and inversely proportional to number of children
    elif method == "score_child_prop":
        commits = list(candidates.keys())
        scores = [candidates[commit] for commit in commits]
        mid_point = np.mean(sorted(scores, reverse=True)[:3])
        scores = [1 / (1 + math.exp(-10 * (score - mid_point))) for score in scores]
        penalties = [math.exp(-(child_counts[commit]/8)**3) for commit in commits]
        combined = [s * p for s, p in zip(scores, penalties)]
        total = sum(combined)
        probabilities = (
            [c / total for c in combined]
            if total > 0
            else [1 / len(combined)] * len(combined)
        )
        return random.choices(commits, weights=probabilities)[0]

    else:
        raise ValueError(f"Unknown method '{method}'")

def get_latest_can_select_parent(archive, output_dir, trunc_genid=None):
    # Truncate archive
    if trunc_genid is not None:
        if is_starting_node(trunc_genid):
            return None
        archive = [genid for genid in archive if is_starting_node(genid) or genid < trunc_genid]

    # Get latest can_select_parent
    for genid in archive[::-1]:
        if is_starting_node(genid):
            return genid
        can_select_next_parent = get_node_metadata_key(output_dir, genid, "can_select_next_parent")
        if can_select_next_parent:
            return genid

    # Shouldn't reach here
    print("shouldn't reach here")
    return None

def run_smoke_test(container, repo_name=REPO_NAME):
    """
    Lightweight smoke test: verify TaskAgent is
    instantiable and forward() has the right
    signature. No actual LLM calls -- just import,
    instantiate, and inspect.
    Returns True on pass, False otherwise.
    """
    smoke_test_code = (
        "from task_agent import TaskAgent; "
        "agent = TaskAgent(model='claude-3-haiku-20240307'); "
        "import inspect; "
        "sig = inspect.signature(agent.forward); "
        "print('smoke_test_passed')"
    )
    command = [
        "timeout",
        "60",  # 60s timeout
        "python",
        "-c",
        smoke_test_code,
    ]
    try:
        exec_result = container.exec_run(cmd=command, workdir=f"/{repo_name}")
        log_container_output(exec_result)
        output = exec_result.output.decode() if exec_result.output else ""
        return exec_result.exit_code == 0 and "smoke_test_passed" in output
    except Exception:
        return False


def run_commands_to_check_compilation(container, run_baseline=None, edit_select_parent=False):
    # Run commands to check if the agents are compilable
    if run_baseline and "dgm" in run_baseline:
        command = [
            "timeout",
            "300",  # 5m timeout
            "python",
            "-c",
            "from coding_agent import CodingAgent",
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)
        if exec_result.exit_code != 0:
            raise Exception("coding_agent is not compilable")

    else:
        command = [
            "timeout",
            "300",  # 5m timeout
            "python",
            "-c",
            "from meta_agent import MetaAgent",
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)
        if exec_result.exit_code != 0:
            raise Exception("meta_agent is not compilable")

    command = [
        "timeout",
        "300",  # 5m timeout
        "python",
        "-c",
        "from task_agent import TaskAgent",
    ]
    exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
    log_container_output(exec_result)
    if exec_result.exit_code != 0:
        raise Exception("task_agent is not compilable")

    if edit_select_parent:
        command = [
            "timeout",
            "300",  # 5m timeout
            "python",
            "-c",
            "from select_next_parent import select_next_parent",
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)
        if exec_result.exit_code != 0:
            raise Exception("select_next_parent is not compilable")
