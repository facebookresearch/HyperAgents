# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import docker
from analysis.plot_progress import plot_progress_single, plot_progress_together
from analysis.visualize_archive import (
    visualize_archive_single,
    visualize_archive_together,
)

from utils.common import file_exist_and_not_empty, load_json_file
from utils.constants import REPO_NAME
from utils.docker_utils import (
    build_container,
    cleanup_container,
    copy_from_container,
    copy_to_container,
    log_container_output,
    safe_log,
    setup_logger,
)
from utils.domain_utils import (
    can_domain_ensembled,
    get_domain_eval_subset,
    get_domain_splits,
    get_domain_stagedeval_samples,
)
from utils.gl_utils import (
    apply_diffs_container,
    get_patch_files,
    get_score,
    load_archive_data,
    run_commands_to_check_compilation,
    run_smoke_test,
    select_parent,
    setup_initial_gen,
    update_and_save_archive,
    update_node_metadata,
    get_latest_can_select_parent,
    is_starting_node,
    process_meta_patch_files,
)


def run_harness_polyglot(root_dir, output_dir, genid, skip_staged_eval=False, num_samples=-1):
    # NOTE: the harness for polyglot is different because each task instance needs a docker container
    from domains.polyglot.harness import harness as harness_polyglot
    from domains.polyglot.report import report as report_polyglot

    eval_output_dir = os.path.join(output_dir, f"gen_{genid}", "polyglot_eval")
    test_more_threshold = 0.4  # NOTE: same setting as that in DGM
    model_name_or_path = "eval_run"
    patch_files = get_patch_files(output_dir, genid)
    run_next_eval = True

    # Small sample size evaluation for staged eval
    if not skip_staged_eval:
        test_task_list = load_json_file("./domains/polyglot/subsets/small.json")
        dnames = harness_polyglot(
            test_task_list=test_task_list,
            num_samples=-1,
            max_workers=10,
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=1,
            num_evals_parallel=1,
            pred_dname=eval_output_dir,
            output_dir=eval_output_dir,
            root_dir=root_dir,
        )
        report_polyglot(output_dir=eval_output_dir, run_keyword=model_name_or_path, expected_num_tasks=len(test_task_list))
        stagedeval_score = get_score("polyglot", output_dir, genid)
        run_next_eval = stagedeval_score is not None and stagedeval_score >= test_more_threshold

    # Check if additional evaluation should be run
    if run_next_eval:
        test_task_list_more = load_json_file("./domains/polyglot/subsets/medium.json")
        dnames = harness_polyglot(
            test_task_list=test_task_list + test_task_list_more,
            num_samples=num_samples,
            max_workers=10,
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=1,
            num_evals_parallel=1,
            pred_dname=eval_output_dir,
            output_dir=eval_output_dir,
            root_dir=root_dir,
        )
        report_polyglot(output_dir=eval_output_dir, run_keyword=model_name_or_path, expected_num_tasks=len(test_task_list + test_task_list_more))

    # Update metadata
    update_node_metadata(output_dir, genid, {"run_full_eval": run_next_eval})

def select_next_parent_container(
    docker_client,
    domains,
    generate_output_dir,
    archive,
    root_dir="./",
    root_commit="HEAD",
    max_attempts=10,
):
    # Setup logger
    logger = setup_logger(os.path.join(generate_output_dir, "select_next_parent.log"))

    # Get the latest node that can select parent
    latest_node = get_latest_can_select_parent(archive, generate_output_dir)
    safe_log(f"select_next_parent_container: latest_node={latest_node}")
    prev_patch_files = get_patch_files(generate_output_dir, latest_node)

    # Create and start the Docker container
    image_name = f"{REPO_NAME}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    container_name = f"{REPO_NAME}-nextparent-container-{run_id}"
    container = build_container(docker_client, root_dir, image_name, container_name, verbose=False)
    container.start()
    container_output_folder = "/tmp/"

    try:
        # Apply all lineage diffs
        commit_hash = apply_diffs_container(container, prev_patch_files, verbose=False)

        # Copy generate_output_dir to container
        container_generate_output_dir = os.path.join(
            container_output_folder, generate_output_dir.split(os.sep)[-1]
        )
        copy_to_container(
            container,
            source_path=generate_output_dir,
            dest_path=container_generate_output_dir,
            verbose=False,
        )

        # Get next parent in container
        command = [
            "timeout",
            "3600",  # 1h timeout
            "python",
            "-m",
            "utils.run_select_next_parent",
            "--domains",
            *domains,
            "--generate_output_dir",
            container_generate_output_dir,
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result, verbose=True)

        # Fail-fast on deterministic script errors (won't succeed on retry)
        if exec_result.exit_code != 0:
            output_text = (
                exec_result.output.decode()
                if exec_result.output else ""
            )
            output_lower = output_text.lower()
            _deterministic_markers = [
                "syntaxerror", "syntax error",
                "importerror", "import error", "no module named",
                "nameerror", "name error",
                "attributeerror", "attribute error",
                "typeerror", "type error",
                "indentationerror", "indentation error",
                "traceback (most recent call last)",
            ]
            if any(marker in output_lower for marker in _deterministic_markers):
                raise RuntimeError(
                    "Deterministic script failure"
                    " in select_next_parent"
                    f" (exit code"
                    f" {exec_result.exit_code}):"
                    f" {output_text[-500:]}"
                )

        # Get next parent outputs
        container_output_strings = exec_result.output.decode().strip().split("\n")
        next_parent_genid = container_output_strings[-1]
        if not is_starting_node(next_parent_genid):
            next_parent_genid = int(next_parent_genid)

    except RuntimeError as e:
        # Deterministic errors — do not retry
        safe_log(
            "Deterministic error in"
            " select_next_parent_container,"
            f" not retrying: {e}"
        )
        update_node_metadata(
            generate_output_dir, latest_node,
            {"can_select_next_parent": False},
        )
        next_parent_genid = None
        max_attempts = 0  # Prevent retry below

    except Exception as e:
        safe_log(
            "Error in"
            f" select_next_parent_container: {e}"
        )
        update_node_metadata(
            generate_output_dir, latest_node,
            {"can_select_next_parent": False},
        )
        next_parent_genid = None

    # Even on errors or KeyboardInterrupt
    finally:
        # Reset to the root commit
        exec_result = container.exec_run(
            cmd=["git", "reset", "--hard", root_commit], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result, verbose=False)
        exec_result = container.exec_run(
            cmd=["git", "clean", "-fd"], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result, verbose=False)

        # Cleanup container
        cleanup_container(container, verbose=False)

    # Try again if no parent is selected
    if next_parent_genid is None:
        if max_attempts > 0:
            next_parent_genid = select_next_parent_container(
                docker_client,
                domains,
                generate_output_dir,
                archive,
                root_dir,
                root_commit,
                max_attempts=max_attempts - 1,
            )
        else:
            # In case of infinite recursive, but it should not happen
            raise Exception("Max attempts reached in select_next_parent_container")

    return next_parent_genid

def get_ensemble_scores_container(
    docker_client,
    domain,
    generate_output_dir,
    gen_output_dir,
    root_dir="./",
    root_commit="HEAD",
    prev_patch_files=[],
    num_samples=-1,
    max_workers=5,
    subsets=[],
):
    # Setup logger
    logger = setup_logger(os.path.join(gen_output_dir, "ensemble.log"))

    # Create and start the Docker container
    image_name = f"{REPO_NAME}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    container_name = f"{REPO_NAME}-ens-container-{run_id}"
    container = build_container(docker_client, root_dir, image_name, container_name)
    container.start()
    container_output_folder = "/tmp/"

    try:
        # Apply all lineage diffs
        commit_hash = apply_diffs_container(container, prev_patch_files)

        # Copy generate_output_dir to container
        container_generate_output_dir = os.path.join(
            container_output_folder, generate_output_dir.split(os.sep)[-1]
        )
        copy_to_container(
            container,
            source_path=generate_output_dir,
            dest_path=container_generate_output_dir,
        )

        # Get ensemble scores in container
        scores = []
        for subset in subsets:
            command = [
                "timeout",
                "10800",  # 3h timeout
                "python",
                "-m",
                "utils.run_ensemble",
                "--domain",
                domain,
                "--generate_output_dir",
                container_generate_output_dir,
                "--num_samples",
                str(num_samples),
                "--max_workers",
                str(max_workers),
                "--subset",
                subset,
            ]
            exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
            log_container_output(exec_result)

            # Get ensemble outputs
            container_output_strings = exec_result.output.decode().strip().split("\n")
            score = float(container_output_strings[-3])
            scores.append(score)
            container_predictions_path = container_output_strings[-2]
            container_report_path = container_output_strings[-1]

            # Copy container outputs to local
            predictions_file = os.path.basename(container_predictions_path)
            report_file = os.path.basename(container_report_path)
            local_predictions_path = os.path.join(gen_output_dir, predictions_file)
            local_report_path = os.path.join(gen_output_dir, report_file)
            copy_from_container(
                container,
                source_path=container_predictions_path,
                dest_path=local_predictions_path,
            )
            copy_from_container(
                container,
                source_path=container_report_path,
                dest_path=local_report_path,
            )

    except Exception as e:
        safe_log(f"Error in get_ensemble_scores_container: {e}")
        scores = [None] * len(subsets)

    # Even on errors or KeyboardInterrupt
    finally:
        # Reset to the root commit
        exec_result = container.exec_run(
            cmd=["git", "reset", "--hard", root_commit], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result)
        exec_result = container.exec_run(
            cmd=["git", "clean", "-fd"], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result)

        # Cleanup container
        cleanup_container(container)

    return scores


def eval_produced_agent(
    container,
    container_output_folder,
    gen_output_dir,
    domain,
    eval_samples=-1,
    eval_workers=10,
    eval_subset="_filtered_100_train",
    eval_test=False,
):
    # Evaluate the produced agent
    splits = get_domain_splits(domain, eval_test=eval_test)
    for split in splits:  # pyright: ignore
        safe_log(f"Evaluating the produced agent on {domain} {eval_samples} {split}...")
        eval_run_id = f"{domain}_eval" if split == "train" else f"{domain}_eval_{split}"
        container_evaloutput_folder = os.path.join(container_output_folder, eval_run_id)
        command = [
            "timeout",
            "18000",  # 5h timeout
            "python",
            "-m",
            "domains.harness",
            "--agent_path",
            "./task_agent.py",
            "--output_dir",
            container_output_folder,
            "--run_id",
            eval_run_id,
            "--domain",
            domain,
            "--num_samples",
            str(eval_samples),
            "--num_workers",
            str(eval_workers),
            "--subset",
            eval_subset.replace("_train", f"_{split}"),
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)
        command = [
            "timeout",
            "10800",  # 3h timeout
            "python",
            "-m",
            "domains.report",
            "--domain",
            domain,
            "--dname",
            os.path.join(container_output_folder, eval_run_id),
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)
        # Copy container outputs to local, evaluation results
        evaloutput_folder = os.path.join(gen_output_dir, eval_run_id)
        copy_from_container(
            container,
            source_path=container_evaloutput_folder,
            dest_path=evaloutput_folder,
        )


def copy_prev_eval_to_container(
    container,
    prev_eval_path,
    container_output_folder,
    current_genid=None,
    container_folder_name=None,
):
    """Copy the entire prev_eval_path into the container, then remove unwanted files/dirs in the container"""
    if not os.path.exists(prev_eval_path):
        raise FileNotFoundError(f"Previous eval path not found: {prev_eval_path}")

    # Normalize and construct destination path in container
    prev_eval_path = os.path.normpath(prev_eval_path)
    tail = os.path.join(*prev_eval_path.split(os.sep)[-1:])
    container_prev_eval_path = os.path.join(container_output_folder, tail)

    # Ensure destination parent exists
    container.exec_run(["mkdir", "-p", container_output_folder], workdir="/")

    # Copy the whole tree into the container in one go
    copy_to_container(
        container, source_path=prev_eval_path, dest_path=container_prev_eval_path
    )

    # Now prune inside the container
    prune_cmds = [
        # Remove current genid folder
        f"find '{container_prev_eval_path}' -type d -name 'gen_{current_genid}' -prune -exec rm -rf {{}} +",
        # 1) Remove val/test eval directories
        f"find '{container_prev_eval_path}' -type d -name '*_eval_val*' -prune -exec rm -rf {{}} +",
        f"find '{container_prev_eval_path}' -type d -name '*_eval_test*' -prune -exec rm -rf {{}} +",
        # 2) Remove any directories containing the repo name (copied worktrees, etc.)
        f"find '{container_prev_eval_path}' -type d -name '*{REPO_NAME}*' -prune -exec rm -rf {{}} +",
        # 3) Remove compiled Python files
        f"find '{container_prev_eval_path}' -type f -name '*.pyc' -delete",
        # 4) Remove files whose base name indicates val/test (with/without extensions)
        #    *_val, *_val.*, *_val_*, and same for _test
        f"find '{container_prev_eval_path}' -type f \\( -name '*_val' -o -name '*_val.*' -o -name '*_val_*' \\) -delete",
        f"find '{container_prev_eval_path}' -type f \\( -name '*_test' -o -name '*_test.*' -o -name '*_test_*' \\) -delete",
    ]

    for cmd in prune_cmds:
        exec_result = container.exec_run(["bash", "-lc", cmd], workdir="/")

    # Confirm files remaining were copied
    exec_result = container.exec_run(
        ["ls", "-l", container_prev_eval_path], workdir="/"
    )
    log_container_output(exec_result)

    # Move the folder to a new name
    if container_folder_name is not None:
        new_container_prev_eval_path = os.path.join(
            container_output_folder, container_folder_name
        )
        container.exec_run(
            ["mv", container_prev_eval_path, new_container_prev_eval_path], workdir="/"
        )
        log_container_output(exec_result)
        container_prev_eval_path = new_container_prev_eval_path

    return container_prev_eval_path


def generate(
    docker_client,
    domains,
    output_dir,
    run_id,
    current_genid,
    parent_genid,
    root_dir,
    root_commit="main",
    eval_samples=-1,
    eval_workers=10,
    eval_subsets="_filtered_100",
    meta_patch_files=None,
    run_meta_agent=True,
    run_baseline=None,
    optimize_option="only_agent",
    agent_archive_path=None,
    eval_test=False,
    skip_staged_eval=False,
    edit_select_parent=False,
    max_generation=None,
):
    # Setup local output folder
    prev_gen_dir = os.path.join(output_dir, f"gen_{parent_genid}")
    gen_output_dir = os.path.join(output_dir, f"gen_{current_genid}")
    os.makedirs(gen_output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(gen_output_dir, "generate.log"))  # Set up logger
    metadata = {
        "gen_output_dir": gen_output_dir,
        "current_genid": current_genid,
        "parent_genid": parent_genid,
        "run_baseline": run_baseline,
        "prev_patch_files": [],
        "curr_patch_files": [],
        "parent_agent_success": not run_meta_agent,  # meta agent success if not run
        "optimize_option": optimize_option,
        "agent_archive_path": agent_archive_path,
        "can_select_next_parent": True,
    }
    run_eval = not run_meta_agent  # always run eval if not running meta agent
    metadata["run_eval"] = run_eval
    print(metadata)

    # Create and start the Docker container
    image_name = f"{REPO_NAME}"
    container_name = f"{REPO_NAME}-gl-container-{run_id}"
    container = build_container(
        docker_client,
        root_dir,
        image_name,
        container_name,
        domains=domains,
    )
    container.start()
    container_output_folder = "/tmp/"

    try:
        # Make a copy of the repo
        if run_baseline and "no_selfimprove" in run_baseline:
            donottouch_reponame = f"/DONOTTOUCH_{REPO_NAME}"
            exec_result = container.exec_run(
                cmd=["cp", "-r", f"/{REPO_NAME}", donottouch_reponame],
                workdir=f"/",
            )
            log_container_output(exec_result)
            meta_patch_files = meta_patch_files or []
            _ = apply_diffs_container(container, meta_patch_files, repo_name=donottouch_reponame)

        # Apply meta patches (only for starting node, because subsequent generations will inherit the patches from the parent)
        if is_starting_node(current_genid):
            meta_patch_files = meta_patch_files or []
            commit_hash = apply_diffs_container(container, meta_patch_files)
            metadata["prev_patch_files"] += meta_patch_files

        # Apply all lineage diffs
        patch_files = get_patch_files(output_dir, parent_genid)
        metadata["prev_patch_files"] += patch_files
        commit_hash = apply_diffs_container(container, patch_files)

        if run_meta_agent:
            if run_baseline and "dgm" in run_baseline:
                # Get problem statement (DGM specific)
                from baselines.dgm.utils import get_problem_statement
                problem_statement = get_problem_statement(
                    root_dir, output_dir, parent_genid, domains,
                    customized="custom" in run_baseline,
                )

            else:
                # Copy another agent archive to container
                if optimize_option == "only_ensemble":
                    container_agent_archive_path = copy_prev_eval_to_container(
                        container,
                        agent_archive_path,
                        container_output_folder,
                        current_genid=current_genid,
                        container_folder_name="agent_archive",
                    )

                # Copy previous generations to container
                if run_baseline == "no_archive":
                    container_prev_eval_path = os.path.join(
                        container_output_folder, *prev_gen_dir.split(os.sep)[-2:]
                    )
                    copy_to_container(
                        container,
                        source_path=prev_gen_dir,
                        dest_path=container_prev_eval_path,
                    )
                else:
                    container_prev_eval_path = copy_prev_eval_to_container(
                        container, output_dir, container_output_folder, current_genid=current_genid,
                    )

            # Run meta agent
            safe_log("Running meta agent...")
            container_agentoutput_folder = os.path.join(
                container_output_folder, "agent_output"
            )
            container_chat_history_file = os.path.join(
                container_agentoutput_folder, "meta_agent_chat_history.md"
            )
            if run_baseline and "dgm" in run_baseline:
                command = [
                    "timeout",
                    "21600",  # 6h timeout
                    "python",
                    "coding_agent.py",
                    "--problem_statement",
                    problem_statement,
                    "--chat_history_file",
                    container_chat_history_file,
                    "--git_dir",
                    f"/{REPO_NAME}",
                    "--base_commit",
                    commit_hash,
                    "--outdir",
                    container_agentoutput_folder,
                ]
            else:
                command = [
                    "timeout",
                    "21600",  # 6h timeout
                    "python",
                    "run_meta_agent.py",
                    "--chat_history_file",
                    container_chat_history_file,
                    "--repo_path",
                    f"/{REPO_NAME}/",
                    "--evals_folder",
                    container_prev_eval_path,
                    "--git_dir",
                    f"/{REPO_NAME}",
                    "--base_commit",
                    commit_hash,
                    "--outdir",
                    container_agentoutput_folder,
                    "--iterations_left",
                    str(max_generation - current_genid),
                    *(
                        # If domain is polyglot, for a fair comparison with DGM
                        ["--model", "claude-3-5-sonnet-20241022"] if domains == ["polyglot"] else []
                    ),
                ]

            run_workdir = (
                f"/DONOTTOUCH_{REPO_NAME}"
                if run_baseline and "no_selfimprove" in run_baseline
                else f"/{REPO_NAME}"
            )
            exec_result = container.exec_run(cmd=command, workdir=run_workdir)
            log_container_output(exec_result)
            metadata["parent_agent_success"] = exec_result.exit_code == 0

            # Copy container outputs to local
            local_agentoutput_folder = os.path.join(gen_output_dir, "agent_output/")
            copy_from_container(
                container,
                source_path=container_agentoutput_folder,
                dest_path=local_agentoutput_folder,
            )

            # Check if agent produced a diff
            local_patch_file = os.path.join(
                local_agentoutput_folder, "model_patch.diff"
            )
            metadata["curr_patch_files"].append(local_patch_file)
            run_eval = file_exist_and_not_empty(local_patch_file)
            metadata["run_eval"] = run_eval

            # Run commands to check if the agents are compilable
            run_commands_to_check_compilation(container, run_baseline=run_baseline, edit_select_parent=edit_select_parent)

            # Lightweight smoke test post-compilation
            if not run_smoke_test(container):
                safe_log(
                    "Smoke test failed: TaskAgent"
                    " not instantiable or forward()"
                    " signature invalid."
                    " Skipping evaluation."
                )
                run_eval = False
                metadata["run_eval"] = False

        # Evaluate the produced agent
        if run_eval and "agent" in optimize_option:
            log_path = os.path.join(gen_output_dir, "generate.log")

            def eval_agent_worker(domain, eval_subset, eval_n):
                setup_logger(log_path)  # Re-setup logger because of threading
                eval_produced_agent(
                    container,
                    container_output_folder,
                    gen_output_dir,
                    domain=domain,
                    eval_samples=eval_n,
                    eval_workers=eval_workers,
                    eval_subset=eval_subset,
                    eval_test=eval_test,
                )

            # Small sample size evaluation for staged eval
            if not skip_staged_eval:
                stagedeval_samples = [
                    get_domain_stagedeval_samples(domain) for domain in domains
                ]
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(eval_agent_worker, d, s, n)
                        for d, s, n in zip(domains, eval_subsets, stagedeval_samples)
                    ]
                    try:
                        for f in futures:
                            f.result()
                    except Exception as e:
                        # Cancel all other futures if any job fails
                        for future in futures:
                            if not future.done():
                                future.cancel()
                        raise
                stagedeval_scores = [
                    get_score(domain, output_dir, current_genid) for domain in domains
                ]
                run_next_eval = all(
                    [x is not None and x > 0 for x in stagedeval_scores]
                )
            else:
                run_next_eval = True

            # Full evaluation
            if run_next_eval:
                _per_domain_eval_samples = eval_samples
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(eval_agent_worker, d, s, n)
                        for d, s, n in zip(
                            domains, eval_subsets, _per_domain_eval_samples
                        )
                    ]
                    try:
                        for f in futures:
                            f.result()
                    except Exception as e:
                        # Cancel all other futures if any job fails
                        for future in futures:
                            if not future.done():
                                future.cancel()
                        raise
                metadata["run_full_eval"] = True

    except Exception as e:
        safe_log(f"Error in generate: {e}")
        metadata["run_eval"] = False

    # Even on errors or KeyboardInterrupt
    finally:
        # Reset to the root commit
        exec_result = container.exec_run(
            cmd=["git", "reset", "--hard", root_commit], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result)
        exec_result = container.exec_run(
            cmd=["git", "clean", "-fd"], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result)

        # Cleanup container
        cleanup_container(container)

        # Save metadata
        eval_successful = all(
            [
                get_score(domain, output_dir, current_genid) is not None
                for domain in domains
            ]
        )
        metadata["valid_parent"] = metadata["run_eval"] and (eval_successful or meta_patch_files is not None)
        with open(os.path.join(gen_output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    return metadata


def generate_loop(
    domains,
    run_id=None,
    max_generation=3,
    eval_samples=-1,
    eval_workers=5,
    eval_subsets=[],
    parent_selection="score_prop",
    resume_from=None,
    output_dir_parent=None,
    meta_patch_files=None,
    reset_task_agent=False,
    reset_meta_agent=False,
    copy_root_dir=None,  # To have the same initial repo
    run_baseline=None,
    optimize_option="only_agent",
    agent_archive_path=None,
    eval_test=False,
    skip_staged_eval=False,
    edit_select_parent=False,
):
    # Initialization
    docker_client = docker.DockerClient()

    # Clean up orphaned containers from previous crashed runs
    try:
        for c in docker_client.containers.list(all=True):
            if c.name and c.name.startswith(f"{REPO_NAME}-"):
                try:
                    safe_log(f"Removing orphaned container: {c.name}")
                    c.remove(force=True)
                except Exception:
                    pass  # Best-effort cleanup
    except Exception:
        pass  # Don't block the main loop if cleanup fails

    parent_selection = "latest" if run_baseline == "no_archive" else parent_selection
    if resume_from:
        output_dir = os.path.normpath(os.path.abspath(resume_from))
        run_id = os.path.basename(output_dir).split("generate_")[-1]
        root_dir, root_commit = setup_initial_gen(
            output_dir,
            domains,
            copy_root_dir=copy_root_dir,
            subsets=eval_subsets,
            resume=True,
            optimize_option=optimize_option,
            run_baseline=run_baseline,
            eval_test=eval_test,
            edit_select_parent=edit_select_parent,
        )
        archive = load_archive_data(
            os.path.join(output_dir, "archive.jsonl"), last_only=True
        )[
            "archive"
        ]  # pyright: ignore
    else:
        run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") if run_id is None else run_id
        )
        output_dir_parent = (
            os.path.join(os.getcwd(), "outputs/")
            if output_dir_parent is None
            else output_dir_parent
        )
        output_dir = os.path.normpath(
            os.path.join(output_dir_parent, f"generate_{run_id}/")
        )
        os.makedirs(output_dir, exist_ok=True)
        root_dir, root_commit = setup_initial_gen(
            output_dir,
            domains,
            copy_root_dir=copy_root_dir,
            subsets=eval_subsets,
            resume=False,
            optimize_option=optimize_option,
            run_baseline=run_baseline,
            eval_test=eval_test,
            edit_select_parent=edit_select_parent,
        )

        # Create initial node
        if meta_patch_files is None or len(meta_patch_files) <= 0:
            archive = update_and_save_archive(output_dir, [], new_node="initial")
            metadata = {
                "gen_output_dir": os.path.join(output_dir, f"gen_initial"),
                "prev_patch_files": [],
                "curr_patch_files": [],
                "run_eval": True,
            }
        elif reset_task_agent:
            # Task agent is the same as initial agent
            meta_patch_files = process_meta_patch_files(meta_patch_files, output_dir, reset_task_agent=reset_task_agent, reset_meta_agent=reset_meta_agent)
            archive = update_and_save_archive(output_dir, [], new_node="initial")
            gen_output_dir = os.path.join(output_dir, f"gen_initial")
            metadata = {
                "gen_output_dir": gen_output_dir,
                "current_genid": "initial",
                "parent_genid": None,
                "run_baseline": run_baseline,
                "prev_patch_files": meta_patch_files,
                "curr_patch_files": [],
                "optimize_option": optimize_option,
                "agent_archive_path": agent_archive_path,
                "can_select_next_parent": True,
                "run_eval": True,
                "run_full_eval": False,
                "valid_parent": True,
            }
            with open(os.path.join(gen_output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)
        else:
            # Process meta patch files
            meta_patch_files = process_meta_patch_files(meta_patch_files, output_dir, reset_task_agent=reset_task_agent, reset_meta_agent=reset_meta_agent)
            # add node 0, which is the evaled version of the patches applied
            archive = update_and_save_archive(output_dir, [], new_node=0)
            metadata = generate(
                docker_client,
                [d for d in domains if d != "polyglot"],
                output_dir,
                run_id,
                current_genid=0,
                parent_genid=None,
                root_dir=root_dir,
                root_commit=root_commit,
                eval_samples=eval_samples,
                eval_workers=eval_workers,
                eval_subsets=eval_subsets,
                meta_patch_files=meta_patch_files,
                run_meta_agent=False,
                run_baseline=run_baseline,
                optimize_option=optimize_option,
                agent_archive_path=agent_archive_path,
                eval_test=eval_test,
                skip_staged_eval=skip_staged_eval,
                edit_select_parent=edit_select_parent,
                max_generation=max_generation,
            )
            print(f"generate_loop: generation 0 completed, parent None")
            # Evaluate the agent on polyglot if needed
            if "polyglot" in domains:
                run_harness_polyglot(root_dir, output_dir, 0, skip_staged_eval=skip_staged_eval, num_samples=eval_samples[domains.index("polyglot")])

        # Evaluate the entire archive as an ensemble
        eval_ensemble = (
            "ensemble" in optimize_option
            and all(can_domain_ensembled(domain) for domain in domains)
            and run_baseline != "no_archive"
        )
        if metadata["run_eval"] and eval_ensemble:
            for domain, eval_subset, eval_n in zip(domains, eval_subsets, eval_samples):
                _ = get_ensemble_scores_container(
                    docker_client,
                    domain,
                    (
                        output_dir
                        if optimize_option != "only_ensemble"
                        else agent_archive_path
                    ),
                    gen_output_dir=metadata["gen_output_dir"],
                    root_dir=root_dir,
                    root_commit=root_commit,
                    prev_patch_files=metadata["prev_patch_files"]
                    + metadata["curr_patch_files"],
                    num_samples=eval_n,
                    subsets=[
                        eval_subset,
                        eval_subset.replace("_train", "_val"),
                        *(
                            [eval_subset.replace("_train", "_test")]
                            if eval_test
                            else []
                        ),
                    ],
                )

    # Save args
    with open(os.path.join(output_dir, "generate_loop.log"), "a") as f:
        args_dict = locals()
        args_str = ", ".join([f"{k}={v}" for k, v in args_dict.items()])
        f.write(f"Args: {args_str}\n")

    # Run generations
    start_genid = len(archive)
    if not edit_select_parent or run_baseline == "no_archive":
        parent_genid = select_parent(archive, output_dir, domains, method=parent_selection)
    else:
        parent_genid = select_next_parent_container(
            docker_client,
            domains,
            output_dir,
            archive,
            root_dir, root_commit,
        )
    for current_genid in range(start_genid, max_generation + 1):
        metadata = generate(
            docker_client,
            [d for d in domains if d != "polyglot"],
            output_dir,
            run_id,
            current_genid,
            parent_genid=parent_genid,
            root_dir=root_dir,
            root_commit=root_commit,
            eval_samples=eval_samples,
            eval_workers=eval_workers,
            eval_subsets=eval_subsets,
            meta_patch_files=meta_patch_files,
            run_meta_agent=True,
            run_baseline=run_baseline,
            optimize_option=optimize_option,
            agent_archive_path=agent_archive_path,
            eval_test=eval_test,
            skip_staged_eval=skip_staged_eval,
            edit_select_parent=edit_select_parent,
            max_generation=max_generation,
        )

        # NOTE: need to update and save archive before running ensembling eval
        archive = update_and_save_archive(output_dir, archive, new_node=current_genid)

        # Parent agent failed, update the metadata in the parent node
        if not metadata["parent_agent_success"]:
            update_node_metadata(output_dir, parent_genid, {"valid_parent": False})

        # Evaluate the agent on polyglot if needed
        if "polyglot" in domains:
            run_harness_polyglot(root_dir, output_dir, current_genid, skip_staged_eval=skip_staged_eval, num_samples=eval_samples[domains.index("polyglot")])

        # Evaluate the entire archive as an ensemble
        eval_ensemble = (
            "ensemble" in optimize_option
            and all(can_domain_ensembled(domain) for domain in domains)
            and run_baseline != "no_archive"
        )
        if metadata["run_eval"] and eval_ensemble:
            for domain, eval_subset, eval_n in zip(domains, eval_subsets, eval_samples):
                _ = get_ensemble_scores_container(
                    docker_client,
                    domain,
                    (
                        output_dir
                        if optimize_option != "only_ensemble"
                        else agent_archive_path
                    ),
                    gen_output_dir=metadata["gen_output_dir"],
                    root_dir=root_dir,
                    root_commit=root_commit,
                    prev_patch_files=metadata["prev_patch_files"]
                    + metadata["curr_patch_files"],
                    num_samples=eval_n,
                    subsets=[
                        eval_subset,
                        eval_subset.replace("_train", "_val"),
                        *(
                            [eval_subset.replace("_train", "_test")]
                            if eval_test
                            else []
                        ),
                    ],
                )

        # Make analysis plots
        # Per-domain plots
        for domain in domains:
            splits = get_domain_splits(domain)
            if optimize_option == "only_ensemble":
                score_types = ["ensemble"]
            elif eval_ensemble:
                score_types = ["agent", "ensemble", "max"]
            else:
                score_types = ["agent"]
            for split in splits:  # pyright: ignore
                for stype in score_types:
                    plot_progress_single(domain, output_dir, split=split, type=stype)
                    visualize_archive_single(
                        domain, output_dir, split=split, type=stype
                    )

        # Combined together plots across all domains (if there is more than one domain)
        if len(domains) > 1:
            domain_splits_sets = [set(get_domain_splits(d)) for d in domains]
            common_splits = (
                sorted(list(set.intersection(*domain_splits_sets)))
                if domain_splits_sets
                else []
            )
            if optimize_option == "only_ensemble":
                together_score_types = ["ensemble"]
            elif eval_ensemble:
                together_score_types = ["agent", "ensemble", "max"]
            else:
                together_score_types = ["agent"]
            for split in common_splits:
                for stype in together_score_types:
                    plot_progress_together(domains, output_dir, split=split, type=stype)
                    visualize_archive_together(
                        domains, output_dir, split=split, type=stype
                    )

        # Select next parent
        parent_genid = None
        if not edit_select_parent or run_baseline == "no_archive":
            parent_genid = select_parent(archive, output_dir, domains, method=parent_selection)
        else:
            parent_genid = select_next_parent_container(
                docker_client,
                domains,
                output_dir,
                archive,
                root_dir, root_commit,
            )

        print(f"generate_loop: generation {current_genid} completed, parent {parent_genid}")

    # Return output dir
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None, help="Run ID")
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",  # one or more domains
        choices=[
            "search_arena",
            "paper_review",
            "balrog_babyai",
            "balrog_babaisai",
            "balrog_minihack",
            "balrog_nle",
            "genesis_go2walking",
            "genesis_go2walkback",
            "genesis_go2hop",
            "polyglot",  # separate harness from the rest
            "imo_grading",
            "imo_proof",
        ],
        required=True,
        help="One or more domains to evaluate (must be from the allowed list)",
    )
    parser.add_argument(
        "--max_generation",
        type=int,
        default=10,
        help="Maximum number of evolution generations",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        nargs="+",
        default=None,
        help="Evaluation samples per domain (-1 for all). Provide one value per domain.",
    )
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=10,
        help="Number of evaluation workers in parallel",
    )
    parser.add_argument(
        "--parent_selection",
        type=str,
        default="score_child_prop",
        choices=["random", "latest", "best", "score_prop", "score_child_prop"],
        help="Parent selection method",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to an existing output folder to resume from",
    )
    parser.add_argument(
        "--output_dir_parent",
        type=str,
        default=None,
        help="Path to the parent output folder",
    )
    parser.add_argument(
        "--meta_patch_files",
        type=str,
        nargs="+",
        default=[],
        help="Meta patch files to apply",
    )
    parser.add_argument(
        "--reset_task_agent",
        default=False,
        action="store_true",
        help="Whether to reset the changes in the task agent (for self-referential self-improvement transfer experiments)",
    )
    parser.add_argument(
        "--reset_meta_agent",
        default=False,
        action="store_true",
        help="Whether to reset the changes in the meta agent (for self-referential self-improvement transfer experiments)",
    )
    parser.add_argument(
        "--copy_root_dir",
        type=str,
        default=None,
        help="Copy root dir for setup_initial_gen",
    )
    parser.add_argument(
        "--run_baseline",
        type=str,
        choices=[
            "no_selfimprove", "no_archive",
            "dgm", "dgm_custom",
            "dgm+no_selfimprove", "dgm_custom+no_selfimprove",
        ],
        default=None,
        help="Run baseline",
    )
    parser.add_argument(
        "--optimize_option",
        type=str,
        default="only_agent",
        choices=["both_agent_ensemble", "only_agent", "only_ensemble"],
        help="Which part of the algorithm to optimize",
    )
    parser.add_argument(
        "--agent_archive_path",
        type=str,
        default=None,
        help="Path to agent archive (required if --optimize_option=only_ensemble)",
    )
    parser.add_argument(
        "--eval_test",
        default=False,
        action="store_true",
        help="Always run test set evaluation",
    )
    parser.add_argument(
        "--skip_staged_eval",
        default=False,
        action="store_true",
        help="Skip staged evaluation",
    )
    parser.add_argument(
        "--edit_select_parent",
        default=False,
        action="store_true",
        help="Whether to allow the agent to edit the selection mechanism",
    )
    args = parser.parse_args()

    # Post-parse validation
    if args.optimize_option == "only_ensemble" and args.agent_archive_path is None:
        parser.error(
            "--agent_archive_path is required when --optimize_option=only_ensemble"
        )
    if args.eval_samples is None:
        eval_samples = [-1] * len(args.domains)
    elif len(args.eval_samples) == len(args.domains):
        eval_samples = args.eval_samples
    else:
        parser.error("--eval_samples must be a one per domain if provided")

    eval_subsets = [get_domain_eval_subset(d) for d in args.domains]
    output_dir = generate_loop(
        domains=args.domains,
        run_id=args.run_id,
        max_generation=args.max_generation,
        eval_samples=eval_samples,
        eval_workers=args.eval_workers,
        eval_subsets=eval_subsets,
        parent_selection=args.parent_selection,
        resume_from=args.resume_from,
        output_dir_parent=args.output_dir_parent,
        meta_patch_files=args.meta_patch_files,
        reset_task_agent=args.reset_task_agent,
        reset_meta_agent=args.reset_meta_agent,
        copy_root_dir=args.copy_root_dir,
        run_baseline=args.run_baseline,
        optimize_option=args.optimize_option,
        agent_archive_path=args.agent_archive_path,
        eval_test=args.eval_test,
        skip_staged_eval=args.skip_staged_eval,
        edit_select_parent=args.edit_select_parent,
    )
