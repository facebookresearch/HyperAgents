"""Shared fixtures for HyperAgents test suite."""

import importlib
import importlib.util
import os
import json
import sys
import tempfile
import shutil
import types

import pytest

# ---- Project root on sys.path ----
_PROJ = os.path.normpath(
    "C:/Users/ryuke/Desktop/Projects/Hyperagents"
)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)


def _install_lightweight_mocks():
    """Install minimal mock modules so that project
    modules can be imported without heavy deps like
    docker, litellm, backoff, torch, etc.

    Only installs mocks for modules NOT already
    present -- safe to call multiple times.
    """
    def _ensure(name, factory):
        if name not in sys.modules:
            sys.modules[name] = factory()

    # docker
    _ensure("docker", lambda: types.ModuleType("docker"))

    # utils.docker_utils
    def _make_docker_utils():
        m = types.ModuleType("utils.docker_utils")
        m.copy_to_container = lambda *a, **k: None
        m.log_container_output = lambda *a, **k: None
        return m
    _ensure("utils.docker_utils", _make_docker_utils)

    # utils.git_utils
    def _make_git_utils():
        m = types.ModuleType("utils.git_utils")
        m.commit_repo = lambda *a, **k: "abc123"
        m.get_git_commit_hash = lambda *a, **k: "abc"
        return m
    _ensure("utils.git_utils", _make_git_utils)

    # backoff
    def _make_backoff():
        m = types.ModuleType("backoff")
        m.expo = "expo"
        m.on_exception = (
            lambda *a, **kw: (lambda f: f)
        )
        return m
    _ensure("backoff", _make_backoff)

    # requests / requests.exceptions
    def _make_requests():
        m = types.ModuleType("requests")
        exc = types.ModuleType("requests.exceptions")
        exc.RequestException = Exception
        m.exceptions = exc
        sys.modules["requests.exceptions"] = exc
        return m
    _ensure("requests", _make_requests)

    # litellm
    def _make_litellm():
        m = types.ModuleType("litellm")
        m.drop_params = True
        m.completion = lambda **kw: None
        return m
    _ensure("litellm", _make_litellm)

    # dotenv
    def _make_dotenv():
        m = types.ModuleType("dotenv")
        m.load_dotenv = lambda *a, **kw: None
        return m
    _ensure("dotenv", _make_dotenv)

    # utils.thread_logger
    def _make_thread_logger():
        m = types.ModuleType("utils.thread_logger")
        class FakeLM:
            def __init__(self, **kw):
                self.log = print
        m.ThreadLoggerManager = FakeLM
        return m
    _ensure(
        "utils.thread_logger", _make_thread_logger
    )

    # tqdm (used by genesis evaluator)
    def _make_tqdm():
        m = types.ModuleType("tqdm")
        m.tqdm = lambda *a, **kw: iter([])
        return m
    _ensure("tqdm", _make_tqdm)

    # pandas (used by ensemble.py)
    def _make_pandas():
        m = types.ModuleType("pandas")
        m.read_csv = lambda *a, **kw: None
        return m
    _ensure("pandas", _make_pandas)
    _ensure("pd", _make_pandas)


# Install mocks at import time so all test modules
# benefit.
_install_lightweight_mocks()


def load_module_from_file(module_name, file_path):
    """Load a Python module directly from a file path,
    bypassing package __init__.py files.

    Useful for modules whose package __init__ imports
    heavy deps (e.g., torch).
    """
    abs_path = os.path.join(_PROJ, file_path)
    spec = importlib.util.spec_from_file_location(
        module_name, abs_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory, cleaned up after test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_archive_jsonl(tmp_dir):
    """Create a sample archive.jsonl file with valid data."""
    path = os.path.join(tmp_dir, "archive.jsonl")
    entries = [
        {
            "current_genid": 0,
            "archive": [0],
        },
        {
            "current_genid": 1,
            "archive": [0, 1],
        },
        {
            "current_genid": 2,
            "archive": [0, 1, 2],
        },
    ]
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


@pytest.fixture
def sample_metadata_dir(tmp_dir):
    """Create gen_X directories with metadata.json files."""
    for genid in range(3):
        gen_dir = os.path.join(
            tmp_dir, f"gen_{genid}"
        )
        os.makedirs(gen_dir, exist_ok=True)
        metadata = {
            "parent_genid": genid - 1 if genid > 0 else None,
            "valid_parent": True,
            "prev_patch_files": [],
            "curr_patch_files": [],
        }
        with open(
            os.path.join(gen_dir, "metadata.json"), "w"
        ) as f:
            json.dump(metadata, f)
    return tmp_dir
