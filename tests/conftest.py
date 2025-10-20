import os
from pathlib import Path
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow end-to-end tests")
    config.addinivalue_line("markers", "arrow: tests that require pyarrow")

def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / "vocab.txt").exists():
            return cur
        cur = cur.parent
    return start.resolve()

@pytest.fixture(scope="session")
def repo_root():
    return find_repo_root(Path(__file__).parent)

@pytest.fixture(scope="session")
def vocab_path(repo_root):
    vp = repo_root / "vocab.txt"
    if not vp.exists():
        pytest.skip("repo_root/vocab.txt not found; please place your official vocab at project root.")
    return str(vp)

@pytest.fixture(scope="session")
def py_env_with_src(repo_root):
    env = os.environ.copy()
    src = repo_root / "src"
    # Prepend repo root and src to PYTHONPATH to import smiles_bd.*
    pp = env.get("PYTHONPATH", "")
    parts = []
    if src.exists():
        parts.append(str(src))
    parts.append(str(repo_root))
    if pp:
        parts.append(pp)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env
