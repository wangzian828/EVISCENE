from __future__ import annotations

from eviscene.environment import get_env

from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv


def project_dotenv_candidates(repo_root: Optional[Path] = None) -> Iterable[Path]:
    """Yield explicit, working-directory, and checkout configuration locations.

    Installed wheels never search site-packages for credentials. Explicit
    configuration is authoritative; a typo must not select a different file.
    """

    root = (repo_root or Path(__file__).resolve().parent.parent).resolve()
    explicit = get_env("EVISCENE_ENV_FILE", "").strip()
    if explicit:
        yield Path(explicit).expanduser().resolve()
        return

    yield Path.cwd() / ".env"
    if repo_root is None and not (root / "pyproject.toml").is_file():
        return
    yield root / ".env"

    git_marker = root / ".git"
    if not git_marker.is_file():
        return
    marker = git_marker.read_text(encoding="utf-8").strip()
    prefix = "gitdir:"
    if not marker.lower().startswith(prefix):
        return
    git_dir = Path(marker[len(prefix) :].strip())
    if not git_dir.is_absolute():
        git_dir = (root / git_dir).resolve()
    common_marker = git_dir / "commondir"
    if not common_marker.is_file():
        return
    common_dir = Path(common_marker.read_text(encoding="utf-8").strip())
    if not common_dir.is_absolute():
        common_dir = (git_dir / common_dir).resolve()
    yield common_dir.parent / ".env"


def load_project_dotenv(repo_root: Optional[Path] = None) -> Optional[Path]:
    """Load the first available project dotenv without overriding process env."""

    seen: set[Path] = set()
    for candidate in project_dotenv_candidates(repo_root):
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            load_dotenv(dotenv_path=resolved, override=False)
            return resolved
    if get_env("EVISCENE_ENV_FILE", "").strip():
        raise FileNotFoundError("The explicitly configured EVISCENE_ENV_FILE does not exist")
    return None


LOADED_DOTENV = load_project_dotenv()
