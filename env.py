"""Load environment variables from the main checkout .env file."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def main_checkout_root(start: Path) -> Path:
    git_path = start / ".git"
    if git_path.is_dir():
        return start
    if not git_path.is_file():
        return start

    gitdir: Path | None = None
    for line in git_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("gitdir:"):
            gitdir = Path(line.split(":", 1)[1].strip())
            break
    if gitdir is None:
        return start
    if not gitdir.is_absolute():
        gitdir = (start / gitdir).resolve()

    commondir_file = gitdir / "commondir"
    if commondir_file.is_file():
        common = Path(commondir_file.read_text(encoding="utf-8").strip())
        if not common.is_absolute():
            common = (gitdir / common).resolve()
        return common.parent
    if gitdir.parent.name == "worktrees":
        return gitdir.parent.parent.parent
    return gitdir.parent


def load() -> None:
    load_dotenv(main_checkout_root(ROOT) / ".env")


def get(name: str, default: str | None = None) -> str | None:
    load()
    value = os.getenv(name, default)
    if value == "":
        return None
    return value


def require(name: str) -> str:
    value = get(name)
    if not value:
        raise SystemExit(
            f"Missing {name}. Set it in the main checkout .env (worktrees load that file)."
        )
    return value
