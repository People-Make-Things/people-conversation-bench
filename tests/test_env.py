"""Tests for main-checkout .env loading from git worktrees."""

from __future__ import annotations

from pathlib import Path

import pytest

import env


def write_worktree(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "main"
    work = tmp_path / "work"
    git = main / ".git"
    wt = git / "worktrees" / "pmt-0"
    wt.mkdir(parents=True)
    (wt / "commondir").write_text("../..\n", encoding="utf-8")
    work.mkdir()
    (work / ".git").write_text(f"gitdir: {wt}\n", encoding="utf-8")
    return main, work


def test_worktree_resolves_main_checkout_root(tmp_path: Path) -> None:
    main, work = write_worktree(tmp_path)
    assert env.main_checkout_root(main) == main
    assert env.main_checkout_root(work) == main


def test_worktree_loads_main_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main, work = write_worktree(tmp_path)
    (main / ".env").write_text("XAI_API_KEY=from-main\n", encoding="utf-8")
    monkeypatch.setattr(env, "ROOT", work)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    env.load()

    assert env.get("XAI_API_KEY") == "from-main"


def test_plain_clone_uses_local_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = tmp_path / "clone"
    (clone / ".git").mkdir(parents=True)
    (clone / ".env").write_text("OPENAI_API_KEY=local\n", encoding="utf-8")
    monkeypatch.setattr(env, "ROOT", clone)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    env.load()

    assert env.get("OPENAI_API_KEY") == "local"


def test_worktree_without_commondir_still_finds_main(tmp_path: Path) -> None:
    main = tmp_path / "main"
    work = tmp_path / "work"
    wt = main / ".git" / "worktrees" / "pmt-0"
    wt.mkdir(parents=True)
    work.mkdir()
    (work / ".git").write_text(f"gitdir: {wt}\n", encoding="utf-8")
    assert env.main_checkout_root(work) == main
