"""The Modal benchmark scripts must resolve the checkout root the same way as entrypoints.

Modal copies the entrypoint file to ``/root/<script>.py`` inside the
container and imports it there; ``Path(__file__).parents[2]`` of that path
does not exist. ``trl_replay_real.repo_root`` resolves to the checkout
locally and to ``/root`` in the container, and every script must use it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("modal")

from benchmarks.modal import forgetting_real, prefcheck_real, trl_replay_real  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = (REPO / "benchmarks" / "modal" / "forgetting_real.py", REPO / "benchmarks" / "modal" / "prefcheck_real.py")


def test_repo_root_is_the_checkout_locally_and_root_in_the_container(monkeypatch):
    assert trl_replay_real.repo_root() == REPO
    monkeypatch.setattr(trl_replay_real, "__file__", "/root/trl_replay_real.py")
    assert trl_replay_real.repo_root() == Path("/root")


@pytest.mark.parametrize("script", (forgetting_real, prefcheck_real), ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_scripts_resolve_their_sources_through_repo_root(script):
    assert script.repo_root is trl_replay_real.repo_root
    assert script.with_sources is trl_replay_real.with_sources
    assert script.image is not None


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_scripts_do_not_index_parents_of_their_own_path(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
            # Only the sys.path bootstrap may look at parents, and it must guard the length.
            assert isinstance(node.ctx, ast.Load)
            line = path.read_text().split("\n")[node.lineno - 1]
            assert "_HERE.parents[2]" in line, f"{path.name}:{node.lineno} indexes parents directly"
    assert "parents[2] / " not in path.read_text()
    assert "except IndexError" not in path.read_text()
