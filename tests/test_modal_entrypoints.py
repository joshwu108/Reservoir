"""The Modal benchmark scripts must resolve the checkout root the same way as entrypoints.

Modal copies the entrypoint file to ``/root/<script>.py`` inside the
container and imports it there; ``Path(__file__).parents[2]`` of that path
does not exist. ``trl_replay_real.repo_root`` resolves to the checkout
locally and to ``/root`` in the container, and every script must go through
it rather than index the parents of its own path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("modal")

from benchmarks.modal import trl_replay_real  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# trl_replay_driver.py is not a Modal entrypoint: accelerate runs it from its mounted
# package path (/root/benchmarks/modal/), where its own parents exist.
SCRIPTS = sorted(p for p in (REPO / "benchmarks" / "modal").glob("*.py")
                 if p.name not in ("__init__.py", "trl_replay_real.py", "trl_replay_driver.py"))


def test_repo_root_is_the_checkout_locally_and_root_in_the_container(monkeypatch):
    assert trl_replay_real.repo_root() == REPO
    monkeypatch.setattr(trl_replay_real, "__file__", "/root/trl_replay_real.py")
    assert trl_replay_real.repo_root() == Path("/root")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_scripts_do_not_index_parents_of_their_own_path(path: Path):
    text = path.read_text()
    assert "Path(__file__).resolve().parents[2]" not in text and "Path(__file__).parents[2]" not in text, (
        f"{path.name} indexes the parents of its own path; use repo_root() from trl_replay_real"
    )
