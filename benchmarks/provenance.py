"""
benchmarks/provenance.py — Where a result JSON came from.

Every result writer under ``benchmarks/`` stamps its report with
``provenance()``: the Reservoir version, the git commit of the checkout
(and which tracked files differed from it), the UTC date, the Python
version, the machine and the GPU name. A number without these cannot be
tied to the code that produced it.

The GPU name is the one the caller measured where the work ran (a Modal
container reports ``torch.cuda.get_device_name``); without one, the local
machine is asked through ``nvidia-smi``. ``None`` means no CUDA GPU was
seen: the CPU benchmarks record that explicitly rather than omitting it.
"""

from __future__ import annotations

import datetime
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
GIT_TIMEOUT_SECONDS = 10


def _git(*args: str) -> Optional[str]:
    if shutil.which("git") is None:
        return None
    try:
        done = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              timeout=GIT_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def git_commit() -> Optional[str]:
    """The checkout's ``HEAD`` commit, or None outside a git checkout (a Modal container)."""
    head = _git("rev-parse", "HEAD")
    return None if head is None else head.strip()


def git_changed_files() -> Optional[list[str]]:
    """Tracked paths that differ from ``HEAD`` (staged or not), sorted; None when git is unavailable."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status is None:
        return None
    return sorted(line[3:] for line in status.splitlines() if line.strip())


def local_gpu_name() -> Optional[str]:
    """The first CUDA GPU ``nvidia-smi`` reports on this machine, or None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    names = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    return names[0] if done.returncode == 0 and names else None


def gpu_of(record: dict) -> Optional[str]:
    """The GPU name a run record measured where it ran, in any of the shapes the writers use; None for CPU."""
    distributed = record.get("distributed") or {}
    candidates = [
        record.get("gpu_name"),
        (record.get("config") or {}).get("gpu_name"),
        (record.get("versions") or {}).get("gpu"),
        *((rank or {}).get("device_name") for rank in distributed.get("ranks") or []),
    ]
    return next((name for name in candidates if isinstance(name, str) and name and name != "cpu"), None)


def provenance(gpu: Optional[str] = None) -> dict:
    """The provenance block; ``gpu`` overrides the local probe when the work ran elsewhere."""
    from reservoir import __version__

    changed = git_changed_files()
    return {
        "reservoir_version": __version__,
        "commit": git_commit(),
        "changed_files": changed,
        "commit_dirty": None if changed is None else bool(changed),
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "gpu": gpu if gpu is not None else local_gpu_name(),
    }


def stamped(report: dict, gpu: Optional[str] = None) -> dict:
    """A copy of ``report`` with ``provenance`` as its first key; an existing block is replaced."""
    rest = {k: v for k, v in report.items() if k != "provenance"}
    return {"provenance": provenance(gpu), **rest}
