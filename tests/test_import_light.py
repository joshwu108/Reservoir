"""Importing the package and using the rollout buffer must need no third-party package.

The classic transition buffers, their wrappers and the fine-tuning tools
need numpy and torch; they resolve lazily on first attribute access and
name the extra to install when a dependency is missing.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).parents[1]
BLOCKED = ("numpy", "torch", "gymnasium", "matplotlib", "transformers", "datasets", "accelerate", "trl")

PROBE = f"""
import json, sys
for name in {BLOCKED!r}:
    sys.modules[name] = None            # any import of it now raises ImportError
import reservoir
from reservoir import Rollout, RolloutBuffer, DurableRolloutBuffer, ExactPERBuffer
from reservoir.attest import AttestationLog
from reservoir_checker.verify import verify_chain
from reservoir_checker.transcript import build_transcript
buf = RolloutBuffer(capacity=4, attest=AttestationLog())
buf.add_group("p", 0, [Rollout([1, 2], [-0.1, -0.2], 1.0)], source="s")
buf.sample(2)
verified = verify_chain(buf.attestation_log.records)
star = {{}}
exec("from reservoir import *", star)
out = {{"ok": True, "version": reservoir.__version__, "star": sorted(k for k in star if not k.startswith("__")),
       "sampled": build_transcript(verified)["sampled_rows"], "loaded": sorted(m for m in sys.modules if m.split(".")[0] in {BLOCKED!r} and sys.modules[m] is not None)}}
for name in ("FastPERBuffer", "ForgettingMonitor", "backend", "DatasetBuffer"):
    try:
        getattr(reservoir, name)
        out[name] = "resolved"
    except ImportError as exc:
        out[name] = {{"type": type(exc).__name__, "message": str(exc)}}
print(json.dumps(out))
"""


def test_rollout_path_and_checker_need_no_third_party_package():
    proc = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["ok"] and out["sampled"] == 2 and out["loaded"] == []
    assert set(out["star"]) == {"ExactPERBuffer", "Rollout", "RolloutGroup", "RolloutBatch", "RolloutBuffer",
                                "DurableRolloutBuffer"}
    for name, extra in (("FastPERBuffer", "classic"), ("backend", "classic"), ("DatasetBuffer", "classic"),
                        ("ForgettingMonitor", "anchor")):
        assert out[name]["type"] == "ImportError"
        assert f"'{extra}' extra" in out[name]["message"] and f"reservoir-replay[{extra}]" in out[name]["message"]


def test_lazy_exports_resolve_when_dependencies_are_present():
    import reservoir

    assert reservoir.backend in ("c", "python")
    assert reservoir.FastPERBuffer is reservoir.FastPERBuffer
    assert reservoir.PyFastPERBuffer.__module__ == "reservoir.fast_buffer"
    assert reservoir.DatasetBuffer is not None
    assert "FastPERBuffer" in dir(reservoir) and "FastPERBuffer" not in reservoir.__all__
    try:
        reservoir.NoSuchName
    except AttributeError as exc:
        assert "NoSuchName" in str(exc)
    else:
        raise AssertionError("unknown attribute did not raise")


def test_genuine_import_errors_are_not_disguised(monkeypatch):
    import reservoir

    monkeypatch.setitem(reservoir._LAZY, "Broken", ("reservoir._does_not_exist", "x", "classic"))
    try:
        reservoir.Broken
    except ImportError as exc:
        assert "extra" not in str(exc)
    finally:
        reservoir.__dict__.pop("Broken", None)


def test_console_scripts_resolve():
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    assert set(scripts) == {"reservoir-verify", "reservoir-transcript", "reservoir-diff"}
    for target in scripts.values():
        module, func = target.split(":")
        assert callable(getattr(importlib.import_module(module), func))
        assert module.startswith("reservoir_checker.")
