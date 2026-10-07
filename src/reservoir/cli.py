"""Command-line environment checks for Reservoir integrations."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import tempfile
import warnings
from pathlib import Path

from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.rollout import Rollout
from reservoir_checker.verify import verify_json_lines


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _compat(module: str, function: str) -> str:
    try:
        import importlib
        guard = getattr(importlib.import_module(module), function)
        with warnings.catch_warnings(record=True) as caught:
            guard()
        return "OK" if not caught else f"WARN: {caught[0].message}"
    except Exception as exc:
        return f"FAIL: {exc}"


def _at_least(version: str, minimum: tuple[int, ...]) -> bool:
    found = re.match(r"^(\d+)\.(\d+)", version)
    return found is not None and tuple(map(int, found.groups())) >= minimum


def doctor_rows() -> list[tuple[str, str, str]]:
    """Return (check, result, detail) rows without writing outside a temp dir."""
    from reservoir.integrations._trl_compat import PINNED_TRL_VERSION
    from reservoir.integrations._verl_compat import PINNED_VERL_VERSION

    rows = []
    for name, pin, module, guard in (
        ("trl", PINNED_TRL_VERSION, "reservoir.integrations._trl_compat", "require_trl"),
        ("verl", PINNED_VERL_VERSION, "reservoir.integrations._verl_compat", "require_verl"),
    ):
        installed = _version(name)
        status = "MISSING" if installed is None else ("OK" if installed == pin else "WARN")
        rows.append((f"{name} version", status, f"installed {installed or 'absent'}; pin {pin}"))
        if installed is None:
            rows.append((f"{name} extra / compat", "MISSING", f"install reservoir-replay[{name}]"))
        else:
            detail = _compat(module, guard)
            rows.append((f"{name} extra / compat", "FAIL" if detail.startswith("FAIL") else "OK", detail))

    torch_version = _version("torch")
    rows.append(("torch version", "OK" if torch_version and _at_least(torch_version, (2, 0)) else
                 ("WARN" if torch_version else "MISSING"),
                 f"installed {torch_version or 'absent'}; required >=2.0.0 for integrations"))
    for name in ("numpy", "transformers"):
        version = _version(name)
        rows.append((f"{name} present", "OK" if version else "MISSING", version or "absent"))
    if torch_version:
        try:
            import torch
            gpu = torch.cuda.is_available()
            rows.append(("GPU visible", "YES" if gpu else "NO", "CUDA available" if gpu else "CPU only"))
        except Exception as exc:
            rows.append(("GPU visible", "FAIL", str(exc)))
    else:
        rows.append(("GPU visible", "NO", "torch absent"))

    try:
        with tempfile.TemporaryDirectory(prefix="reservoir-doctor-") as folder:
            root = Path(folder)
            log = root / "attest.jsonl"
            manifest = root / "manifest.jsonl"
            settings = dict(capacity=4, half_life=2, max_policy_age=4,
                            attest=log, manifest=manifest)
            buffer = DurableRolloutBuffer(root / "buffer", **settings)
            buffer.add_group("doctor", 0, [Rollout(tokens=[1], logprobs=[-0.1], reward=1.0)])
            buffer.close()
            reopened = DurableRolloutBuffer(root / "buffer", **settings)
            ok = reopened.size == 1 and reopened.entry(reopened.live_positions()[0])[0].tokens == (1,)
            reopened.close()
            rows.append(("durable round trip", "OK" if ok else "FAIL", "temp directory writable; reopen restored row"))
            verify_json_lines(log.read_text(), manifest=manifest.read_text())
            rows.append(("independent checker", "OK", "attestation and manifest verified"))
    except Exception as exc:
        rows.append(("durable round trip / checker", "FAIL", str(exc)))
    rows.append(("trainer logging", "INFO", "<output_dir>/reservoir/attest.jsonl and manifest.jsonl"))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reservoir")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="check integrations, GPU, durable storage and verifier")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        rows = doctor_rows()
        width = max(len(label) for label, _, _ in rows)
        print(f"{'Check':<{width}}  Status   Detail")
        print(f"{'-' * width}  -------  ------")
        for label, status, detail in rows:
            print(f"{label:<{width}}  {status:<7}  {detail}")
        return 1 if any(status == "FAIL" for _, status, _ in rows) else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
