"""benchmarks/modal/trl_replay_driver.py

The script a distributed launcher runs for the two-process tier
(``trl_replay_distributed.py``). Every rank calls ``run_grpo`` from the
tier-1 script, so the training path is the one the single-process run
uses; ``ReservoirGRPOTrainer`` attaches the ``ReservoirReplay`` to the
trainer's accelerator at construction, which makes rank 0 the owner of
the buffer, the attestation log and the manifest, and leaves every other
rank without a buffer. After training each rank's summary is gathered to
rank 0, which writes the run record (rank 0's record from ``run_grpo``
plus a ``distributed`` section) to ``--out`` as JSON, once.

Not a Modal entrypoint and not meant to be started by hand: it has to
run under ``accelerate launch`` or ``torchrun`` so that ``RANK``,
``LOCAL_RANK`` and ``WORLD_SIZE`` are set before ``ReservoirReplay`` is
built. See ``trl_replay_distributed.py`` for both launch commands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _add_import_paths() -> None:
    """``benchmarks`` next to the repo root (or /root) and the package sources."""
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    for entry in (str(root), str(src) if src.is_dir() else "/reservoir_src"):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("--out", type=Path, required=True, help="where rank 0 writes the run record")
    parser.add_argument("--attest", required=True, help="attestation log path (rank 0 writes it)")
    parser.add_argument("--manifest", required=True, help="manifest path (rank 0 writes it)")
    parser.add_argument("--output-dir", required=True, help="the trainer's output_dir")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="train on CPU (the local gloo run)")
    return parser.parse_args(argv)


def _rank_summary(record: dict, state) -> dict:
    """What this rank reports to rank 0 about its own part of the run."""
    import torch

    local = int(state.local_process_index)
    device = record.get("device") or record.get("config", {}).get("device")
    device_name = torch.cuda.get_device_name(local) if torch.cuda.is_available() else "cpu"
    return {
        "rank": int(state.process_index),
        "local_rank": local,
        "env_rank": os.environ.get("RANK"),
        "env_local_rank": os.environ.get("LOCAL_RANK"),
        "is_owner": bool(record["is_owner"]),
        "device": device,
        "device_name": device_name,
        "hook_calls": record["totals"]["hook_calls"],
        "logprob_forwards": record["totals"]["logprob_forwards"],
        "wall_clock_seconds": record["wall_clock_seconds"],
    }


def main(argv: list[str] | None = None) -> None:
    _add_import_paths()
    cli = _parse(argv)
    if os.environ.get("RANK") is None:
        raise SystemExit("RANK is not set: run this driver under accelerate launch or torchrun")

    from benchmarks.modal.trl_replay_real import run_grpo

    record = run_grpo(
        max_steps=cli.max_steps, seed=cli.seed, use_cpu=cli.cpu,
        attest_path=cli.attest, manifest_path=cli.manifest, output_dir=cli.output_dir,
    )

    # The trainer's Accelerator initialised the process group; PartialState is its singleton.
    import torch.distributed as dist
    from accelerate import PartialState
    from accelerate.utils import gather_object

    state = PartialState()
    summaries = sorted(gather_object([_rank_summary(record, state)]), key=lambda s: s["rank"])
    if not state.is_main_process:
        return
    if not record["is_owner"]:
        raise RuntimeError(f"rank {state.process_index} is the main process but does not own the buffer")
    hook_calls = {s["hook_calls"] for s in summaries}
    record["distributed"] = {
        "world_size": int(state.num_processes),
        "backend": dist.get_backend() if dist.is_initialized() else None,
        "distributed_type": str(state.distributed_type),
        "ranks": summaries,
        "hook_calls_agree": len(hook_calls) == 1,
    }
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(record))
    print(f"rank 0 wrote {cli.out} (world_size={state.num_processes}, backend={record['distributed']['backend']})",
          flush=True)


if __name__ == "__main__":
    main()
