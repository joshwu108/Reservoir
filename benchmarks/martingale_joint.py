"""Joint CPU acceptance: one TRL GRPO run with Reservoir's replay and Martingale's recorder attached.

Run from the Reservoir checkout (Martingale >= 4416046, which has register_replayed with
weight and rescale):

    uv run --with <path to Martingale> python benchmarks/martingale_joint.py OUT [--mass-cap C] [--beta B]

The defaults (beta 0.4, mass cap 0.05) exercise non-unit importance weights; ``--beta 0
--mass-cap 0`` makes the group-mass cap bind on this tiny model, so rescales differ from 1.
Results are recorded in docs/staleness.md ("Replayed rows in a token record").

Checks that every row Reservoir replayed is registered in Martingale's record with the
buffer's exact importance weight and staleness rescale, that both records verify with
their own checkers, and that the two records join draw by draw.
"""
import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import torch
from datasets import Dataset
from trl import GRPOConfig

from martingale.diagnostics import decompose
from martingale.integrations.trl import MartingaleRecorder, build_trainer_class
from checker.verify_tokens import verify_export
from reservoir.integrations._trl_staleness import StalenessPolicy
from reservoir.integrations.trl import ReservoirReplay, ReservoirReplayMixin
from reservoir.integrations.trl import build_trainer_class as reservoir_trainer_class
from reservoir_checker.verify import verify_json_lines

# Appended, not prepended: the repo root holds Reservoir's ``checker`` alias, which must not shadow
# Martingale's ``checker`` package imported above.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from benchmarks.provenance import stamped  # noqa: E402

MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
STEPS = 12


def reward(completions, **kwargs):
    """Parity of the completion length: plenty of all-equal (dead) groups at num_generations=2."""
    return [float(len(c) % 2) for c in completions]


def main(out: Path, mass_cap: float = 0.05, beta: float = 0.4) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    dataset = Dataset.from_dict({"prompt": [f"Count to {i}:" for i in range(64)]})
    args = GRPOConfig(output_dir=str(out / "trainer"), per_device_train_batch_size=4, num_generations=2,
                      max_completion_length=8, max_steps=STEPS, learning_rate=1e-3, beta=0.0, logging_steps=1,
                      save_strategy="no", report_to="none", seed=0, use_cpu=True)
    replay = ReservoirReplay(capacity=64, half_life=4, max_policy_age=8, seed=0, beta=beta,
                             attest=str(out / "attest.jsonl"), manifest=str(out / "manifest.jsonl"),
                             staleness_policy=StalenessPolicy(mass_cap=mass_cap))
    flight = MartingaleRecorder(str(out / "ws"), tokenizer=MODEL.encode())

    class Trainer(ReservoirReplayMixin, build_trainer_class()):
        """Martingale records the fresh batch (inner), then Reservoir replaces dead rows (outer)."""

        def __init__(self, *a, replay_buffer, **kw):
            super().__init__(*a, **kw)
            self.replay_buffer = replay_buffer
            replay_buffer.attach(self.accelerator)
            self.add_callback(reservoir_trainer_class().Callback(replay_buffer))

    trainer = Trainer(model=MODEL, reward_funcs=[reward], args=args, train_dataset=dataset,
                      replay_buffer=replay, flight_recorder=flight, halt_on_error=False)
    trainer.train()
    replay.close()

    records = [json.loads(line) for line in (out / "attest.jsonl").read_text().splitlines() if line]
    verified = verify_json_lines((out / "attest.jsonl").read_text(), manifest=(out / "manifest.jsonl").read_text())
    by_draw = {}
    for rec in records:
        if rec["op"] == "sample":
            for k, s in enumerate(rec["samples"]):
                by_draw[f"{rec['op_counter']}/{k}"] = Fraction(int(s["is_weight_num"]), int(s["is_weight_den"]))
    scales = {}
    for rec in records:
        if rec["op"] == "telemetry" and rec.get("decisions"):
            for d in rec["decisions"]:
                if d["reason"] is None:
                    scales[f"{rec['sample_op_counter']}/{d['draw']}"] = Fraction(int(d["scale_num"]),
                                                                              int(d["scale_den"]))

    replayed = [s for s in flight.recorder.ledger.all_sequences() if s.replay is not None]
    joined, mismatched = 0, []
    for seq in replayed:
        prov = seq.replay
        weight = Fraction(prov.is_weight_num, prov.is_weight_den)
        rescale = Fraction(prov.rescale_num, prov.rescale_den)
        if by_draw.get(prov.draw_id) == weight and scales.get(prov.draw_id, Fraction(1)) == rescale:
            joined += 1
        else:
            mismatched.append(prov.draw_id)
    export = flight.recorder.export_for_checker(out / "export")
    checked = verify_export(export, expected_head=flight.head())
    report = decompose(flight.recorder.ledger)["replayed"]
    result = {
        "steps": STEPS, "mass_cap": mass_cap, "beta": beta,
        "reservoir": {"format": records[0]["format"], "min_entry_age": records[0].get("min_entry_age"),
                      "records": len(records), "verified": True, "manifest_matched": verified.content.manifest_matched,
                      "replaced_rows": replay.stats["replaced_rows"], "rescaled_rows": replay.stats["rescaled_rows"],
                      "declined_rows": replay.stats["declined_rows"]},
        "martingale": {"verify_ok": checked["ok"], "errors": checked["errors"],
                       "replayed_sequences": len(replayed), "unregistered_origins": report["n_unregistered_origins"],
                       "scored_replayed": report["n_scored_sequences"],
                       "rescale_not_one": sum(1 for s in replayed if (s.replay.rescale_num, s.replay.rescale_den) != (1, 1)),
                       "skipped_rows": flight.stats.get("skipped_rows"), "unmatched_rows": flight.stats.get("unmatched_rows")},
        "join": {"joined": joined, "mismatched": mismatched},
    }
    result = stamped(result)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--mass-cap", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.4)
    cli = parser.parse_args()
    print(json.dumps(main(cli.out, cli.mass_cap, cli.beta), indent=2, default=str))
