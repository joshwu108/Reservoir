"""benchmarks/modal/verl_replay_real.py

Integration check for ``reservoir.integrations.verl``: a short GRPO run of
verl's ``DataProto`` trainer on one T4 with ``ReservoirRayPPOTrainer``
replaying dead groups from a durable ``ReservoirReplay`` buffer, producing
an attestation log and manifest that the independent checker
(``python -m checker.verify``) accepts.

This is an integration check, not a benchmark: the model is a 135M
instruct model that cannot learn the task, and no training-quality claim is
made for replay (``docs/nonclaims.md``). What the run establishes is that
the adapter survives verl's real call path (Ray driver, vLLM rollout via
the agent loop, FSDP actor, the default ``rollout_log_probs`` column,
checkpoint saves), that dead groups were replaced (the run fails if none
were), that the buffer is bound to the trainer's checkpoints, and that the
resulting log verifies. With one GPU ``balance_batch`` has one partition,
so nothing is claimed about reordering.

Reward: 1.0 if the response text has an even number of characters, else
0.0, as in ``trl_replay_real.py``: close to a coin flip for a near-random
model, so most groups have variance (which fills the buffer) and about one
group in eight with four responses is dead (which triggers replay).

verl 0.9.1's default trainer is the TransferQueue-based V1 loop; this run
sets ``trainer.use_v1=false`` to use the ``DataProto`` trainer the adapter
targets. The T4 has no bf16 and no flash-attention: the rollout and the
FSDP mixed precision run in float16 (verl attaches a sharded grad scaler
for fp16), the HF model uses SDPA attention, and ``use_remove_padding`` is
off. Cost estimate: T4 at $0.59/h for about 15 minutes plus the image
build, under $0.25.

Usage
-----
    modal run benchmarks/modal/verl_replay_real.py                       # T4, 12 steps
    modal run benchmarks/modal/verl_replay_real.py --max-steps 6

Results (JSON plus the attestation log and manifest, verified with the
checker before writing) land in benchmarks/modal/results/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import modal

from benchmarks.modal.trl_replay_real import hf_cache, label_for as _unused_label_for, report, with_sources, write_results  # noqa: F401

VERL_VERSION = "0.9.1"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
RESULTS_DIR = Path(__file__).parent / "results"

# torch from the cu130 index first (Modal's T4 hosts run a CUDA 13 driver; the vLLM
# 0.24.0 wheel is a cu130 build), then verl with its vllm extra, which pins
# vllm==0.24.0, torch==2.11.0 and transformers==5.9.0 and brings Ray.
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0",
                 index_url="https://download.pytorch.org/whl/cu130")
    .pip_install(f"verl[vllm]=={VERL_VERSION}")
    # Ray actors are separate processes that do not inherit the driver's sys.path, so the
    # mounted package sources go on PYTHONPATH for every process in the container.
    .env({"HF_HOME": "/hf_cache", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/reservoir_src:/root"})
)
image = with_sources(base_image)

app = modal.App("reservoir-verl-replay-real")

REWARD_SOURCE = '''
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """1.0 when the response text has an even number of characters, else 0.0."""
    return 1.0 if len(solution_str) % 2 == 0 else 0.0
'''

TOPICS = [
    "rivers", "clocks", "bread", "the moon", "trains", "gardens", "libraries", "winter", "bicycles", "maps",
    "coffee", "mountains", "kites", "lanterns", "harbors", "violins", "deserts", "umbrellas", "bees", "glaciers",
    "pianos", "bridges", "candles", "forests", "windmills", "islands", "telescopes", "orchards", "canals", "tides",
    "lighthouses", "meadows",
]


def write_dataset(path: Path, copies: int = 2) -> int:
    """A small parquet in verl's RL dataset schema; returns the number of rows."""
    from datasets import Dataset

    rows = []
    for copy_index in range(copies):
        for i, topic in enumerate(TOPICS):
            rows.append({
                "data_source": "reservoir/even_length",
                "prompt": [{"role": "user", "content": f"Write one short sentence about {topic}."}],
                "ability": "style",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"index": copy_index * len(TOPICS) + i, "split": "train"},
            })
    Dataset.from_list(rows).to_parquet(str(path))
    return len(rows)


def overrides(*, train_parquet: str, reward_py: str, max_steps: int, seed: int, ckpt_dir: str,
              save_freq: int, n: int, train_batch_size: int, max_prompt_length: int, max_response_length: int,
              model_id: str) -> list[str]:
    """Hydra overrides for a one-T4 GRPO run of the DataProto trainer."""
    return [
        "trainer.use_v1=false",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={train_parquet}",
        f"data.val_files={train_parquet}",
        f"data.train_batch_size={train_batch_size}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.dataloader_num_workers=0",
        f"actor_rollout_ref.model.path={model_id}",
        "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={train_batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.dtype=float16",
        # Not in the YAML struct (FSDPEngineConfig.mixed_precision defaults to None, which the
        # engine reads as bf16), so it is added with "+".
        "+actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype:fp16,reduce_dtype:fp32,buffer_dtype:fp32}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={n}",
        "actor_rollout_ref.rollout.dtype=float16",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.3",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
        f"actor_rollout_ref.rollout.seed={seed}",
        f"reward.custom_reward_function.path={reward_py}",
        "reward.custom_reward_function.name=compute_score",
        "trainer.logger=[console]",
        "trainer.project_name=reservoir",
        "trainer.experiment_name=verl_replay",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        f"trainer.save_freq={save_freq}",
        f"trainer.default_local_dir={ckpt_dir}",
        f"trainer.total_training_steps={max_steps}",
        "trainer.total_epochs=1000",
        "trainer.balance_batch=True",
        "trainer.resume_mode=disable",
        "ray_kwargs.ray_init.num_cpus=8",
    ]


def run_grpo(
    *,
    max_steps: int = 12,
    seed: int = 42,
    n: int = 4,
    train_batch_size: int = 8,
    max_prompt_length: int = 96,
    max_response_length: int = 48,
    save_freq: int = 5,
    capacity: int = 256,
    half_life: int = 8,
    max_policy_age: int = 24,
    buffer_seed: int = 0,
    work_dir: str = "/tmp/reservoir_verl",
    model_id: str = MODEL_ID,
) -> dict:
    """Train for ``max_steps`` with replay and return the run record (log and manifest included)."""
    import ray
    import torch
    import verl
    import vllm
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    import verl.trainer
    from verl.trainer.main_ppo import get_ppo_ray_runtime_env

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    train_parquet = work / "train.parquet"
    n_rows = write_dataset(train_parquet)
    reward_py = work / "reservoir_verl_reward.py"
    reward_py.write_text(REWARD_SOURCE)
    paths = {
        "attest": str(work / "attest.jsonl"), "manifest": str(work / "manifest.jsonl"),
        "buffer": str(work / "buffer"), "ckpt": str(work / "ckpt"), "results": str(work / "results.json"),
    }
    for key in ("attest", "manifest"):
        Path(paths[key]).unlink(missing_ok=True)

    config_dir = str(Path(verl.trainer.__file__).parent / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides(
            train_parquet=str(train_parquet), reward_py=str(reward_py), max_steps=max_steps, seed=seed,
            ckpt_dir=paths["ckpt"], save_freq=save_freq, n=n, train_batch_size=train_batch_size,
            max_prompt_length=max_prompt_length, max_response_length=max_response_length, model_id=model_id,
        ))
    with open_dict(config.data):
        config.data.seed = seed

    replay_kwargs = dict(capacity=capacity, half_life=half_life, max_policy_age=max_policy_age, seed=buffer_seed,
                         attest=paths["attest"], manifest=paths["manifest"], directory=paths["buffer"],
                         source="even_length")
    runner_cls = make_task_runner()
    if not ray.is_initialized():
        runtime_env = OmegaConf.merge(get_ppo_ray_runtime_env(config), config.ray_kwargs.get("ray_init", {}).get("runtime_env", {}))
        ray.init(num_cpus=8, runtime_env=OmegaConf.to_container(runtime_env))
    started = time.time()
    record = ray.get(runner_cls.remote().run.remote(config, replay_kwargs, paths))
    wall_clock = time.time() - started
    ray.shutdown()

    record.update({
        "model": model_id,
        "dataset": f"synthetic even-length prompts ({n_rows} rows)",
        "config": {
            "max_steps": max_steps, "seed": seed, "n": n, "train_batch_size": train_batch_size,
            "max_prompt_length": max_prompt_length, "max_response_length": max_response_length,
            "save_freq": save_freq, "capacity": capacity, "half_life": half_life, "max_policy_age": max_policy_age,
            "buffer_seed": buffer_seed, "beta_is": record.pop("beta_is"), "source": "even_length",
            "trainer_class": record.pop("trainer_class"), "use_v1": record.pop("use_v1"),
            "rollout": record.pop("rollout"), "actor": record.pop("actor"),
        },
        "versions": {"verl": verl.__version__, "vllm": vllm.__version__, "torch": torch.__version__,
                     "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "wall_clock_seconds": wall_clock,
        "attestation": {
            **record.pop("attestation"),
            "text": Path(paths["attest"]).read_text(),
            "manifest_text": Path(paths["manifest"]).read_text(),
        },
    })
    return record


def make_task_runner():
    """The Ray actor that builds ``ReservoirRayPPOTrainer`` where verl's ``TaskRunner`` builds ``RayPPOTrainer``.

    The body mirrors ``verl.trainer.main_ppo_v0.TaskRunner.run`` (verl 0.9.1):
    the worker mapping, resource pool, tokenizer and datasets are verl's; the
    trainer class and the replay buffer are ours.
    """
    import ray
    from verl.trainer.main_ppo_v0 import BaseTaskRunner

    @ray.remote
    class ReservoirTaskRunner(BaseTaskRunner):
        def run(self, config, replay_kwargs: dict, paths: dict) -> dict:
            import os
            import socket

            from omegaconf import OmegaConf

            from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
            from verl.utils.config import omega_conf_to_dataclass, validate_config
            from verl.utils.dataset.rl_dataset import collate_fn
            from verl.workers.config import HFModelConfig

            from reservoir.integrations.verl import ReservoirRayPPOTrainer, ReservoirReplay

            print(f"ReservoirTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
            OmegaConf.resolve(config)
            actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.add_reward_model_resource_pool(config)
            self.add_teacher_model_resource_pool(config)
            self.add_ref_policy_worker(config, actor_rollout_cls)
            validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
            model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
            tokenizer, processor = model_config.tokenizer, model_config.processor
            resource_pool_manager = self.init_resource_pool_mgr(config)
            train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True,
                                              max_samples=config.data.get("train_max_samples", -1))
            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False,
                                            max_samples=config.data.get("val_max_samples", -1))
            train_sampler = create_rl_sampler(config.data, train_dataset)

            replay = ReservoirReplay(**replay_kwargs)
            steps: list[dict] = []
            original_mix = replay.mix

            def recording_mix(data, trainer):
                out = original_mix(data, trainer)
                point = replay.last_telemetry
                steps.append({
                    "global_steps": int(trainer.global_steps), "batch_rows": len(data), "buffer_size": replay.buffer.size,
                    "buffer_version": replay.buffer.current_version, "replaced_this_step": point.replaced_rows if point else None,
                    "dead_groups_this_step": point.dead_groups if point else None,
                    "log_ratio_mean_abs": point.log_ratio_mean_abs if point else None,
                    "rows_rebuilt": out is not data, **replay.stats,
                })
                return out

            replay.mix = recording_mix
            trainer = ReservoirRayPPOTrainer(
                config=config, tokenizer=tokenizer, processor=processor, role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls,
                train_dataset=train_dataset, val_dataset=val_dataset, collate_fn=collate_fn, train_sampler=train_sampler,
                replay_buffer=replay,
            )
            trainer.init_workers()
            trainer.fit()
            if replay.stats["replaced_rows"] == 0:
                raise RuntimeError(
                    f"the run replaced no rows (stats={replay.stats}); it establishes nothing about replay, "
                    "so no record is written. Run more steps or a reward with more dead groups"
                )
            checkpoints = list(replay.buffer.checkpoints())
            trainer_ckpts = sorted(p.name for p in Path(paths["ckpt"]).iterdir() if p.is_dir()) if Path(paths["ckpt"]).is_dir() else []
            log = replay.buffer.attestation_log
            actor_cfg = config.actor_rollout_ref.actor
            record = {
                "beta_is": replay.beta,
                "trainer_class": next(f"{c.__module__}.{c.__name__}" for c in type(trainer).__mro__
                                      if c.__module__.startswith("verl.")),
                "rollout": {"name": config.actor_rollout_ref.rollout.name, "dtype": config.actor_rollout_ref.rollout.dtype,
                            "calculate_log_probs": bool(config.actor_rollout_ref.rollout.calculate_log_probs)},
                "actor": {"strategy": actor_cfg.fsdp_config.strategy,
                          "mixed_precision": OmegaConf.to_container(actor_cfg.fsdp_config.mixed_precision)
                          if actor_cfg.fsdp_config.get("mixed_precision") is not None else None,
                          "use_kl_loss": bool(actor_cfg.use_kl_loss),
                          "loss_mode": str(actor_cfg.policy_loss.loss_mode)},
                "use_v1": bool(config.trainer.use_v1),
                "totals": dict(replay.stats),
                "final_buffer_size": replay.buffer.size,
                "final_buffer_version": replay.buffer.current_version,
                "n_rebases": replay.buffer.n_rebases,
                "buffer_checkpoints": checkpoints,
                "trainer_checkpoints": trainer_ckpts,
                "steps": steps,
                "attestation": {"records": len(log.records), "head_digest": log.head_digest},
            }
            replay.close()
            return record

    return ReservoirTaskRunner


@app.function(gpu="T4", image=image, volumes={"/hf_cache": hf_cache}, timeout=5400, cpu=4, memory=24576)
def run_grpo_remote(**kwargs) -> dict:
    sys.path.insert(0, "/reservoir_src")
    return run_grpo(**kwargs)


def label_for(results: dict) -> str:
    cfg = results["config"]
    model = results["model"].rsplit("/", 1)[-1].lower()
    return f"verl_replay_{model}_{cfg['max_steps']}steps_seed{cfg['seed']}"


@app.local_entrypoint()
def main(max_steps: int = 12, seed: int = 42):
    results = run_grpo_remote.remote(max_steps=max_steps, seed=seed)
    path = write_results(results, label_for(results), RESULTS_DIR)
    report(results, path)
    print(f"buffer checkpoints: {results['buffer_checkpoints']}  trainer checkpoints: {results['trainer_checkpoints']}")
    print(json.dumps(results["versions"]))
