# Adapted from CleanRL (https://github.com/vwxyzjn/cleanrl)
# Original: cleanrl/dqn_atari.py
# License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
#
# Modifications:
#   - Added --buffer-type flag (c / python / uniform)
#   - Added --output flag for JSON result logging
#   - Wired reservoir CFastPERBuffer and PyFastPERBuffer as PER replay options
"""DQN for Atari with pluggable replay buffer.

Usage:
    python -m benchmarks.atari_dqn --env-id BreakoutNoFrameskip-v4 --buffer-type c --seed 1
    python -m benchmarks.atari_dqn --env-id PongNoFrameskip-v4 --buffer-type uniform --output results/pong.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    RecordEpisodeStatistics,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DQN Atari (reservoir benchmark)")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-network-frequency", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.01)
    parser.add_argument("--exploration-fraction", type=float, default=0.10)
    parser.add_argument("--learning-starts", type=int, default=80_000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument(
        "--buffer-type",
        type=str,
        default="uniform",
        choices=["c", "python", "uniform"],
        help="Replay buffer: c=CFastPERBuffer, python=PyFastPERBuffer, uniform=SB3 ReplayBuffer",
    )
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)
    parser.add_argument("--per-epsilon", type=float, default=1e-6)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write JSON result file")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env = RecordEpisodeStatistics(env)
    env = AtariPreprocessing(env)
    env = FrameStackObservation(env, 4)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# Replay buffer factory
# ---------------------------------------------------------------------------

def make_buffer(buffer_type: str, buffer_size: int, obs_shape: tuple,
                n_actions: int, alpha: float, beta: float, epsilon: float,
                device: str):
    if buffer_type == "uniform":
        from stable_baselines3.common.buffers import ReplayBuffer
        import gymnasium as _gym
        obs_space = _gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        act_space = _gym.spaces.Discrete(n_actions)
        rb = ReplayBuffer(
            buffer_size,
            obs_space,
            act_space,
            device=device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
        rb._is_per = False
        return rb
    elif buffer_type == "c":
        from reservoir.c_buffer import CFastPERBuffer
        rb = CFastPERBuffer(
            capacity=buffer_size,
            obs_shape=obs_shape,
            action_dim=1,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            device=device,
        )
        rb._is_per = True
        return rb
    else:  # python
        from reservoir.fast_buffer import FastPERBuffer
        rb = FastPERBuffer(
            capacity=buffer_size,
            obs_shape=obs_shape,
            action_dim=1,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            device=device,
        )
        rb._is_per = True
        return rb


# ---------------------------------------------------------------------------
# Linear epsilon schedule
# ---------------------------------------------------------------------------

def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args=None):
    if args is None:
        args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Buffer: {args.buffer_type} | Env: {args.env_id}")

    env = make_env(args.env_id, args.seed)
    obs_shape = env.observation_space.shape  # (4, 84, 84)
    n_actions = env.action_space.n

    q_network = QNetwork(n_actions).to(device)
    target_network = QNetwork(n_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    rb = make_buffer(
        args.buffer_type, args.buffer_size, obs_shape, n_actions,
        args.per_alpha, args.per_beta, args.per_epsilon, str(device),
    )

    episodic_returns: list[float] = []
    start_time = time.time()

    obs, _ = env.reset(seed=args.seed)
    obs = np.array(obs)

    for global_step in range(args.total_timesteps):
        # Epsilon-greedy action
        epsilon = linear_schedule(
            args.start_e, args.end_e,
            int(args.exploration_fraction * args.total_timesteps),
            global_step,
        )
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals = q_network(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device))
                action = int(q_vals.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.array(next_obs)
        done = terminated or truncated

        # Store transition
        if rb._is_per:
            rb.add(obs, action, float(reward), next_obs, done)
        else:
            rb.add(
                np.array([obs]), np.array([[action]]),
                np.array([reward]), np.array([next_obs]),
                np.array([done]), np.array([{}]),
            )

        if done:
            ep_return = info.get("episode", {}).get("r", None)
            if ep_return is not None:
                episodic_returns.append(float(ep_return))
                if len(episodic_returns) % 10 == 0:
                    mean100 = np.mean(episodic_returns[-100:])
                    print(f"step={global_step} episodes={len(episodic_returns)} mean100={mean100:.1f}")
            obs, _ = env.reset()
            obs = np.array(obs)
        else:
            obs = next_obs

        # Training
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            buf_size = rb.size if rb._is_per else rb.pos if not rb.full else rb.buffer_size

            if buf_size >= args.batch_size:
                if rb._is_per:
                    batch = rb.sample(args.batch_size)
                    states      = batch.states.float().to(device)
                    actions     = batch.actions.long().to(device)
                    rewards_t   = batch.rewards.float().to(device)
                    next_states = batch.next_states.float().to(device)
                    dones_t     = batch.dones.float().to(device)
                    is_weights  = batch.is_weights.float().to(device)
                    indices     = batch.indices
                else:
                    batch = rb.sample(args.batch_size)
                    states      = batch.observations.squeeze(1).float()
                    actions     = batch.actions.squeeze(1).long()
                    rewards_t   = batch.rewards.squeeze(1).float()
                    next_states = batch.next_observations.squeeze(1).float()
                    dones_t     = batch.dones.squeeze(1).float()
                    is_weights  = torch.ones(args.batch_size, device=device)
                    indices     = None

                with torch.no_grad():
                    target_max = target_network(next_states).max(dim=1).values
                    td_target = rewards_t + args.gamma * target_max * (1 - dones_t)

                current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                td_errors = (td_target - current_q).detach()
                loss = (is_weights * F.mse_loss(current_q, td_target, reduction="none")).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), 10)
                optimizer.step()

                # Update PER priorities
                if rb._is_per and indices is not None:
                    rb.update_priorities(indices, td_errors.abs().cpu().numpy())
                    rb.anneal_beta(global_step, args.total_timesteps)

        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    env.close()
    duration = time.time() - start_time

    # Write results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_mean = float(np.mean(episodic_returns[-100:])) if episodic_returns else 0.0
        with open(out_path, "w") as f:
            json.dump({
                "game": args.env_id,
                "buffer_type": args.buffer_type,
                "seed": args.seed,
                "total_steps": args.total_timesteps,
                "episode_rewards": episodic_returns,
                "final_mean_reward_100ep": final_mean,
                "duration_seconds": round(duration, 1),
            }, f, indent=2)
        print(f"Result written to {args.output}")

    return episodic_returns


if __name__ == "__main__":
    train()
