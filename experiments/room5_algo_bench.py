"""Room 5 — function-approximator shoot-out (STANDALONE, not used by the app).

The live app solves Room 5 with **tabular Q-Learning over a discretised look-ahead
sensor**, which tops out around a low escape rate now that waves spawn up to 8
obstacles at once.  This script benchmarks a few *approximation functions* on the
exact same environment to see which lifts the clean-escape rate, per the RL
algorithm families in the comparison table (value-based, discrete actions,
model-free, partial observation):

    1. Tabular Q-Learning        — the current baseline (discrete 3-lane sensor)
    2. Tile-coding linear FA      — Q(s,a)=wₐ·φ(s) over a 5-D CONTINUOUS sensor
    3. DQN (neural-net FA)        — an MLP Q-network over a richer 14-D sensor

Each is trained, then its greedy policy is evaluated on the SAME 40 random rooms
(identical seeds → identical obstacle streams), scored by clean-escape rate
(a full run with 0 collisions) and average reward.

Run:  python experiments/room5_algo_bench.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import algorithms as A          # noqa: E402
import environments as E        # noqa: E402
from algorithms import rollout  # noqa: E402

VISION, SPAWN, VMAX, STEPS = 3.0, 15, 3.0, 500
EVAL_ROOMS = 40


# --------------------------------------------------------------------------- #
# Continuous-observation wrappers (same dynamics, richer sensor for the FA's)
# --------------------------------------------------------------------------- #
class Room5Compact(E.Room5SpaceEscape):
    """5-D continuous sensor for tile coding:  [X, Vx, dyL, dyC, dyR]
    (dy* = distance up to the nearest obstacle in the left/centre/right lane,
    = vision when the lane is clear)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.state_low = np.array([0.0, -self.v_max, 0.0, 0.0, 0.0])
        self.state_high = np.array([10.0, self.v_max, self.vision, self.vision, self.vision])

    def _lane_dy(self, lo, hi):
        best = self.vision
        for o in self.obstacles:
            if lo <= (o["x"] - self.X) < hi:
                dy = o["y"] - self.AGENT_Y
                if 0.0 <= dy <= self.vision:
                    best = min(best, dy)
        return best

    def _obs(self):
        return np.array([self.X, self.Vx, self._lane_dy(-1.5, -0.5),
                         self._lane_dy(-0.5, 0.5), self._lane_dy(0.5, 1.5)], dtype=float)

    def encode(self, obs):
        return obs


class Room5Rich(E.Room5SpaceEscape):
    """14-D continuous sensor for DQN: agent (X, Vx, shots, ready) + for each of
    5 relative-x sectors the nearest obstacle's (distance, is-asteroid)."""

    SECTORS = [(-2.5, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 2.5)]

    def __init__(self, **kw):
        super().__init__(**kw)
        low = [0.0, -self.v_max, 0.0, 0.0]
        high = [10.0, self.v_max, float(self.N_SHOTS), 1.0]
        for _ in self.SECTORS:
            low += [0.0, 0.0]
            high += [self.vision, 1.0]
        self.state_low = np.array(low)
        self.state_high = np.array(high)

    def _obs(self):
        feats = [self.X, self.Vx, float(self.shots), 1.0 if self._ready() else 0.0]
        for lo, hi in self.SECTORS:
            bdy, ast = self.vision, 0.0
            for o in self.obstacles:
                if lo <= (o["x"] - self.X) < hi:
                    dy = o["y"] - self.AGENT_Y
                    if 0.0 <= dy <= self.vision and dy < bdy:
                        bdy, ast = dy, (1.0 if o["type"] == "asteroid" else 0.0)
            feats += [bdy, ast]
        return np.array(feats, dtype=float)

    def encode(self, obs):
        return obs


# --------------------------------------------------------------------------- #
def evaluate(cls, policy, n=EVAL_ROOMS):
    """Greedy escape rate (survived S sec with health left) + avg reward on n
    identical random rooms."""
    escaped, rew = 0, []
    for k in range(n):
        env = cls(vision=VISION, spawn_every=SPAWN, v_max=VMAX, max_steps=STEPS, seed=7000 + k)
        r = rollout(env, policy, max_steps=STEPS)
        rew.append(r["reward"])
        escaped += int(env.is_success())
    return 100.0 * escaped / n, float(np.mean(rew))


def _rich(seed):
    return Room5Rich(vision=VISION, spawn_every=SPAWN, v_max=VMAX, max_steps=STEPS, seed=seed)


def _disc(seed):
    return E.Room5SpaceEscape(vision=VISION, spawn_every=SPAWN, v_max=VMAX, max_steps=STEPS, seed=seed)


def bench():
    import extra_algos as X
    rows = []

    def record(name, cls, policy, t0):
        cr, ar = evaluate(cls, policy)
        rows.append((name, cr, ar, time.time() - t0))
        print(f"     → escape {cr:.1f}% · avg reward {ar:.1f}", flush=True)

    print("1  Tabular Q-Learning (discrete sensor)…", flush=True)
    t = time.time()
    r = A.QLearning(_disc(0), alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_k=0.0003,
                    epsilon_min=0.05, episodes=5000, max_steps=STEPS, optimistic_init=0.0,
                    seed=0).train(snapshots=1)
    record("Tabular Q-Learning", E.Room5SpaceEscape, r["final_policy"], t)

    print("2  SARSA — on-policy tabular (discrete sensor)…", flush=True)
    t = time.time()
    r = A.Sarsa(_disc(0), alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_k=0.0003,
                epsilon_min=0.05, episodes=5000, max_steps=STEPS, optimistic_init=0.0,
                seed=0).train(snapshots=1)
    record("SARSA (on-policy)", E.Room5SpaceEscape, r["final_policy"], t)

    print("3  Dyna-Q — model-based tabular (discrete sensor)…", flush=True)
    t = time.time()
    pol = X.train_dyna_q(_disc, episodes=5000, planning=5, seed=0)
    record("Dyna-Q (model-based)", E.Room5SpaceEscape, pol, t)

    print("4  Tile-coding linear FA (5-D continuous)…", flush=True)
    t = time.time()
    r = A.LinearFAAgent(Room5Compact(vision=VISION, spawn_every=SPAWN, v_max=VMAX,
                        max_steps=STEPS, seed=0), alpha=0.3, gamma=0.95, epsilon=1.0,
                        epsilon_k=0.0004, epsilon_min=0.05, episodes=5000, n_tilings=8,
                        n_bins=6, optimistic_init=0.0, max_steps=STEPS, seed=0).train(snapshots=1)
    record("Tile-coding FA", Room5Compact, r["final_policy"], t)

    print("5  DQN — MLP Q-network (rich 14-D)…", flush=True)
    t = time.time()
    r = A.DQNAgent(_rich(0), alpha=5e-4, gamma=0.97, epsilon=1.0, epsilon_k=0.0015,
                   epsilon_min=0.05, episodes=4000, max_steps=STEPS, hidden=128, batch=128,
                   target_every=1000, train_freq=2, warmup=2000, seed=0).train(snapshots=1)
    record("DQN (14-D MLP)", Room5Rich, r["final_policy"], t)

    print("6  REINFORCE — MC policy gradient (rich 14-D)…", flush=True)
    t = time.time()
    record("REINFORCE", Room5Rich, X.train_reinforce(_rich, episodes=2500, seed=0), t)

    print("7  A2C — advantage actor-critic (rich 14-D)…", flush=True)
    t = time.time()
    record("A2C", Room5Rich, X.train_a2c(_rich, episodes=2500, seed=0), t)

    print("8  PPO — clipped actor-critic + GAE (rich 14-D)…  [slowest]", flush=True)
    t = time.time()
    record("PPO", Room5Rich, X.train_ppo(_rich, updates=250, seed=0), t)

    print(f"\n=== Room 5 algorithm comparison — escape rate on {EVAL_ROOMS} random rooms ===")
    for name, cr, ar, el in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:24s} escape {cr:5.1f}%   avg reward {ar:8.1f}   (train {el:.0f}s)")
    print("\n  N/A (not applicable to this task):")
    print("  SAC / DDPG / TD3   — continuous-action algorithms; Room 5 actions are discrete")
    print("  AlphaZero          — needs a known enumerable model + MCTS (state is continuous)")
    print("  RLHF               — LLM alignment over token sequences; unrelated to control")


if __name__ == "__main__":
    bench()
