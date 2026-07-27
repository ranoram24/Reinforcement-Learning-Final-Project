"""Room 4 — algorithm shoot-out (STANDALONE, not used by the app).

Room 4 (Tokyo-Drift canyon) is a DETERMINISTIC single track with 9 discrete
velocity actions and a continuous state [X, Y, Vx, Vy].  We benchmark the
applicable algorithm families from the comparison table and report, for each,
a robustness "escape rate" = fraction of 12 slightly-perturbed-start rollouts
whose greedy policy reaches the finish, plus the lap length and total reward.

    • Tabular Q-Learning / SARSA / Dyna-Q — on a DISCRETISED state (20×20×3×3)
    • Tile-coding linear FA               — continuous [X,Y,Vx,Vy]
    • DQN / REINFORCE / A2C / PPO         — neural, continuous [X,Y,Vx,Vy]

Not applicable (reported as N/A, not faked):
    • SAC / DDPG / TD3 — continuous-ACTION algorithms; Room 4's actions are the
      9 discrete (Vx,Vy) — testing them needs a continuous-action re-cast (a
      design change to the room).
    • AlphaZero — needs a known enumerable model + MCTS (state is continuous).
    • RLHF      — LLM alignment; unrelated to control.

Run:  python experiments/room4_algo_bench.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import algorithms as A          # noqa: E402
import environments as E        # noqa: E402
import extra_algos as X         # noqa: E402

STEPS = 2000            # max steps / lap (a lap is ~1,256 steps)
SHAPE_COEF = 30.0       # follow-the-road shaping (dense reward helps every learner)
EVAL_N = 12             # perturbed-start rollouts for the robustness rate


def _cont(seed=0):
    """Continuous-state Room 4 (for FA / DQN / policy-gradient)."""
    return E.Room4Garage(max_steps=STEPS, shaping=True, shaping_coef=SHAPE_COEF)


class Room4Disc(E.Room4Garage):
    """Room 4 with a DISCRETISED observation (20×20 position × 3×3 velocity) so
    the tabular learners (Q / SARSA / Dyna-Q) can index a Q-table."""

    XB = YB = 20

    def _obs(self):
        xb = min(self.XB - 1, max(0, int(self.X / 10.0 * self.XB)))
        yb = min(self.YB - 1, max(0, int(self.Y / 10.0 * self.YB)))
        return (xb, yb, int(round(self.Vx)) + 1, int(round(self.Vy)) + 1)

    def encode(self, obs):
        return tuple(obs)


def _disc(seed=0):
    return Room4Disc(max_steps=STEPS, shaping=True, shaping_coef=SHAPE_COEF)


def evaluate(make_env, policy, n=EVAL_N):
    """Escape rate over n slightly-perturbed starts + avg lap steps + avg reward."""
    ok, steps, rew = 0, [], []
    for k in range(n):
        env = make_env(0)
        env.reset()
        rng = np.random.default_rng(500 + k)
        env.X = float(np.clip(env.X + rng.uniform(-0.25, 0.25), 0.0, 10.0))
        env.Y = float(np.clip(env.Y + rng.uniform(-0.25, 0.25), 0.0, 10.0))
        obs = env._obs()
        done, total, st = False, 0.0, 0
        while not done and st < env.max_steps:
            obs, r, done = env.step(policy.action(obs))
            total += r; st += 1
        rew.append(total)
        if env.is_success():
            ok += 1; steps.append(st)
    return 100.0 * ok / n, (np.mean(steps) if steps else float("nan")), float(np.mean(rew))


def bench():
    rows = []

    def record(name, make_env, policy, t0):
        rate, lap, ar = evaluate(make_env, policy)
        rows.append((name, rate, lap, ar, time.time() - t0))
        lap_s = "—" if np.isnan(lap) else f"{lap:.0f}"
        print(f"     → escape {rate:.1f}% · lap {lap_s} steps · avg reward {ar:.0f}", flush=True)

    print("1  Tabular Q-Learning (discretised)…", flush=True)
    t = time.time()
    r = A.QLearning(_disc(), alpha=0.3, gamma=0.99, epsilon=1.0, epsilon_k=0.0005,
                    epsilon_min=0.05, episodes=3000, max_steps=STEPS, optimistic_init=0.0,
                    seed=0).train(snapshots=1)
    record("Tabular Q-Learning", _disc, r["final_policy"], t)

    print("2  SARSA (discretised)…", flush=True)
    t = time.time()
    r = A.Sarsa(_disc(), alpha=0.3, gamma=0.99, epsilon=1.0, epsilon_k=0.0005,
                epsilon_min=0.05, episodes=3000, max_steps=STEPS, optimistic_init=0.0,
                seed=0).train(snapshots=1)
    record("SARSA (on-policy)", _disc, r["final_policy"], t)

    print("3  Dyna-Q (discretised, model-based)…", flush=True)
    t = time.time()
    pol = X.train_dyna_q(_disc, episodes=2500, gamma=0.99, eps_k=0.0006, planning=3, seed=0)
    record("Dyna-Q (model-based)", _disc, pol, t)

    print("4  Tile-coding linear FA (continuous)…", flush=True)
    t = time.time()
    r = A.LinearFAAgent(_cont(), alpha=0.3, gamma=0.99, epsilon=0.5, epsilon_k=0.0004,
                        epsilon_min=0.0, episodes=3000, n_tilings=8, n_bins=10,
                        optimistic_init=60.0, max_steps=STEPS, seed=0).train(snapshots=1)
    record("Tile-coding FA", _cont, r["final_policy"], t)

    print("5  DQN — MLP Q-network (continuous)…", flush=True)
    t = time.time()
    r = A.DQNAgent(_cont(), alpha=5e-4, gamma=0.99, epsilon=1.0, epsilon_k=0.002,
                   epsilon_min=0.05, episodes=1000, max_steps=STEPS, hidden=128, batch=128,
                   target_every=1000, train_freq=2, warmup=2000, seed=0).train(snapshots=1)
    record("DQN (MLP)", _cont, r["final_policy"], t)

    print("6  REINFORCE (continuous)…", flush=True)
    t = time.time()
    record("REINFORCE", _cont, X.train_reinforce(_cont, episodes=1200, gamma=0.99, seed=0), t)

    print("7  A2C (continuous)…", flush=True)
    t = time.time()
    record("A2C", _cont, X.train_a2c(_cont, episodes=1200, gamma=0.99, seed=0), t)

    print("8  PPO (continuous)…  [slowest]", flush=True)
    t = time.time()
    record("PPO", _cont, X.train_ppo(_cont, updates=200, gamma=0.99, seed=0), t)

    print(f"\n=== Room 4 algorithm comparison — escape rate over {EVAL_N} perturbed starts ===")
    for name, rate, lap, ar, el in sorted(rows, key=lambda r: (-r[1], r[2] if not np.isnan(r[2]) else 9e9)):
        lap_s = "—" if np.isnan(lap) else f"{lap:.0f}"
        print(f"  {name:24s} escape {rate:5.1f}%   lap {lap_s:>5s} steps   avg reward {ar:8.0f}   (train {el:.0f}s)")
    print("\n  N/A (not applicable to this task):")
    print("  SAC / DDPG / TD3   — continuous-action algos; Room 4 uses 9 discrete velocity actions")
    print("  AlphaZero          — needs a known enumerable model + MCTS (state is continuous)")
    print("  RLHF               — LLM alignment over token sequences; unrelated to control")


if __name__ == "__main__":
    bench()
