"""Unit tests for the environment dynamics and the learners.

Run with either:
    pytest -q
    python tests/test_core.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import algorithms as A          # noqa: E402
import environments as E        # noqa: E402


# --------------------------------------------------------------------------- #
# Shared grid dynamics
# --------------------------------------------------------------------------- #
def test_slip_distribution_sums_to_one():
    for env in (E.Room1FrozenArchive(), E.Room2DarkTemple()):
        for s in env.states():
            if env.is_terminal(s):
                continue
            for a in env.actions:
                total = sum(p for p, _ in env.outcomes(s, a))
                assert abs(total - 1.0) < 1e-9


def test_slippery_tiles_are_stochastic_others_not():
    env = E.Room1FrozenArchive()
    ice = next(iter(env.slippery))
    assert len(env.outcomes(ice, E.UP)) > 1             # ice → several outcomes
    assert len(env.outcomes((0, 0), E.UP)) == 1         # normal → deterministic


def test_terminals_absorbing_and_rewarded():
    env = E.Room2DarkTemple()
    r, done = env.reward_done(env.goal)
    assert done and r == E.GOAL_REWARD
    for pit in env.traps:
        r, done = env.reward_done(pit)
        assert done and r == E.TRAP_REWARD
    # terminals never appear as decision states in the DP model
    model = E.Room1FrozenArchive().build_model()
    assert (9, 9) not in model


def test_wall_blocks_movement():
    env = E.Room1FrozenArchive()
    wx, wy = next(iter(env.walls))              # step towards a wall from its left
    outs = dict((s2, p) for p, s2 in env.outcomes((wx - 1, wy), E.RIGHT))
    assert (wx, wy) not in outs                 # cannot enter the wall


# --------------------------------------------------------------------------- #
# Room 1 — Value Iteration
# --------------------------------------------------------------------------- #
def test_value_iteration_converges_and_solves():
    env = E.Room1FrozenArchive()
    res = A.ValueIteration(env, gamma=0.99, theta=1e-4).run()
    assert res["deltas"][-1] < 1e-4                       # converged
    assert res["V"][(0, 0)] > 0                            # start is valuable
    roll = A.rollout(env, res["final_policy"], max_steps=200)
    assert roll["success"]                                 # optimal policy escapes


# --------------------------------------------------------------------------- #
# Room 2 / Room 3 — TD control
# --------------------------------------------------------------------------- #
def test_sarsa_learns_to_escape():
    env = E.Room2DarkTemple(seed=0)
    res = A.Sarsa(env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                  episodes=1500, max_steps=300, seed=0).train(snapshots=3)
    # robust to slip stochasticity: the learned policy escapes the large majority of runs
    ev = E.Room2DarkTemple(seed=1)
    wins = sum(A.rollout(ev, res["final_policy"], max_steps=200)["success"] for _ in range(20))
    assert wins >= 18


def test_cliff_resets_not_terminal_and_qlearning_hugs_cliff():
    env = E.Room3CloningLab()
    env.reset()
    env._state = (1, 1)
    s2, r, done = env.step(E.DOWN)                         # (1,1) -> (1,0) is cliff
    assert s2 == env.start and r == E.TRAP_REWARD and not done
    env._state = (9, 1)
    s2, r, done = env.step(E.DOWN)                         # (9,1) -> (9,0) is the exit
    assert s2 == env.goal and r == E.GOAL_REWARD and done
    # learned greedy path avoids the cliff and reaches the goal
    res = A.QLearning(env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999,
                      episodes=800, max_steps=200, seed=0).train(snapshots=2)
    roll = A.rollout(env, res["final_policy"], max_steps=100)
    assert roll["success"]
    path = {f["agent"] for f in roll["frames"]}
    assert path.isdisjoint(env.cliff)                      # never steps on the cliff


# --------------------------------------------------------------------------- #
# Room 4 — continuous physics + tile coding
# --------------------------------------------------------------------------- #
def test_room4_collision_exit_and_soft_walls():
    env = E.Room4Garage(shaping=False)
    assert env._in_obstacle(3.5, 3.0)                      # inside a parked car
    env.reset(); env.X, env.Y, env.Vx, env.Vy = 3.4, 3.0, 3.0, 0.0
    _, r, done = env.step(4)                               # action 4 == (0,0) accel
    assert done and r == -100.0                            # drifts into the car → crash
    env.reset(); env.X, env.Y = 8.6, 8.6                   # already in the exit corner
    _, r, done = env.step(4)                               # (0,0) accel: stays put
    assert done and r == 100.0 and env.is_success()        # exit region reached
    # soft wall clamps instead of killing
    env.reset(); env.X, env.Vx = 9.98, 3.0
    _, r, done = env.step(5)
    assert not done and env.X <= 10.0 and env.Vx == 0.0


def test_room4_wavefront_reachable_from_start():
    env = E.Room4Garage()
    assert env._phi(1.0, 1.0) > env._phi(3.5, 3.0)         # start better than inside a car
    assert env._phi(9.0, 9.0) > env._phi(1.0, 1.0)         # nearer exit ⇒ higher potential


def test_tilecoder_shapes():
    tc = A.TileCoder(low=[0, 0, -3, -3], high=[10, 10, 3, 3], n_tilings=8, n_bins=8)
    feats = tc.features([1.0, 1.0, 0.0, 0.0])
    assert len(feats) == 8
    assert feats.min() >= 0 and feats.max() < tc.n_features


# --------------------------------------------------------------------------- #
# Room 5 — partial observability + dynamic obstacles
# --------------------------------------------------------------------------- #
def test_room5_sensor_and_collision():
    env = E.Room5SpaceEscape(vision=3.0, seed=0)
    env.reset()
    env.obstacles = []
    assert env._obs()[1] == -1.0                           # nothing in range → -1
    env.X = 5.0
    env.obstacles = [[5.0, 4.0]]                            # aligned, 2 m above, within vision
    d = env._obs()[1]
    assert abs(d - 2.0) < 1e-9
    env.obstacles = [[5.0, 9.0]]                            # aligned but beyond vision (7 m)
    assert env._obs()[1] == -1.0
    env.obstacles = [[9.0, 4.0]]                            # in range but not aligned
    assert env._obs()[1] == -1.0
    # collision within 0.5 m centre-to-centre → -1000 terminal
    env.reset(); env.X = 5.0; env.obstacles = [[5.0, 2.05]]
    _, r, done = env.step(1)                                # action 1 == 0 accel
    assert done and r == -1000.0


def test_room5_encode_discrete():
    env = E.Room5SpaceEscape(vision=3.0)
    a = env.encode(np.array([0.0, -1.0]))                   # clear ahead
    b = env.encode(np.array([0.0, 0.1]))                    # very close obstacle
    assert a[1] == 0 and b[1] >= 1 and isinstance(a, tuple)


# --------------------------------------------------------------------------- #
def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
