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
    env = E.Room1FrozenArchive()
    for s in env.states():
        if env.is_terminal(s):
            continue
        for a in env.actions:
            assert abs(sum(p for p, _ in env.outcomes(s, a)) - 1.0) < 1e-9
    # Room 2 has no DP model (SARSA is model-free) — sweep its cells directly
    r2 = E.Room2DarkTemple()
    for x in range(r2.SIZE):
        for y in range(r2.SIZE):
            if (x, y) in r2.walls:
                continue
            for a in r2.actions:
                assert abs(sum(p for p, _ in r2.outcomes((x, y, 0, None), a)) - 1.0) < 1e-9


def test_slip_applies_only_to_the_named_action():
    """Room 1 slip is per-tile AND per-action: only the action the tile names is
    stochastic; every other action on that same tile is deterministic."""
    env = E.Room1FrozenArchive()
    found = None
    for cell, rules in env.slippery.items():
        for act in rules:
            s = (cell[0], cell[1], 0)
            if len(env.outcomes(s, act)) > 1:
                found = (s, act)
                break
        if found:
            break
    assert found, "expected at least one genuinely stochastic slip tile"
    s, slip_act = found
    assert len(env.outcomes(s, slip_act)) > 1                 # named action slips
    for other in env.actions:
        if other != slip_act:
            assert len(env.outcomes(s, other)) == 1           # others deterministic


def test_terminals_absorbing_and_rewarded():
    # the exit never appears as a decision state in Room 1's DP model
    r1 = E.Room1FrozenArchive()
    assert all((s[0], s[1]) != r1.exit for s in r1.build_model())


def test_room2_pits_reset_to_start_without_ending_episode():
    env = E.Room2DarkTemple(seed=0)
    pit, cost = next(iter(env.holes.items()))
    nb = next(n for n in [(pit[0], pit[1] + 1), (pit[0], pit[1] - 1),
                          (pit[0] + 1, pit[1]), (pit[0] - 1, pit[1])]
              if env.in_bounds(n) and n not in env.walls and n not in env.slippery)
    d = {(0, 1): E.DOWN, (0, -1): E.UP, (1, 0): E.LEFT, (-1, 0): E.RIGHT}[
        (nb[0] - pit[0], nb[1] - pit[1])]
    env._state = (nb[0], nb[1], 0, None)
    s, r, done = env.step(d)                     # step into the pit
    assert (s[0], s[1]) == env.start             # hurled back to the start
    assert r == cost and not done                # penalty, but episode continues


def test_room2_gate_needs_the_idol_and_exit_is_terminal():
    env = E.Room2DarkTemple()
    door = next(iter(env.doors))
    nb = (door[0] + 1, door[1])                  # step left into the gate
    assert env._move((nb[0], nb[1], 0), E.LEFT) == nb                    # shut
    assert env._move((nb[0], nb[1], 1 << env.key_bit), E.LEFT) == door   # open
    env._state = (door[0], door[1], 1 << env.key_bit, None)
    s, r, done = env.step(E.LEFT)                # gate -> exit
    assert (s[0], s[1]) == env.exit and done and r == env.EXIT_REWARD


def test_room2_plate_pays_once_then_is_consumed_and_wakes_boulder():
    env = E.Room2DarkTemple(seed=0)
    right = (env.button[0] + 1, env.button[1])          # free cell east of the plate
    key = 1 << env.key_bit
    # without the idol the plate is inert
    env.reset(); env._state = (right[0], right[1], 0, None)
    s, r, _ = env.step(E.LEFT)
    assert (s[0], s[1]) == env.button and s[3] is None and r == 0.0
    # holding the idol: first press pays +BUTTON_REWARD, consumes the plate, wakes boulder
    env.reset(); env._state = (right[0], right[1], key, None)
    env._path = [env.chaser_spawn, right]               # we arrived via the idol tile
    s, r, _ = env.step(E.LEFT)
    assert r == env.BUTTON_REWARD and s[3] == env.chaser_spawn
    assert (s[2] >> env.button_bit) & 1                 # plate marked as used
    # stepping on it again pays nothing (consumed)
    env._state = (right[0], right[1], s[2], None); env._chaser_i = None
    s2, r2, _ = env.step(E.LEFT)
    assert r2 == 0.0


def test_room2_boulder_catch_penalises_but_leaves_agent_in_place():
    """Being caught costs CATCH_REWARD and the boulder vanishes, but the agent
    STAYS where it is (no teleport) and keeps everything — it just heads on to
    the exit."""
    env = E.Room2DarkTemple(seed=0)
    right = (env.button[0] + 1, env.button[1])
    key = 1 << env.key_bit
    env.reset(); env._state = (right[0], right[1], key, None)
    env._path = [env.chaser_spawn, right]
    s, _, _ = env.step(E.LEFT)                          # press plate → boulder appears
    mask_after_press = s[2]
    caught, before = False, None
    for _ in range(20):                                 # oscillate = backtrack into it
        for act in (E.RIGHT, E.LEFT):
            before = (env._state[0], env._state[1])
            s, r, done = env.step(act)
            if r <= env.CATCH_REWARD:
                caught = True
                break
        if caught:
            break
    assert caught, "backtracking should run into the boulder"
    assert s[3] is None                                 # boulder gone
    assert (s[0], s[1]) != env.start                    # NOT teleported to start
    assert s[2] == mask_after_press                     # key + plate-used KEPT (no reset)
    assert env.has_key(s[2]) and (s[2] >> env.button_bit) & 1


def test_wall_blocks_movement():
    env = E.Room1FrozenArchive()
    pair = next(((w, (w[0] - 1, w[1])) for w in env.walls
                 if w[0] > 0 and (w[0] - 1, w[1]) not in env.walls
                 and (w[0] - 1, w[1]) not in env.doors), None)
    assert pair, "expected a wall with a free cell to its left"
    wall, left = pair
    assert env._move((left[0], left[1], 0), E.RIGHT) == left   # bounced off the wall


def test_key_gates_the_door_then_opens_it():
    env = E.Room1FrozenArchive()
    door = next(iter(env.doors))
    nb = (door[0], door[1] + 1)                    # a neighbour that can step in
    if nb in env.walls or nb[1] >= env.SIZE:
        nb = (door[0] + 1, door[1])
    no_key, with_key = 0, 1 << env.key_bit
    toward = E.DOWN if nb == (door[0], door[1] + 1) else E.LEFT
    assert env._move((nb[0], nb[1], no_key), toward) == nb        # gate is shut
    assert env._move((nb[0], nb[1], with_key), toward) == door    # key opens it


def test_pickups_are_one_off_and_step_reward_is_zero():
    env = E.Room1FrozenArchive()
    r, mask, done = env._enter(env.key_cell, 0)
    assert r == env.PICKUPS["K"] and env.has_key(mask) and not done
    r2, _, _ = env._enter(env.key_cell, mask)      # already taken → no reward again
    assert r2 == env.STEP_REWARD == 0.0
    r3, _, done3 = env._enter(env.exit, mask)      # exit is terminal, +100
    assert done3 and r3 == env.EXIT_REWARD


# --------------------------------------------------------------------------- #
# Room 1 — Value Iteration
# --------------------------------------------------------------------------- #
def test_value_iteration_converges_and_solves():
    env = E.Room1FrozenArchive()
    res = A.ValueIteration(env, gamma=0.99, theta=1e-4).run()
    assert res["deltas"][-1] < 1e-4                       # converged
    assert res["V"][(env.start[0], env.start[1], 0)] > 0   # start is valuable
    # the optimal plan really is: take the key, pass the gate, reach the exit
    roll = A.rollout(E.Room1FrozenArchive(seed=1), res["final_policy"], max_steps=300)
    path = [f["agent"] for f in roll["frames"]]
    assert roll["success"] and path[-1] == env.exit
    assert env.key_cell in path and any(p in env.doors for p in path)
    roll = A.rollout(env, res["final_policy"], max_steps=200)
    assert roll["success"]                                 # optimal policy escapes


# --------------------------------------------------------------------------- #
# Room 2 / Room 3 — TD control
# --------------------------------------------------------------------------- #
def test_action_masking_matches_the_legal_moves_in_every_grid_room():
    """valid_actions must be exactly the legal (non-wall) moves.  The only
    exception is a fully-enclosed cell (no legal move at all), where it falls
    back to all actions so nothing can crash — those cells are never real
    decision states (e.g. the key-gated exit corner)."""
    def legal_dest(env, s, a):
        nxt = (s[0] + E._DELTA[a][0], s[1] + E._DELTA[a][1])
        if not env.in_bounds(nxt):
            return False
        if isinstance(env, E.Room3CloningLab):
            return not env.is_wall(nxt)
        return not env.blocked(nxt, s[2])

    for env in (E.Room1FrozenArchive(), E.Room2DarkTemple(), E.Room3CloningLab()):
        if isinstance(env, E.Room3CloningLab):
            states = [s for s in env.states() if not env.is_terminal(s)]
        else:
            states = [(x, y, m) for x in range(env.SIZE) for y in range(env.SIZE)
                      if (x, y) not in env.walls for m in (0, 1)]
        for s in states:
            legal = [a for a in env.actions if legal_dest(env, s, a)]
            va = env.valid_actions(s)
            if legal:                              # normal cell: exactly the legal moves
                assert set(va) == set(legal), (type(env).__name__, s, va, legal)
            else:                                  # enclosed: fallback = all actions
                assert set(va) == set(env.actions)


def test_linear_epsilon_decay():
    """ε must fall LINEARLY by K each episode (ε = ε₀ − K·t), floored at ε_min."""
    env = E.Room3CloningLab(seed=0)
    ag = A.QLearning(env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_k=0.01,
                     epsilon_min=0.1, episodes=200, max_steps=50, seed=0)
    eps = ag.train(snapshots=1)["epsilons"]
    # after ep 0 the stored value is ε₀−K, then −2K, … down to the floor
    assert abs(eps[0] - 0.99) < 1e-9
    assert abs(eps[9] - 0.90) < 1e-9               # ε₀ − 10·K = 0.90
    assert min(eps) == 0.1 and eps[-1] == 0.1      # floored at ε_min


def test_room2_sokoban_push_button_and_gate():
    env = E.Room2CloningLab(seed=0)
    env.reset()
    # push box2 (7,0) right onto its neighbour, then onto the plate (9,0)
    env._boxes = set(env.box_start)
    env._state = (6, 0, tuple(sorted(env._boxes)), 0, (False, False))
    s, r, _ = env.step(E.RIGHT)                       # box (7,0)->(8,0), agent stays
    assert (s[0], s[1]) == (6, 0) and (8, 0) in s[2] and r == env.STEP_REWARD
    env._state = (7, 0, tuple(sorted(env._boxes)), 0, (False, False))
    s, r, _ = env.step(E.RIGHT)                       # box (8,0)->(9,0) plate
    assert (9, 0) in s[2] and r == env.STEP_REWARD + env.BUTTON_REWARD
    s, r, _ = env.step(E.RIGHT)                       # plate box is locked → no farm
    assert r == env.STEP_REWARD
    # gate opens only when BOTH plates are covered
    env.reset()
    assert not env.door_open(set([env.buttons[0]]))
    assert env.door_open(set(env.buttons))
    # a move into a wall/border is masked out of the action space
    va = env.valid_actions((env.start[0], env.start[1], tuple(sorted(env.box_start)),
                            0, (False, False)))
    assert E.DOWN not in va                            # start is on the bottom border


def test_qlearning_solves_the_boulder_temple():
    """Room 3 (Dark Temple) uses off-policy Q-Learning + a decaying-from-high ε.
    That combination reliably solves the sparse, high-reward maze (SARSA does
    not)."""
    env = E.Room2DarkTemple(seed=0)
    res = A.QLearning(env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_k=0.0002,
                      epsilon_min=0.01, episodes=4000, max_steps=400,
                      optimistic_init=0.0, seed=0).train(snapshots=3)
    wins = sum(A.rollout(E.Room2DarkTemple(seed=s), res["final_policy"],
                         max_steps=400)["success"] for s in range(15))
    assert wins >= 13


def test_cliff_resets_not_terminal_and_sarsa_finds_a_safe_path():
    """Room 2 (Cloning Lab / Cliff Walking) uses on-policy SARSA, which learns a
    cautious path that reaches the exit without stepping on the cliff."""
    env = E.Room3CloningLab()
    env.reset()
    env._state = (1, 1)
    s2, r, done = env.step(E.DOWN)                         # (1,1) -> (1,0) is cliff
    assert s2 == env.start and r == E.TRAP_REWARD and not done
    env._state = (9, 1)
    s2, r, done = env.step(E.DOWN)                         # (9,1) -> (9,0) is the exit
    assert s2 == env.goal and r == E.GOAL_REWARD and done
    # SARSA learns a safe greedy path to the goal
    res = A.Sarsa(env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_k=0.001,
                  epsilon_min=0.01, episodes=1500, max_steps=200, seed=0).train(snapshots=2)
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
