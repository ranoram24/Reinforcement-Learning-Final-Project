"""Environments for the RL Escape Room — "Hezki the Dog vs. Men in Black".

Coordinate convention (grid rooms)
----------------------------------
state = (x, y),  x = column (0..size-1),  y = row (0..size-1).
Movement:  RIGHT = x+1,  LEFT = x-1,  UP = y+1,  DOWN = y-1.
Rendering (see utils.py) draws y=0 at the BOTTOM so the geometry reads naturally
(this matters for Cliff Walking: start bottom-left, goal bottom-right).

Terminal states are ABSORBING everywhere (the spec's critical rule): once entered
the episode ends and no further movement / sliding is computed.  The single
exception is Room 3's cliff, which — per the canonical Cliff-Walking task — is a
penalty-and-reset, not a true terminal (see Room3CloningLab).
"""
from __future__ import annotations

from collections import deque

import numpy as np

# --------------------------------------------------------------------------- #
# Shared 4-directional grid actions
# --------------------------------------------------------------------------- #
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
GRID_ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = {UP: "Up", DOWN: "Down", LEFT: "Left", RIGHT: "Right"}
ACTION_ARROWS = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}
_DELTA = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}

# Shared grid reward structure
STEP_REWARD = -1.0
GOAL_REWARD = 100.0
TRAP_REWARD = -100.0


def _rot_ccw(d):   # +90° (left relative to the action)
    return (-d[1], d[0])


def _rot_cw(d):    # -90° (right relative to the action)
    return (d[1], -d[0])


def _rot_180(d):   # backwards
    return (-d[0], -d[1])


# --------------------------------------------------------------------------- #
# Grid base class (Rooms 1-3)
# --------------------------------------------------------------------------- #
class GridWorld:
    """A 10x10 grid MDP with walls, slippery tiles and terminal traps.

    Provides BOTH a model-free interface (`reset`/`step`, used by SARSA/Q-Learning
    which must not peek at the dynamics) and, via `build_model`, the full transition
    model P(s'|s,a) used by Value Iteration in Room 1.
    """

    NAME = "GridWorld"

    def __init__(self, size=10, start=(0, 0), goal=(9, 9),
                 walls=(), slippery=(), traps=(), seed=None):
        self.size = size
        self.start = start
        self.goal = goal
        self.walls = set(walls)
        self.slippery = set(slippery)
        self.traps = set(traps)
        self.terminals = {goal} | self.traps
        self.actions = GRID_ACTIONS
        self.n_actions = 4
        self.rng = np.random.default_rng(seed)     # slips are reproducible
        self._state = start

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    # ---- geometry -------------------------------------------------------- #
    def in_bounds(self, c):
        return 0 <= c[0] < self.size and 0 <= c[1] < self.size

    def is_wall(self, c):
        return c in self.walls

    def is_terminal(self, c):
        return c in self.terminals

    def _apply(self, s, delta):
        """Deterministically apply a delta; stay put if it hits a wall/boundary."""
        nxt = (s[0] + delta[0], s[1] + delta[1])
        if not self.in_bounds(nxt) or self.is_wall(nxt):
            return s
        return nxt

    # ---- transition dynamics -------------------------------------------- #
    def outcomes(self, s, a):
        """List of (prob, s') for taking action `a` in state `s`.

        Slippery tile: 0.70 intended / 0.10 left / 0.10 right / 0.10 backwards.
        Any outcome blocked by a wall/boundary collapses onto `s` (stay).
        """
        delta = _DELTA[a]
        if s in self.slippery:
            candidates = [(delta, 0.70), (_rot_ccw(delta), 0.10),
                          (_rot_cw(delta), 0.10), (_rot_180(delta), 0.10)]
        else:
            candidates = [(delta, 1.0)]
        dist = {}
        for d, p in candidates:
            s2 = self._apply(s, d)
            dist[s2] = dist.get(s2, 0.0) + p
        return [(p, s2) for s2, p in dist.items()]

    def reward_done(self, s2):
        if s2 == self.goal:
            return GOAL_REWARD, True
        if s2 in self.traps:
            return TRAP_REWARD, True
        return STEP_REWARD, False

    # ---- model-free interface (SARSA / Q-Learning) ---------------------- #
    def reset(self):
        self._state = self.start
        return self._state

    def step(self, a):
        outs = self.outcomes(self._state, a)
        probs = [p for p, _ in outs]
        s2 = outs[int(self.rng.choice(len(outs), p=probs))][1]
        r, done = self.reward_done(s2)
        self._state = s2
        return s2, r, done

    # ---- helpers --------------------------------------------------------- #
    def states(self):
        return [(x, y) for x in range(self.size) for y in range(self.size)
                if (x, y) not in self.walls]

    def build_model(self):
        """Full model P[s][a] -> list of (prob, s', reward, done). For DP."""
        P = {}
        for s in self.states():
            if self.is_terminal(s):
                continue
            P[s] = {a: [(p, s2, *self.reward_done(s2)) for p, s2 in self.outcomes(s, a)]
                    for a in self.actions}
        return P

    def is_success(self):
        return self._state == self.goal

    def encode(self, s):
        """Grid states are already discrete/hashable — identity encoding."""
        return s

    def render_frame(self):
        return {"agent": self._state}

    def render_meta(self):
        """Static layout used by the HTML renderer."""
        return dict(size=self.size, start=self.start, goal=self.goal,
                    walls=self.walls, slippery=self.slippery, traps=self.traps,
                    cliff=set())


# --------------------------------------------------------------------------- #
# Room 1 — The Frozen Archive (Ice Age) — Dynamic Programming
# --------------------------------------------------------------------------- #
class Room1FrozenArchive(GridWorld):
    """LEVEL 1 (easiest): cross a frozen lake. No deadly traps — just a big patch
    of slippery ice in the middle and a couple of ice-boulders. The agent may go
    the safe way around or risk the slippery short-cut; DP weighs the stochastic
    slips exactly."""

    NAME = "Room 1 · The Frozen Archive"
    MOVIE = "Ice Age (2002)"
    ALGO = "Value Iteration (Dynamic Programming)"

    def __init__(self, seed=None):
        super().__init__(
            size=10, start=(0, 0), goal=(9, 9), seed=seed,
            # ice-boulders block a few cells; a 3x3+ frozen lake sits mid-board
            walls=[(6, 2), (2, 6), (7, 6)],
            slippery=[(3, 3), (4, 3), (5, 3), (3, 4), (4, 4), (5, 4),
                      (3, 5), (4, 5), (5, 5), (6, 5)],   # (9,9) is NOT slippery
            traps=[],
        )


# --------------------------------------------------------------------------- #
# Room 2 — The Dark Temple (Indiana Jones) — SARSA
# --------------------------------------------------------------------------- #
class Room2DarkTemple(GridWorld):
    """LEVEL 2 (moderate): a booby-trapped temple. A gauntlet of spike pits
    (instant death) with slippery mud beside them, plus stone walls shaping the
    corridors. SARSA must learn a *safe* path that keeps clear of the mud-next-to-
    pit cells."""

    NAME = "Room 2 · The Dark Temple"
    MOVIE = "Raiders of the Lost Ark (1981)"
    ALGO = "SARSA (on-policy TD control)"

    def __init__(self, seed=None):
        super().__init__(
            size=10, start=(0, 0), goal=(9, 9), seed=seed,
            walls=[(1, 7), (7, 3), (5, 8), (8, 5)],          # temple stones
            slippery=[(2, 3), (4, 5), (5, 5), (3, 6), (6, 6)],   # mud beside pits
            traps=[(2, 2), (5, 2), (3, 5), (6, 4), (4, 7), (7, 7)],  # spike pits
        )


# --------------------------------------------------------------------------- #
# Room 3 — The Cloning Lab (The Matrix) — Q-Learning (Cliff Walking)
# --------------------------------------------------------------------------- #
class Room3CloningLab(GridWorld):
    NAME = "Room 3 · The Cloning Lab"
    MOVIE = "The Matrix (1999)"
    ALGO = "Q-Learning (off-policy TD control)"

    def __init__(self, seed=None):
        # "Firewall" pillars force the safe route up and over, sharpening the
        # risk/reward of hugging the cliff of clones along the bottom.
        super().__init__(size=10, start=(0, 0), goal=(9, 0), seed=seed,
                         walls=[(3, 2), (6, 2)], slippery=[], traps=[])
        # Cliff = bottom row between start and goal.  Canonical Cliff-Walking:
        # stepping on it costs -100 and RESETS to start (episode continues).
        self.cliff = {(x, 0) for x in range(1, 9)}           # x = 1..8

    def step(self, a):
        s2 = self._apply(self._state, _DELTA[a])             # deterministic
        if s2 in self.cliff:
            self._state = self.start
            return self.start, TRAP_REWARD, False            # penalty + reset
        r, done = self.reward_done(s2)
        self._state = s2
        return s2, r, done

    def render_meta(self):
        m = super().render_meta()
        m["cliff"] = self.cliff
        return m


# --------------------------------------------------------------------------- #
# Room 4 — The Hovercar Garage (Fast & Furious) — Function Approximation
# --------------------------------------------------------------------------- #
# 9 discrete acceleration actions (ax, ay) in {-1,0,1}^2.
ACCEL_ACTIONS = [(ax, ay) for ax in (-1, 0, 1) for ay in (-1, 0, 1)]


class Room4Garage:
    """Continuous 2-D chase. State = [X, Y, Vx, Vy], solved with tile-coded
    linear function approximation (see algorithms.LinearFAAgent)."""

    NAME = "Room 4 · The Hovercar Garage"
    MOVIE = "The Fast and the Furious (2001)"
    ALGO = "Semi-gradient Q-Learning + tile-coding (Function Approximation)"
    DT = 0.02

    def __init__(self, v_max=3.0, max_steps=800, shaping=True,
                 shaping_coef=5.0, shaping_gamma=1.0, hard_walls=False):
        self.v_max = float(v_max)
        self.max_steps = int(max_steps)
        self.hard_walls = bool(hard_walls)
        self.start = (1.0, 1.0)
        self.bounds = (0.0, 10.0, 0.0, 10.0)                 # xmin,xmax,ymin,ymax
        # Parked cars as axis-aligned boxes (xmin, xmax, ymin, ymax).
        self.obstacles = [(3.0, 4.0, 0.0, 6.0), (6.0, 7.0, 4.0, 10.0)]
        self.exit_region = (8.5, 8.5)                        # X>8.5 AND Y>8.5
        self.n_actions = 9
        self.actions = list(range(9))
        self.state_low = np.array([0.0, 0.0, -self.v_max, -self.v_max])
        self.state_high = np.array([10.0, 10.0, self.v_max, self.v_max])
        # Potential-based reward shaping (Ng et al. 1999) using an obstacle-aware
        # wave-front distance-to-goal.  Policy-invariant, but turns the sparse
        # serpentine maze into a learnable gradient.  Toggerable from the UI.
        self.shaping = bool(shaping)
        self.shaping_coef = float(shaping_coef)
        self.shaping_gamma = float(shaping_gamma)
        self._build_distance_map()
        self._succeeded = False
        self.reset()

    # ---- obstacle-aware wave-front potential ----------------------------- #
    def _build_distance_map(self, res=0.2, margin=0.15):
        n = int(round(10.0 / res))
        self._dm_res, self._dm_n = res, n
        centres = (np.arange(n) + 0.5) * res
        blocked = np.zeros((n, n), dtype=bool)               # index [ix, iy]
        for xn, xx, yn, yx in self.obstacles:
            mx = (centres >= xn - margin) & (centres <= xx + margin)
            my = (centres >= yn - margin) & (centres <= yx + margin)
            blocked |= np.outer(mx, my)
        INF = 10 ** 9
        dist = np.full((n, n), INF, dtype=float)
        dq = deque()
        goal = np.outer(centres > 8.5, centres > 8.5) & ~blocked
        for ix, iy in zip(*np.where(goal)):
            dist[ix, iy] = 0.0
            dq.append((ix, iy))
        while dq:                                            # BFS from the exit
            ix, iy = dq.popleft()
            d0 = dist[ix, iy]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < n and 0 <= jy < n and not blocked[jx, jy] \
                        and dist[jx, jy] > d0 + 1:
                    dist[jx, jy] = d0 + 1
                    dq.append((jx, jy))
        phi = -dist * res                                    # potential in metres
        far = phi[dist < INF].min() if np.any(dist < INF) else 0.0
        phi[dist >= INF] = far - 5.0                          # walls/unreachable
        self._phi_map = phi

    def _phi(self, x, y):
        ix = min(self._dm_n - 1, max(0, int(x / self._dm_res)))
        iy = min(self._dm_n - 1, max(0, int(y / self._dm_res)))
        return self._phi_map[ix, iy]

    def reset(self):
        self.X, self.Y, self.Vx, self.Vy = 1.0, 1.0, 0.0, 0.0
        self.t = 0
        self._succeeded = False
        return self._obs()

    def _obs(self):
        return np.array([self.X, self.Y, self.Vx, self.Vy], dtype=float)

    def _in_obstacle(self, x, y):
        return any(xn <= x <= xx and yn <= y <= yx
                   for (xn, xx, yn, yx) in self.obstacles)

    def _out_of_bounds(self, x, y):
        return x < 0.0 or x > 10.0 or y < 0.0 or y > 10.0

    def step(self, a):
        ox, oy = self.X, self.Y
        ax, ay = ACCEL_ACTIONS[a]
        self.Vx = float(np.clip(self.Vx + ax, -self.v_max, self.v_max))
        self.Vy = float(np.clip(self.Vy + ay, -self.v_max, self.v_max))
        self.X += self.Vx * self.DT
        self.Y += self.Vy * self.DT
        self.t += 1

        # Parked cars are always a fatal crash (absorbing, -100).
        if self._in_obstacle(self.X, self.Y):
            return self._obs(), -100.0, True

        bumped = False
        if self._out_of_bounds(self.X, self.Y):
            if self.hard_walls:                              # strict MD reading
                return self._obs(), -100.0, True
            # soft garage wall: clamp position, kill inward velocity, small nudge
            if self.X < 0.0 or self.X > 10.0:
                self.X = float(np.clip(self.X, 0.0, 10.0)); self.Vx = 0.0
            if self.Y < 0.0 or self.Y > 10.0:
                self.Y = float(np.clip(self.Y, 0.0, 10.0)); self.Vy = 0.0
            bumped = True

        if self.X > self.exit_region[0] and self.Y > self.exit_region[1]:
            self._succeeded = True
            return self._obs(), 100.0, True

        reward = -0.1 + (-1.0 if bumped else 0.0)
        if self.shaping:                                     # potential-based shaping
            reward += self.shaping_coef * (
                self.shaping_gamma * self._phi(self.X, self.Y) - self._phi(ox, oy))
        if self.t >= self.max_steps:
            return self._obs(), reward, True                 # timeout (not success)
        return self._obs(), reward, False

    def is_success(self):
        return self._succeeded

    def render_frame(self):
        return {"agent": (self.X, self.Y)}

    def render_meta(self):
        return dict(kind="garage", bounds=self.bounds, obstacles=self.obstacles,
                    exit_region=self.exit_region, start=self.start,
                    agent_y=None)


# --------------------------------------------------------------------------- #
# Room 5 — The Space Escape (Star Wars) — dynamic obstacles + partial observ.
# --------------------------------------------------------------------------- #
class Room5SpaceEscape:
    """Dodge falling neuralyzer-drones.  The agent moves on the X axis only
    (Y fixed) and observes ONLY [Vx, distance-to-nearest-aligned-obstacle]
    (or -1 when nothing is within its vision range) — a deliberate POMDP."""

    NAME = "Room 5 · The Space Escape"
    MOVIE = "Star Wars (1977)"
    ALGO = "Tabular Q-Learning over a partial-observation sensor"
    DT = 0.02
    V_OBS = -5.0            # obstacle downward speed
    OBS_WIDTH = 0.5        # obstacle width == collision radius (centre-to-centre)
    AGENT_Y = 2.0
    D_BINS = 10            # discretisation of the distance sensor

    def __init__(self, vision=3.0, spawn_every=15, v_max=5.0,
                 max_steps=500, seed=None):
        self.vision = float(vision)
        self.spawn_every = int(spawn_every)
        self.v_max = float(v_max)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.n_actions = 3                                   # ax in {-1, 0, 1}
        self.actions = [0, 1, 2]
        self._survived = False
        self.reset()

    def reset(self):
        self.X = 5.0
        self.Vx = 0.0
        self.obstacles = []                                  # list of [x, y]
        self.t = 0
        self._survived = False
        return self._obs()

    def random_room(self, seed=None):
        """Fresh episode with a new random obstacle stream (test button)."""
        self.rng = np.random.default_rng(seed)
        return self.reset()

    def _spawn(self):
        self.obstacles.append([float(self.rng.uniform(1.0, 9.0)), 10.0])

    def _obs(self):
        """Sensor = [Vx, d].  d = vertical gap to the nearest obstacle that is
        above the agent AND horizontally aligned (|dx| <= width), if that gap
        is within `vision`; otherwise -1."""
        best = -1.0
        best_d = np.inf
        for ox, oy in self.obstacles:
            if oy >= self.AGENT_Y and abs(ox - self.X) <= self.OBS_WIDTH:
                d = oy - self.AGENT_Y
                if d <= self.vision and d < best_d:
                    best_d, best = d, d
        return np.array([self.Vx, best], dtype=float)

    def encode(self, obs):
        """Map the continuous observation onto a discrete (vx, d) tabular state."""
        vx, d = obs
        vx_idx = int(round(vx)) + int(self.v_max)            # 0 .. 2*v_max
        if d < 0:
            d_idx = 0                                        # "clear ahead"
        else:
            frac = min(0.999999, d / max(self.vision, 1e-9))
            d_idx = 1 + int(frac * self.D_BINS)              # 1 .. D_BINS
        return (vx_idx, d_idx)

    def step(self, a):
        accel = a - 1                                        # {0,1,2} -> {-1,0,1}
        self.Vx = float(np.clip(self.Vx + accel, -self.v_max, self.v_max))
        self.X = float(np.clip(self.X + self.Vx * self.DT, 0.0, 10.0))

        for o in self.obstacles:                             # fall
            o[1] += self.V_OBS * self.DT
        self.obstacles = [o for o in self.obstacles if o[1] > -1.0]
        if self.t % self.spawn_every == 0:                   # dynamic spawn
            self._spawn()
        self.t += 1

        for ox, oy in self.obstacles:                        # collision
            if np.hypot(self.X - ox, self.AGENT_Y - oy) < self.OBS_WIDTH:
                return self._obs(), -1000.0, True
        if self.t >= self.max_steps:
            self._survived = True
            return self._obs(), 1.0, True                    # escaped
        return self._obs(), 1.0, False

    def is_success(self):
        return self._survived

    def render_frame(self):
        return {"agent": (self.X, self.AGENT_Y),
                "obstacles": [(ox, oy) for ox, oy in self.obstacles]}

    def render_meta(self):
        return dict(kind="space", bounds=(0.0, 10.0, 0.0, 10.0),
                    agent_y=self.AGENT_Y, obs_width=self.OBS_WIDTH,
                    vision=self.vision)


# --------------------------------------------------------------------------- #
# Registry — drives the sidebar's difficulty progression
# --------------------------------------------------------------------------- #
ROOM_REGISTRY = [
    dict(key="room1", cls=Room1FrozenArchive, stars=1, emoji="❄️"),
    dict(key="room2", cls=Room2DarkTemple,    stars=2, emoji="🏛️"),
    dict(key="room3", cls=Room3CloningLab,    stars=3, emoji="🕶️"),
    dict(key="room4", cls=Room4Garage,        stars=4, emoji="🏎️"),
    dict(key="room5", cls=Room5SpaceEscape,   stars=5, emoji="🚀"),
]
