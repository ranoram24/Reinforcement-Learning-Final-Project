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
        self._va_cache = {}
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

    def valid_actions(self, s):
        """Action masking: only actions whose intended move actually leaves the
        current cell (i.e. NOT into a wall or off the board). If a cell were
        fully enclosed we fall back to all actions so nothing can crash."""
        va = self._va_cache.get(s)
        if va is None:
            va = [a for a in self.actions if self._apply(s, _DELTA[a]) != s]
            va = va or list(self.actions)
            self._va_cache[s] = va
        return va

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
class Room1FrozenArchive:
    """LEVEL 1 — Ice Age key-and-door puzzle.

    Hezki must collect the KEY, which melts the ice-gate (door), then reach the
    EXIT.  Optional bonus tiles are one-off pickups; a hazard tile costs points
    but is not terminal.

    Because pickups are one-off, the state is AUGMENTED with a bitmask of what
    has been collected:  state = (x, y, mask).  Step reward is 0 (a gentle first
    room) — discounting (γ<1) is what still rewards a short route.

    Slip model (per-tile AND per-action): a tile may say "DOWN -> 60% down /
    40% right".  That slip only applies when the agent chooses THAT action; any
    other action on the same tile is deterministic.
    """

    NAME = "Room 1 · The Frozen Archive"
    MOVIE = "Ice Age (2002)"
    ALGO = "Value Iteration (Dynamic Programming)"
    SIZE = 10
    STEP_REWARD = 0.0
    EXIT_REWARD = 100.0

    # ---- Layout, written TOP row first (row 0 = top), 10x10 ---------------- #
    #   '.' blank   '#' wall   'S' start   'E' exit(+100, terminal)
    #   'D' door (acts as a wall until the key is held, then vanishes)
    #   'K' key     'a''b''c' bonus pickups   'x' hazard   '~' slippery
    LAYOUT = [
        "...~~....S",
        ".##..####.",
        ".##..#K##.",
        "~....a~...",
        ".#########",
        ".....~..~~",
        ".~~##b##~c",
        "#..##.##~~",
        "D~...~.#..",
        "E#......x.",
    ]
    PICKUPS = {"K": 50.0, "a": 15.0, "b": 5.0, "c": 20.0}
    HAZARD = -50.0

    # Slip rules keyed by (col, row-from-top) -> {action: [(prob, direction)]}
    SLIP = {
        (3, 0): {DOWN:  [(0.60, DOWN),  (0.40, LEFT)]},
        (4, 0): {DOWN:  [(0.50, DOWN),  (0.50, RIGHT)]},
        (0, 3): {DOWN:  [(0.70, DOWN),  (0.30, RIGHT)]},
        (6, 3): {UP:    [(0.60, UP),    (0.40, RIGHT)]},   # the only way to the key
        (5, 5): {DOWN:  [(0.40, DOWN),  (0.60, RIGHT)]},
        (8, 5): {DOWN:  [(0.40, DOWN),  (0.60, LEFT)]},
        (9, 5): {DOWN:  [(0.40, DOWN),  (0.60, LEFT)]},
        (1, 6): {DOWN:  [(0.60, DOWN),  (0.40, RIGHT)]},
        (2, 6): {DOWN:  [(0.60, DOWN),  (0.40, UP)]},
        (8, 6): {RIGHT: [(0.30, RIGHT), (0.70, DOWN)]},
        (8, 7): {RIGHT: [(0.30, RIGHT), (0.70, DOWN)]},
        (9, 7): {DOWN:  [(0.40, DOWN),  (0.60, LEFT)]},
        (1, 8): {LEFT:  [(0.60, LEFT),  (0.40, UP)]},      # the only way to the gate
        (5, 8): {UP:    [(0.60, UP),    (0.40, RIGHT)]},
    }

    def __init__(self, seed=None):
        self.size = self.SIZE
        self.actions = GRID_ACTIONS
        self.n_actions = 4
        self.rng = np.random.default_rng(seed)
        self._va_cache = {}

        self.walls, self.doors, self.slippery = set(), set(), {}
        self.pickup_at, self.hazards = {}, set()
        self.start = self.exit = None
        ice_tiles = set()
        for row, line in enumerate(self.LAYOUT):
            y = self.SIZE - 1 - row                       # flip: row 0 is the TOP
            for x, ch in enumerate(line):
                cell = (x, y)
                if ch == "#":   self.walls.add(cell)
                elif ch == "D": self.doors.add(cell)
                elif ch == "S": self.start = cell
                elif ch == "E": self.exit = cell
                elif ch == "x": self.hazards.add(cell)
                elif ch == "~": ice_tiles.add(cell)
                elif ch in self.PICKUPS: self.pickup_at[cell] = ch
        for (cx, cr), rules in self.SLIP.items():
            self.slippery[(cx, self.SIZE - 1 - cr)] = rules

        # the drawn ice ('~') and the slip rules must describe the same tiles
        if ice_tiles != set(self.slippery):
            raise ValueError(
                "Room 1 layout/SLIP mismatch — "
                f"in LAYOUT only: {sorted(ice_tiles - set(self.slippery))}; "
                f"in SLIP only: {sorted(set(self.slippery) - ice_tiles)}")

        # stable bit order for the collected-mask
        self.pickup_order = sorted(self.pickup_at)          # list of cells
        self.bit = {c: i for i, c in enumerate(self.pickup_order)}
        self.n_pickups = len(self.pickup_order)
        self.key_cell = next((c for c, ch in self.pickup_at.items() if ch == "K"), None)
        self.key_bit = self.bit[self.key_cell] if self.key_cell else None
        self.reset()

    # ---- helpers ---------------------------------------------------------- #
    def has_key(self, mask):
        return self.key_bit is None or bool((mask >> self.key_bit) & 1)

    def in_bounds(self, c):
        return 0 <= c[0] < self.SIZE and 0 <= c[1] < self.SIZE

    def blocked(self, cell, mask):
        """Walls always block; a door blocks only while the key is missing."""
        return (cell in self.walls) or (cell in self.doors and not self.has_key(mask))

    def is_terminal(self, s):
        return (s[0], s[1]) == self.exit

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def _move(self, s, direction):
        x, y, mask = s
        nxt = (x + _DELTA[direction][0], y + _DELTA[direction][1])
        if not self.in_bounds(nxt) or self.blocked(nxt, mask):
            return (x, y)
        return nxt

    def valid_actions(self, s):
        """Action masking — only moves that leave the cell (a shut gate counts
        as a wall until the key is held). Cached per (cell, key-held)."""
        k = (s[0], s[1], self.has_key(s[2]))
        va = self._va_cache.get(k)
        if va is None:
            here = (s[0], s[1])
            va = [a for a in self.actions if self._move((s[0], s[1], s[2]), a) != here]
            va = va or list(self.actions)
            self._va_cache[k] = va
        return va

    def outcomes(self, s, a):
        """[(prob, cell)] — slip applies only to the action the tile names."""
        rules = self.slippery.get((s[0], s[1]), {})
        cands = rules.get(a, [(1.0, a)])
        dist = {}
        for p, direction in cands:
            c = self._move(s, direction)
            dist[c] = dist.get(c, 0.0) + p
        return [(p, c) for c, p in dist.items()]

    def _enter(self, cell, mask):
        """(reward, new_mask, done) for stepping onto `cell` holding `mask`."""
        r, done = self.STEP_REWARD, False
        if cell in self.pickup_at:
            i = self.bit[cell]
            if not (mask >> i) & 1:                        # one-off pickup
                r += self.PICKUPS[self.pickup_at[cell]]
                mask |= (1 << i)
        if cell in self.hazards:
            r += self.HAZARD
        if cell == self.exit:
            r += self.EXIT_REWARD
            done = True
        return r, mask, done

    def transitions(self, s, a):
        out = []
        for p, cell in self.outcomes(s, a):
            r, m2, done = self._enter(cell, s[2])
            out.append((p, (cell[0], cell[1], m2), r, done))
        return out

    # ---- DP interface ----------------------------------------------------- #
    def states(self):
        return [(x, y, m)
                for x in range(self.SIZE) for y in range(self.SIZE)
                if (x, y) not in self.walls
                for m in range(1 << self.n_pickups)]

    def build_model(self):
        return {s: {a: self.transitions(s, a) for a in self.actions}
                for s in self.states() if not self.is_terminal(s)}

    # ---- model-free interface --------------------------------------------- #
    def reset(self):
        self._state = (self.start[0], self.start[1], 0)
        return self._state

    def step(self, a):
        outs = self.outcomes(self._state, a)
        probs = [p for p, _ in outs]
        cell = outs[int(self.rng.choice(len(outs), p=probs))][1]
        r, m2, done = self._enter(cell, self._state[2])
        self._state = (cell[0], cell[1], m2)
        return self._state, r, done

    def is_success(self):
        return (self._state[0], self._state[1]) == self.exit

    def encode(self, s):
        return s

    def render_frame(self):
        return {"agent": (self._state[0], self._state[1]), "mask": self._state[2]}

    def render_meta(self):
        return dict(kind="keydoor", size=self.SIZE, start=self.start, goal=self.exit,
                    walls=self.walls, doors=self.doors, slippery=self.slippery,
                    pickups=self.pickup_at, pickup_rewards=self.PICKUPS,
                    bit=self.bit, hazards=self.hazards, hazard_reward=self.HAZARD,
                    exit_reward=self.EXIT_REWARD, traps=set(), cliff=set())


# --------------------------------------------------------------------------- #
# Room 2 — The Dark Temple (Indiana Jones) — SARSA
# --------------------------------------------------------------------------- #
class Room2DarkTemple:
    """LEVEL 2 — Raiders temple: golden idol (key) → stone door → exit.

    Mechanics
    ---------
    * step reward 0; idol +1000; treasure +100 (one-off); exit +2000 (terminal).
    * **Spike pits** cost −50/−100 and throw the agent back to the start, but do
      NOT end the episode.
    * The **stone door** blocks the exit until the idol is taken, then vanishes.
    * A **pressure plate** appears once the idol is held.  Standing on it wakes
      the **boulder**, which then *retraces the agent's own trail* a few steps
      behind — so it only catches you if you double back.  Being caught costs
      −1500, throws you to the start and RESETS the temple to its default layout
      (idol and treasure respawn, door returns, boulder gone).
    * The plate is a pure trap: the exit is already opened by the idol alone.

    State = (x, y, collected-mask, chaser-position or None).
    """

    NAME = "Room 3 · The Dark Temple"     # (this env is used at grid position 'room3')
    MOVIE = "Raiders of the Lost Ark (1981)"
    ALGO = "Q-Learning (off-policy TD control)"
    SIZE = 10
    STEP_REWARD = 0.0
    EXIT_REWARD = 200000.0
    CATCH_REWARD = -100000.0
    BUTTON_REWARD = 10000.0        # pressing the plate pays once, then it's gone

    # row 0 = top.  '.' blank  '#' wall  '~' slippery  'S' start  'E' exit
    # 'D' door  'K' idol/key  'G' treasure  'h' pit −50  'H' pit −100
    # 'B' pressure plate (only exists once the idol is held)
    LAYOUT = [
        "K.B.#GGGGG",
        "~.#.#GGGGG",
        "..#G#H#HH~",
        "..#.#.#...",
        "..#.#.#.HH",
        "~.#.#.#..G",
        "h.#.#.###G",
        "..#..~....",
        ".######.#.",
        "S#ED....#.",
    ]
    PICKUPS = {"K": 100000.0, "G": 10000.0}
    HOLE_REWARD = {"h": -500.0, "H": -1000.0}
    SLIP = {
        (0, 1): {UP:    [(0.50, UP),    (0.50, DOWN)]},
        (9, 2): {UP:    [(0.60, UP),    (0.40, LEFT)]},
        (0, 5): {UP:    [(0.70, UP),    (0.30, DOWN)]},
        (5, 7): {RIGHT: [(0.80, RIGHT), (0.20, UP)]},
    }

    def __init__(self, seed=None):
        self.size = self.SIZE
        self.actions = GRID_ACTIONS
        self.n_actions = 4
        self.rng = np.random.default_rng(seed)
        self._va_cache = {}

        self.walls, self.doors, self.slippery = set(), set(), {}
        self.pickup_at, self.holes = {}, {}
        self.start = self.exit = self.button = None
        ice_tiles = set()
        for row, line in enumerate(self.LAYOUT):
            y = self.SIZE - 1 - row                       # flip: row 0 is the TOP
            for x, ch in enumerate(line):
                c = (x, y)
                if ch == "#":   self.walls.add(c)
                elif ch == "D": self.doors.add(c)
                elif ch == "S": self.start = c
                elif ch == "E": self.exit = c
                elif ch == "B": self.button = c
                elif ch == "~": ice_tiles.add(c)
                elif ch in self.HOLE_REWARD: self.holes[c] = self.HOLE_REWARD[ch]
                elif ch in self.PICKUPS: self.pickup_at[c] = ch
        for (cx, cr), rules in self.SLIP.items():
            self.slippery[(cx, self.SIZE - 1 - cr)] = rules
        if ice_tiles != set(self.slippery):
            raise ValueError(
                "Room 2 layout/SLIP mismatch — "
                f"in LAYOUT only: {sorted(ice_tiles - set(self.slippery))}; "
                f"in SLIP only: {sorted(set(self.slippery) - ice_tiles)}")

        self.pickup_order = sorted(self.pickup_at)
        self.bit = {c: i for i, c in enumerate(self.pickup_order)}
        self.n_pickups = len(self.pickup_order)
        self.button_bit = self.n_pickups             # extra mask bit: plate pressed?
        self.key_cell = next(c for c, ch in self.pickup_at.items() if ch == "K")
        self.key_bit = self.bit[self.key_cell]
        self.chaser_spawn = self.key_cell            # boulder wakes where the idol was
        self.reset()

    # ---- helpers ---------------------------------------------------------- #
    def has_key(self, mask):
        return bool((mask >> self.key_bit) & 1)

    def in_bounds(self, c):
        return 0 <= c[0] < self.SIZE and 0 <= c[1] < self.SIZE

    def blocked(self, cell, mask):
        return (cell in self.walls) or (cell in self.doors and not self.has_key(mask))

    def is_terminal(self, s):
        return (s[0], s[1]) == self.exit

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def _move(self, s, direction):
        x, y, mask = s[0], s[1], s[2]
        nxt = (x + _DELTA[direction][0], y + _DELTA[direction][1])
        if not self.in_bounds(nxt) or self.blocked(nxt, mask):
            return (x, y)
        return nxt

    def valid_actions(self, s):
        """Action masking — only moves that leave the cell (a shut gate counts
        as a wall until the idol is held). Cached per (cell, idol-held)."""
        k = (s[0], s[1], self.has_key(s[2]))
        va = self._va_cache.get(k)
        if va is None:
            here = (s[0], s[1])
            va = [a for a in self.actions if self._move(s, a) != here]
            va = va or list(self.actions)
            self._va_cache[k] = va
        return va

    def outcomes(self, s, a):
        """[(prob, cell)] — slip applies only to the action the tile names."""
        cands = self.slippery.get((s[0], s[1]), {}).get(a, [(1.0, a)])
        dist = {}
        for p, direction in cands:
            c = self._move(s, direction)
            dist[c] = dist.get(c, 0.0) + p
        return [(p, c) for c, p in dist.items()]

    # ---- model-free interface --------------------------------------------- #
    def reset(self):
        self._path = [self.start]        # every cell the agent has stood on
        self._chaser_i = None            # boulder's index into that path
        self._state = (self.start[0], self.start[1], 0, None)
        return self._state

    def step(self, a):
        x, y, mask, chaser = self._state
        outs = self.outcomes(self._state, a)
        cell = outs[int(self.rng.choice(len(outs), p=[p for p, _ in outs]))][1]
        r, done = self.STEP_REWARD, False

        if cell in self.pickup_at:                          # idol / treasure
            i = self.bit[cell]
            if not (mask >> i) & 1:
                r += self.PICKUPS[self.pickup_at[cell]]
                mask |= 1 << i

        if cell in self.holes:                               # pit: penalty + restart
            r += self.holes[cell]
            cell = self.start

        if cell == self.exit:
            r += self.EXIT_REWARD
            done = True

        self._path.append(cell)                              # record where we now stand

        prev_chaser = chaser
        pressed = bool((mask >> self.button_bit) & 1)
        if cell == self.button and self.has_key(mask) and not pressed:
            # First (and ONLY) press: pays +BUTTON_REWARD, then the plate is
            # consumed. The boulder appears at once where the idol lay and from
            # here retraces the agent's own route one cell per step.
            r += self.BUTTON_REWARD
            mask |= 1 << self.button_bit                     # plate gone for good
            self._chaser_i = max((i for i, c in enumerate(self._path)
                                  if c == self.chaser_spawn), default=0)
            chaser = self._path[self._chaser_i]
        elif self._chaser_i is not None:
            self._chaser_i = min(self._chaser_i + 1, len(self._path) - 1)
            chaser = self._path[self._chaser_i]

        if chaser is not None and not done and (cell == chaser or cell == prev_chaser):
            r += self.CATCH_REWARD                           # caught: one-time big penalty
            chaser, self._chaser_i = None, None              # boulder vanishes; the agent
            #   STAYS where it is and keeps everything (gate open, plate used) — the plate
            #   can't be re-pressed, so the boulder never returns; just head for the exit.

        self._state = (cell[0], cell[1], mask, chaser)
        return self._state, r, done

    def encode(self, s):
        """Agent observation — a deliberate STATE ABSTRACTION.

        The *environment* still tracks the full one-off pickup bitmask, so
        nothing can ever be farmed twice.  The *agent* however only observes
        `(x, y, holding-the-idol?, boulder relative position)`.

        Why: exposing all 2^10 pickup combinations left almost every state
        barely visited, so the greedy policy contained cycles and 22 % of runs
        dead-looped until the step cap (78 % escape).  With this abstraction the
        table halves and the room is solved 100 % of the time.
        """
        x, y, mask, ch = s
        rel = None if ch is None else (max(-3, min(3, ch[0] - x)),
                                       max(-3, min(3, ch[1] - y)))
        return (x, y, bool((mask >> self.key_bit) & 1),
                bool((mask >> self.button_bit) & 1), rel)

    def is_success(self):
        return (self._state[0], self._state[1]) == self.exit

    def render_frame(self):
        return {"agent": (self._state[0], self._state[1]),
                "mask": self._state[2], "chaser": self._state[3]}

    def render_meta(self):
        return dict(kind="keydoor", size=self.SIZE, start=self.start, goal=self.exit,
                    walls=self.walls, doors=self.doors, slippery=self.slippery,
                    pickups=self.pickup_at, pickup_rewards=self.PICKUPS,
                    bit=self.bit, hazards=dict(self.holes),
                    hazard_resets=True, button=self.button,
                    button_bit=self.button_bit, button_reward=self.BUTTON_REWARD,
                    catch_reward=self.CATCH_REWARD,
                    exit_reward=self.EXIT_REWARD, traps=set(), cliff=set())


# --------------------------------------------------------------------------- #
# Room 3 — The Cloning Lab (The Matrix) — Q-Learning (Cliff Walking)
# --------------------------------------------------------------------------- #
class Room3CloningLab(GridWorld):
    NAME = "Room 2 · The Cloning Lab"     # (this env is used at grid position 'room2')
    MOVIE = "The Matrix (1999)"
    ALGO = "SARSA (on-policy TD control)"

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
# Room 2 — The Cloning Lab (The Matrix) — SARSA — SOKOBAN box puzzle
# --------------------------------------------------------------------------- #
class Room2CloningLab:
    """Push the two boxes onto the two pressure plates; that opens the ice-gate,
    then reach the exit.

    Mechanics
    ---------
    * step −1; each plate pays +500000 the FIRST time a box lands on it; the
      exit pays +1000000 (terminal). One-off bonus tiles pay +100000 each.
    * **Pushing**: moving into a box slides it one cell IF the cell beyond is
      free (no wall/border/box); the agent then STAYS put. A box already on a
      plate is locked and cannot be pushed.
    * **Door** at the top-left is a wall until BOTH plates hold a box; then it
      opens and the exit is reachable through it.
    * **Reset tile**: stepping on it returns every position (agent + boxes) to
      the start — a way out of a dead-lock — but does NOT reset the collected
      bonuses / already-paid plates (so nothing can be farmed), nor the steps.
    * Per-tile / per-action slip, and action masking (no wall/border moves).

    The ENVIRONMENT tracks the full state (agent, boxes, bonus mask, paid
    plates).  The AGENT only OBSERVES (agent, box positions) — the door and
    plate coverage are derivable from the box positions — which keeps the
    tabular state space to a few thousand.
    """

    NAME = "Room 2 · The Cloning Lab"
    MOVIE = "The Matrix (1999)"
    ALGO = "SARSA (on-policy TD control)"
    SIZE = 10
    STEP_REWARD = -1.0
    EXIT_REWARD = 10000.0
    BUTTON_REWARD = 5000.0
    BONUS_REWARD = 1000.0
    NEG_REWARD = -100.0

    # row 0 = top.  '.' blank  '#' wall  '~' slippery  'S' start  'E' exit
    # 'D' door  'B' plate/button  'X' box  'G' bonus (+1000, one-off)  'R' reset
    # 'N' negative-reward tile (−100, every time you step on it)
    LAYOUT = [
        "ED..NN~...",
        "#NN~...NN.",
        ".....####.",
        "~~~..#B...",
        "#~#..#.X.#",
        "GGG#.#...#",
        "GGG#.##..#",
        "GG#......#",
        "##..######",
        "S......X.B",
    ]
    SLIP = {
        (6, 0): {DOWN:  [(0.40, DOWN),  (0.60, RIGHT)]},   # row 1, col 7
        (3, 1): {UP:    [(0.40, LEFT),  (0.60, UP)]},      # row 2, col 4
        (0, 3): {RIGHT: [(0.30, RIGHT), (0.70, UP)]},      # row 4, col 1
        (1, 3): {DOWN:  [(0.40, DOWN),  (0.60, RIGHT)]},   # row 4, col 2
        (2, 3): {LEFT:  [(0.40, LEFT),  (0.60, UP)]},      # row 4, col 3
        (1, 4): {DOWN:  [(0.40, DOWN),  (0.60, UP)]},      # row 5, col 2
    }

    def __init__(self, seed=None):
        self.size = self.SIZE
        self.actions = GRID_ACTIONS
        self.n_actions = 4
        self.rng = np.random.default_rng(seed)
        self._va_cache = {}

        self.walls, self.doors, self.slippery, self.bonus_at = set(), set(), {}, {}
        self.buttons, self.box_start = [], []
        self.negatives = set()
        self.start = self.exit = self.reset_tile = None
        ice = set()
        for row, line in enumerate(self.LAYOUT):
            y = self.SIZE - 1 - row
            for x, ch in enumerate(line):
                c = (x, y)
                if ch == "#":   self.walls.add(c)
                elif ch == "D": self.doors.add(c)
                elif ch == "E": self.exit = c
                elif ch == "S": self.start = c
                elif ch == "R": self.reset_tile = c
                elif ch == "B": self.buttons.append(c)
                elif ch == "X": self.box_start.append(c)
                elif ch == "G": self.bonus_at[c] = None
                elif ch == "N": self.negatives.add(c)
                elif ch == "~": ice.add(c)
        for (cx, cr), rules in self.SLIP.items():
            self.slippery[(cx, self.SIZE - 1 - cr)] = rules
        if ice != set(self.slippery):
            raise ValueError("Room 2 layout/SLIP mismatch — "
                             f"LAYOUT only: {sorted(ice - set(self.slippery))}; "
                             f"SLIP only: {sorted(set(self.slippery) - ice)}")
        self.bonus_order = sorted(self.bonus_at)
        self.bonus_bit = {c: i for i, c in enumerate(self.bonus_order)}
        self.n_bonus = len(self.bonus_order)
        self.buttons = tuple(sorted(self.buttons))
        self.box_start = tuple(sorted(self.box_start))
        self.reset()

    # ---- helpers ---------------------------------------------------------- #
    def in_bounds(self, c):
        return 0 <= c[0] < self.SIZE and 0 <= c[1] < self.SIZE

    def door_open(self, boxes):
        return all(b in boxes for b in self.buttons)

    def _box_blocked(self, c, boxes):
        """Can a box occupy cell c?  (walls, borders, doors, exit, reset, boxes)"""
        return (not self.in_bounds(c) or c in self.walls or c in self.doors
                or c == self.exit or c == self.reset_tile or c in boxes)

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def valid_actions(self, s):
        """Action masking — no moves into a wall/border/closed gate.  (Moving
        into a box is allowed: it's a push attempt.)"""
        ax, ay, boxes = s[0], s[1], s[2]
        dopen = self.door_open(boxes)
        k = (ax, ay, dopen)
        va = self._va_cache.get(k)
        if va is None:
            va = []
            for a in self.actions:
                c = (ax + _DELTA[a][0], ay + _DELTA[a][1])
                if not self.in_bounds(c) or c in self.walls:
                    continue
                if c in self.doors and not dopen:
                    continue
                va.append(a)
            va = va or list(self.actions)
            self._va_cache[k] = va
        return va

    def encode(self, s):
        """Agent observation: position + box configuration (door derivable)."""
        return (s[0], s[1], s[2])

    # ---- model-free interface --------------------------------------------- #
    def reset(self):
        self._boxes = set(self.box_start)
        self._mask = 0
        self._paid = [False] * len(self.buttons)
        self._state = (self.start[0], self.start[1], tuple(sorted(self._boxes)),
                       self._mask, tuple(self._paid))
        return self._state

    def step(self, a):
        ax, ay = self._state[0], self._state[1]
        boxes, mask, paid = set(self._boxes), self._mask, list(self._paid)
        dopen = self.door_open(boxes)

        # per-action slip → sample the ACTUAL direction
        cands = self.slippery.get((ax, ay), {}).get(a, [(1.0, a)])
        dirs = [d for _, d in cands]
        direction = dirs[int(self.rng.choice(len(dirs), p=[p for p, _ in cands]))]
        dx, dy = _DELTA[direction]
        nxt = (ax + dx, ay + dy)

        r, done, agent = self.STEP_REWARD, False, (ax, ay)
        blocked_wall = (not self.in_bounds(nxt) or nxt in self.walls
                        or (nxt in self.doors and not dopen))
        if blocked_wall:
            pass                                             # stay in place
        elif nxt in boxes:                                   # push attempt
            behind = (nxt[0] + dx, nxt[1] + dy)
            if nxt not in self.buttons and not self._box_blocked(behind, boxes):
                boxes.discard(nxt); boxes.add(behind)        # box slides, agent stays
                for i, btn in enumerate(self.buttons):
                    if behind == btn and not paid[i]:
                        r += self.BUTTON_REWARD; paid[i] = True
        else:                                                # free move
            agent = nxt
            if nxt in self.bonus_at:                         # one-off bonus
                i = self.bonus_bit[nxt]
                if not (mask >> i) & 1:
                    r += self.BONUS_REWARD; mask |= 1 << i
            if nxt in self.negatives:                        # penalty tile (repeatable)
                r += self.NEG_REWARD
            if nxt == self.reset_tile:                       # dead-lock escape
                agent = self.start
                boxes = set(self.box_start)                  # positions only; keep mask/paid
            elif nxt == self.exit:
                r += self.EXIT_REWARD; done = True

        self._boxes, self._mask, self._paid = boxes, mask, paid
        self._state = (agent[0], agent[1], tuple(sorted(boxes)), mask, tuple(paid))
        return self._state, r, done

    def is_success(self):
        return (self._state[0], self._state[1]) == self.exit

    def render_frame(self):
        return {"agent": (self._state[0], self._state[1]),
                "boxes": list(self._boxes), "mask": self._mask,
                "door_open": self.door_open(self._boxes)}

    def render_meta(self):
        return dict(kind="sokoban", size=self.SIZE, start=self.start, goal=self.exit,
                    walls=self.walls, doors=self.doors, slippery=self.slippery,
                    buttons=self.buttons, box_start=self.box_start,
                    bonuses=self.bonus_at, bonus_bit=self.bonus_bit,
                    negatives=self.negatives, neg_reward=self.NEG_REWARD,
                    reset_tile=self.reset_tile, button_reward=self.BUTTON_REWARD,
                    bonus_reward=self.BONUS_REWARD, exit_reward=self.EXIT_REWARD,
                    traps=set(), cliff=set())


# --------------------------------------------------------------------------- #
# Room 4 — The Hovercar Garage (Fast & Furious) — Function Approximation
# --------------------------------------------------------------------------- #
# 9 discrete VELOCITY actions (vx, vy) in {-1,0,1}^2 — the car sets its velocity.
VEL_ACTIONS = [(vx, vy) for vx in (-1, 0, 1) for vy in (-1, 0, 1)]
ACCEL_ACTIONS = VEL_ACTIONS                                   # backwards-compat alias


class Room4Garage:
    """Continuous 2-D time-trial on a winding Tokyo-Drift canyon road.

    The LAYOUT is an occupancy grid ('#' = black wall, '.'/'S'/'T' = drivable
    road) mapped onto the continuous 10×10 m space — but the car itself moves
    CONTINUOUSLY and diagonally; the grid is only how the track is defined and
    drawn (rendered without grid lines, so it reads as a smooth track).

    State = [X, Y, Vx, Vy]; nine discrete VELOCITY actions (vx, vy) ∈ {-1,0,1}²;
    every step advances DT = 0.02 s, so the car moves v·DT metres.  The car has
    a hitbox and CANNOT pass through walls: a move whose hitbox would touch a
    wall is blocked (the car stops) and costs COLLISION_PENALTY that step.
    Reaching the finish cell pays FINISH_BONUS / total_time (total_time =
    steps · DT), so a faster lap scores higher.  Solved with tile-coded linear
    function approximation; a wave-front potential over the free cells provides
    follow-the-road shaping (φ = −metres-left-to-finish).
    """

    NAME = "Room 4 · Tokyo Drift Canyon"
    MOVIE = "The Fast and the Furious: Tokyo Drift (2006)"
    ALGO = "Semi-gradient Q-Learning + tile-coding (Function Approximation)"
    DT = 0.02
    COLLISION_PENALTY = -10.0        # mild nudge (walls already block movement) — a huge
    FINISH_BONUS = 1000.0            # penalty just made the car too scared to move / diverge
    HITBOX = 0.25                 # car half-size (metres) — fits the 1 m lanes

    # 10×10 occupancy grid (each cell = 1×1 m).  '#' wall · '.' road ·
    # 'S' start · 'T' finish.  Row 0 = top.  This is only the layout definition;
    # the car itself moves continuously (see the class docstring).
    TRACK_GRID = [
        "#......###",
        "#.###...##",
        "#....#...#",
        "####..#..#",
        "#....#...#",
        "...##.....",
        "..##...##.",
        ".##...##..",
        ".##..##...",
        "S#T..#####",
    ]

    def __init__(self, max_steps=2500, shaping=True, shaping_coef=15.0,
                 shaping_gamma=1.0, seed=None, **_legacy):
        self.max_steps = int(max_steps)
        self.shaping = bool(shaping)
        self.shaping_coef = float(shaping_coef)
        self.shaping_gamma = float(shaping_gamma)
        self.bounds = (0.0, 10.0, 0.0, 10.0)                 # xmin,xmax,ymin,ymax
        self.n_actions = 9
        self.actions = list(range(9))
        self.state_low = np.array([0.0, 0.0, -1.0, -1.0])
        self.state_high = np.array([10.0, 10.0, 1.0, 1.0])
        self._build_track()
        self._build_maps()
        self._succeeded = False
        self.reset()

    # ---- grid geometry --------------------------------------------------- #
    def _build_track(self):
        g = self.TRACK_GRID
        self.NROW, self.NCOL = len(g), len(g[0])
        self.cw, self.ch = 10.0 / self.NCOL, 10.0 / self.NROW
        self._start_cell = self._finish_cell = None
        for r, row in enumerate(g):
            for c, ce in enumerate(row):
                if ce == "S":   self._start_cell = (r, c)
                elif ce == "T": self._finish_cell = (r, c)
        self.start = self._cell_center(*self._start_cell)
        self.finish = self._cell_center(*self._finish_cell)

    def _cell_center(self, r, c):
        return ((c + 0.5) * self.cw, (self.NROW - 1 - r + 0.5) * self.ch)

    def _cell_of(self, x, y):
        c = min(self.NCOL - 1, max(0, int(x / self.cw)))
        r = min(self.NROW - 1, max(0, self.NROW - 1 - int(y / self.ch)))
        return r, c

    def _is_wall_pt(self, x, y):
        if x < 0.0 or x > 10.0 or y < 0.0 or y > 10.0:
            return True
        r, c = self._cell_of(x, y)
        return self.TRACK_GRID[r][c] == "#"

    def _collides(self, x, y):
        """True if the car's hitbox at (x, y) touches a wall or the border."""
        h = self.HITBOX
        return any(self._is_wall_pt(x + dx, y + dy)
                   for dx in (-h, 0.0, h) for dy in (-h, 0.0, h))

    def _build_maps(self, res=0.1):
        """Wave-front BFS over the free cells → −(metres left to finish), for
        potential-based shaping.  Pre-computed on a fine grid → O(1) look-ups."""
        n = int(round(10.0 / res)); self._res, self._n = res, n
        blocked = np.zeros((n, n), dtype=bool)               # [ix, iy]
        for ix in range(n):
            x = (ix + 0.5) * res
            for iy in range(n):
                blocked[ix, iy] = self._is_wall_pt(x, (iy + 0.5) * res)
        INF = 10 ** 9
        dist = np.full((n, n), INF, dtype=float)
        dq = deque()
        fr, fc = self._finish_cell
        for ix in range(n):
            for iy in range(n):
                if not blocked[ix, iy] and \
                        self._cell_of((ix + 0.5) * res, (iy + 0.5) * res) == (fr, fc):
                    dist[ix, iy] = 0.0; dq.append((ix, iy))
        while dq:                                            # BFS from the finish
            ix, iy = dq.popleft(); d0 = dist[ix, iy]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < n and 0 <= jy < n and not blocked[jx, jy] \
                        and dist[jx, jy] > d0 + 1:
                    dist[jx, jy] = d0 + 1; dq.append((jx, jy))
        phi = -dist * res
        far = phi[dist < INF].min() if np.any(dist < INF) else 0.0
        phi[dist >= INF] = far - 5.0                          # walls / unreachable
        self._phi_map = phi

    def _phi(self, x, y):
        ix = min(self._n - 1, max(0, int(x / self._res)))
        iy = min(self._n - 1, max(0, int(y / self._res)))
        return float(self._phi_map[ix, iy])

    # ---- model-free interface -------------------------------------------- #
    def reset(self):
        self.X, self.Y = self.start
        self.Vx = self.Vy = 0.0
        self.t = 0
        self._succeeded = False
        return self._obs()

    def _obs(self):
        return np.array([self.X, self.Y, self.Vx, self.Vy], dtype=float)

    def step(self, a):
        ox, oy = self.X, self.Y
        vx, vy = VEL_ACTIONS[a]
        nx, ny = self.X + vx * self.DT, self.Y + vy * self.DT
        self.t += 1

        if self._collides(nx, ny):                           # can't drive through walls
            self.Vx = self.Vy = 0.0                          # stopped by the barrier
            reward = self.COLLISION_PENALTY
        else:
            self.X, self.Y = nx, ny
            self.Vx, self.Vy = float(vx), float(vy)
            reward = 0.0

        # finish cell → +1000 / total_time (faster lap ⇒ bigger reward)
        if self._cell_of(self.X, self.Y) == self._finish_cell:
            self._succeeded = True
            total_time = self.t * self.DT
            return self._obs(), self.FINISH_BONUS / total_time, True

        if self.shaping:                                     # potential-based shaping
            reward += self.shaping_coef * (
                self.shaping_gamma * self._phi(self.X, self.Y) - self._phi(ox, oy))
        if self.t >= self.max_steps:
            return self._obs(), reward, True                 # timeout (not success)
        return self._obs(), reward, False

    def is_success(self):
        return self._succeeded

    def render_frame(self):
        return {"agent": (self.X, self.Y), "vel": (self.Vx, self.Vy)}

    def render_meta(self):
        return dict(kind="track", bounds=self.bounds, grid=list(self.TRACK_GRID),
                    ncol=self.NCOL, nrow=self.NROW, start=self.start,
                    finish=self.finish, hitbox=self.HITBOX,
                    collision_penalty=self.COLLISION_PENALTY)


# --------------------------------------------------------------------------- #
# Room 5 — The Space Escape (Star Wars) — dynamic obstacles + partial observ.
# --------------------------------------------------------------------------- #
class Room5SpaceEscape:
    """Star-Wars gauntlet.  A ship that moves only on X must survive S seconds
    against a DYNAMIC stream of falling obstacles — each wave spawns UP TO
    SPAWN_MAX obstacles at once (random position + type; no on-screen cap).  It
    carries N_SHOTS laser shots; a shot damages the nearest obstacle aligned
    with its column (± SHOT_DX leniency), and destroying one pays +DESTROY_REWARD.
    Partial observation: beyond its own (position, velocity) the ship senses
    obstacles only up to `vision` metres ahead, centre-to-centre — a deliberate
    POMDP solved with tabular Q-Learning over a discretised look-ahead sensor.

    The ship has HEALTH hit-points; each collision costs one.  Running out of
    health ENDS the episode as a TERMINAL failure (DEATH_REWARD = −100).
    Surviving the full S seconds ENDS it as a TERMINAL success
    (SURVIVE_REWARD = +100).

    Obstacles (all width 0.5):
      • spaceship debris — fall speed 6, drifts sideways as it falls, 1 shot, −25 on hit
      • asteroid         — fall speed 3, falls straight,               1 shot, −15 on hit
    Each shot has a SHOT_COOLDOWN-step cooldown; destroying an obstacle pays +DESTROY_REWARD.
    """

    NAME = "Room 5 · The Space Escape"
    MOVIE = "Star Wars (1977)"
    ALGO = "Tabular Q-Learning over a partial-observation sensor"
    DT = 0.02
    AGENT_Y = 1.5
    SPAWN_MAX = 4            # up to this many obstacles spawn per wave (no on-screen cap)
    HEALTH = 3              # hit-points; hitting 0 = failure (episode still finishes)
    N_SHOTS = 4
    SHOT_COOLDOWN = 5         # steps the agent must wait between shots
    DEBRIS_MOVE_PROB = 0.30  # per-step chance a debris also drifts left/right (diagonal fall)
    EDGE_PENALTY = -5.0      # per-step penalty for hugging a wall
    EDGE_MARGIN = 0.5        # how close to a wall counts as "hugging the edge"
    OBS_WIDTH = 0.5
    COLLIDE_DX = 0.5           # |Δx| for an obstacle to hit the ship
    SHOT_DX = 0.5             # |Δx| leniency for a laser to connect
    DESTROY_REWARD = 30.0
    DEATH_REWARD = -100.0    # terminal: health hits 0
    SURVIVE_REWARD = 100.0   # terminal: survived the full S seconds
    TYPES = {                 # length, fall speed, hit-points (shots to kill), collision reward
        "debris":   dict(length=0.5, speed=6.0, hp=1, collide=-25.0),
        "asteroid": dict(length=0.5, speed=3.0, hp=1, collide=-15.0),
    }

    def __init__(self, vision=3.0, spawn_every=15, v_max=3.0,
                 max_steps=500, seed=None, speed_mult=1.0, debris_prob=0.5):
        self.vision = float(vision)
        self.spawn_every = int(spawn_every)
        self.v_max = float(v_max)
        self.max_steps = int(max_steps)
        self.speed_mult = float(speed_mult)   # global fall-speed scale (random rooms vary it)
        self.debris_prob = float(debris_prob) # P(debris) vs asteroid at spawn
        self.rng = np.random.default_rng(seed)
        self.n_actions = 4                    # 0 left · 1 stay · 2 right · 3 shoot
        self.actions = [0, 1, 2, 3]
        self._survived = False
        self.reset()

    def reset(self):
        self.X = 5.0
        self.Vx = 0.0
        self.obstacles = []                   # dicts: x, y(centre), type, hp
        self.shots = self.N_SHOTS
        self.health = self.HEALTH             # hit-points; 0 = failure
        self.cooldown = 0                     # steps until the laser can fire again
        self._fired = False                  # did a shot fire THIS step? (for the viz)
        self.t = 0
        self.collisions = 0
        self.destroyed = 0
        self._survived = False
        return self._obs()

    def random_room(self, seed=None):
        """Fresh episode with a new random obstacle stream (test button)."""
        self.rng = np.random.default_rng(seed)
        return self.reset()

    def _spawn(self):
        """A wave: spawn 1..SPAWN_MAX obstacles at once (random x + type)."""
        for _ in range(int(self.rng.integers(1, self.SPAWN_MAX + 1))):
            kind = "debris" if self.rng.random() < self.debris_prob else "asteroid"
            self.obstacles.append(dict(x=float(self.rng.uniform(0.5, 9.5)), y=10.0,
                                       type=kind, hp=self.TYPES[kind]["hp"]))

    # ---- shooting -------------------------------------------------------- #
    def _shoot(self):
        """Fire a laser up the ship's column, if ready (off cooldown + ammo left).
        Damages the nearest column-aligned obstacle, +DESTROY_REWARD on a kill.
        Firing spends a shot and starts a SHOT_COOLDOWN-step cooldown."""
        if self.cooldown > 0 or self.shots <= 0:
            return 0.0                        # not ready → no shot
        self.shots -= 1
        self.cooldown = self.SHOT_COOLDOWN
        self._fired = True
        target, best_dy = None, np.inf
        for o in self.obstacles:                             # only what the ship can SEE
            dy = o["y"] - self.AGENT_Y
            if 0.0 < dy <= self.vision and abs(o["x"] - self.X) <= self.SHOT_DX and dy < best_dy:
                best_dy, target = dy, o
        if target is None:
            return 0.0                        # missed — laser still spent
        target["hp"] -= 1
        if target["hp"] <= 0:
            self.obstacles.remove(target)
            self.destroyed += 1
            return self.DESTROY_REWARD
        return 0.0

    def _ready(self):
        return self.cooldown == 0 and self.shots > 0

    def valid_actions(self, obs):
        """Mask the shoot action while on cooldown / out of ammo (obs[2] = ready)."""
        return [0, 1, 2, 3] if obs[2] else [0, 1, 2]

    # ---- discretised look-ahead sensor ----------------------------------- #
    def _lane(self, lo, hi):
        """Nearest obstacle whose Δx is in [lo,hi) and within `vision` ahead:
        0 clear · 1 wall (off-board) · 2/3 debris far/near · 4/5 asteroid far/near."""
        if self.X + (lo + hi) / 2.0 < 0.0 or self.X + (lo + hi) / 2.0 > 10.0:
            return 1
        best_dy, best_type = None, None
        for o in self.obstacles:
            if lo <= (o["x"] - self.X) < hi:
                dy = o["y"] - self.AGENT_Y
                if 0.0 <= dy <= self.vision and (best_dy is None or dy < best_dy):
                    best_dy, best_type = dy, o["type"]
        if best_dy is None:
            return 0
        near = best_dy < self.vision * 0.5
        if best_type == "debris":
            return 3 if near else 2
        return 5 if near else 4

    def _obs(self):
        vx = 0 if abs(self.Vx) < 0.5 else (1 if self.Vx > 0 else 2)
        return (vx, int(self.shots), int(self._ready()),
                self._lane(-1.5, -0.5), self._lane(-0.5, 0.5), self._lane(0.5, 1.5))

    def encode(self, obs):
        return tuple(obs)                     # the sensor is already discrete

    def step(self, a):
        reward = 0.0
        self._fired = False
        if a == 3:
            reward += self._shoot()
        else:
            self.Vx = float(np.clip(self.Vx + (a - 1), -self.v_max, self.v_max))
        self.X = float(np.clip(self.X + self.Vx * self.DT, 0.0, 10.0))
        if self.X < self.EDGE_MARGIN or self.X > 10.0 - self.EDGE_MARGIN:
            reward += self.EDGE_PENALTY       # discourage hugging a wall

        for o in self.obstacles:              # fall (per-type speed × room scale)
            fall = self.TYPES[o["type"]]["speed"] * self.speed_mult * self.DT
            o["y"] -= fall
            if o["type"] == "debris" and self.rng.random() < self.DEBRIS_MOVE_PROB:
                drift = fall if self.rng.random() < 0.5 else -fall   # 50/50 left/right
                o["x"] = float(np.clip(o["x"] + drift, 0.0, 10.0))

        survivors = []                        # collisions with the ship
        for o in self.obstacles:
            half = self.TYPES[o["type"]]["length"] / 2.0
            if (abs(o["x"] - self.X) <= self.COLLIDE_DX
                    and (o["y"] - half) <= self.AGENT_Y <= (o["y"] + half)):
                reward += self.TYPES[o["type"]]["collide"]
                self.collisions += 1          # obstacle consumed on impact
                self.health = max(0, self.health - 1)
            elif o["y"] > 0.0:                # keep only obstacles still on the board
                survivors.append(o)
        self.obstacles = survivors
        self.t += 1

        if self.health <= 0:                  # out of health → TERMINAL failure (−100)
            return self._obs(), reward + self.DEATH_REWARD, True
        if not self._fired:                   # tick the shot cooldown down
            self.cooldown = max(0, self.cooldown - 1)
        if self.t >= self.max_steps:          # survived the full S seconds → TERMINAL success (+100)
            self._survived = True
            return self._obs(), reward + self.SURVIVE_REWARD, True
        if self.t % self.spawn_every == 0:    # dynamic spawn (up to SPAWN_MAX at once)
            self._spawn()
        return self._obs(), reward, False

    def is_success(self):
        # "escaped" = survived the full S seconds with health left (took < HEALTH hits)
        return self._survived and self.health > 0

    def render_frame(self):
        return {"agent": (self.X, self.AGENT_Y), "shots": self.shots,
                "fired": self._fired, "cooldown": self.cooldown, "health": self.health,
                "collisions": self.collisions, "destroyed": self.destroyed,
                "obstacles": [(o["x"], o["y"], o["type"], o["hp"]) for o in self.obstacles]}

    def render_meta(self):
        return dict(kind="space", bounds=(0.0, 10.0, 0.0, 10.0),
                    agent_y=self.AGENT_Y, obs_width=self.OBS_WIDTH,
                    vision=self.vision, n_shots=self.N_SHOTS, health_max=self.HEALTH)


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
