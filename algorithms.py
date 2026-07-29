"""RL algorithms for the Escape Room.

    * ValueIteration  — Dynamic Programming with a KNOWN model (Room 1).
    * Sarsa           — on-policy TD control (Room 2).
    * QLearning       — off-policy TD control (Rooms 3 & 5; Room 5 discretises a
                        partial observation via env.encode).
    * LinearFAAgent   — semi-gradient Q-Learning over TILE-CODED features, i.e.
                        genuine function approximation for the continuous Room 4.

Every learner produces a lightweight *policy object* (``.action(raw_state) -> a``)
and periodic snapshots of it, so the app can replay "what the agent knew at each
stage of training" through the single uniform ``rollout`` helper below.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _argmax_random(q, rng):
    """argmax with random tie-breaking (avoids a systematic action bias)."""
    q = np.asarray(q, dtype=float)
    ties = np.flatnonzero(q == q.max())
    return int(ties[0]) if ties.size == 1 else int(rng.choice(ties))


def _milestones(episodes, k):
    """`k` evenly spaced episode indices in [0, episodes-1] for snapshots."""
    if episodes <= 1 or k <= 1:
        return set()
    return {int(p) for p in np.linspace(0, episodes - 1, k).astype(int)}


def _record_points(episodes, k):
    """Episode indices whose full trajectory we keep for the replay browser."""
    if episodes <= 0 or k <= 0:
        return set()
    k = min(k, episodes)
    return {int(p) for p in np.linspace(0, episodes - 1, k).astype(int)}


# --------------------------------------------------------------------------- #
# Policy objects (used for evaluation / replay)
# --------------------------------------------------------------------------- #
class DPPolicy:
    """Deterministic policy from Value Iteration: state -> action."""

    def __init__(self, policy):
        self.policy = policy

    def action(self, raw_state):
        return int(self.policy.get(raw_state, 0))


class TabularPolicy:
    """Greedy policy over a (snapshotted) tabular Q, restricted to each state's
    VALID (non-wall) actions. Unseen states act randomly but legally."""

    def __init__(self, Q, encode, n_actions, seed=0, valid_fn=None):
        self.Q = Q
        self.encode = encode
        self.n_actions = n_actions
        self.valid_fn = valid_fn
        self.rng = np.random.default_rng(seed)

    def action(self, raw_state):
        s = self.encode(raw_state)
        valid = self.valid_fn(raw_state) if self.valid_fn else list(range(self.n_actions))
        if s in self.Q:
            q = self.Q[s]
            vals = np.array([q[a] for a in valid])
            # random tie-break (deterministic argmax makes states dead-loop)
            ties = [valid[i] for i in np.flatnonzero(vals == vals.max())]
            return int(ties[0]) if len(ties) == 1 else int(self.rng.choice(ties))
        return int(self.rng.choice(valid))


class FAPolicy:
    """Greedy policy over tile-coded linear weights."""

    def __init__(self, w, coder):
        self.w = w
        self.coder = coder

    def action(self, raw_state):
        feats = self.coder.features(raw_state)
        return int(np.argmax(self.w[:, feats].sum(axis=1)))


# --------------------------------------------------------------------------- #
# Room 1 — Value Iteration (Dynamic Programming)
# --------------------------------------------------------------------------- #
class ValueIteration:
    def __init__(self, env, gamma=0.99, theta=1e-4):
        self.env = env
        self.model = env.build_model()
        self.actions = env.actions
        self.gamma = float(gamma)
        self.theta = float(theta)
        self.states = [s for s in env.states() if not env.is_terminal(s)]
        self._valid = getattr(env, "valid_actions", None)

    def acts(self, s):
        """Only the legal (non-wall) actions of state s."""
        return self._valid(s) if self._valid else self.actions

    def _q(self, s, a, V):
        return sum(p * (r + (0.0 if done else self.gamma * V[s2]))
                   for p, s2, r, done in self.model[s][a])

    def run(self, max_iter=5000, progress=None):
        V = {s: 0.0 for s in self.env.states()}   # terminals stay 0 (absorbing)
        deltas = []
        for it in range(max_iter):
            delta = 0.0
            for s in self.states:
                v_old = V[s]
                V[s] = max(self._q(s, a, V) for a in self.acts(s))
                delta = max(delta, abs(v_old - V[s]))
            deltas.append(delta)
            if progress:
                progress(it + 1, max_iter)
            if delta < self.theta:
                break
        policy = {}
        for s in self.states:
            acts = self.acts(s)
            policy[s] = int(acts[int(np.argmax([self._q(s, a, V) for a in acts]))])
        return dict(V=V, policy=policy, deltas=deltas,
                    iterations=len(deltas), final_policy=DPPolicy(policy))


# --------------------------------------------------------------------------- #
# Temporal-difference base (shared by SARSA and Q-Learning)
# --------------------------------------------------------------------------- #
class TDAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_k=0.0, epsilon_min=0.01, episodes=2000,
                 max_steps=300, optimistic_init=0.0, seed=None):
        self.env = env
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps0 = float(epsilon)
        self.eps_k = float(epsilon_k)          # LINEAR decrement per episode
        self.eps_min = float(epsilon_min)
        self.episodes = int(episodes)
        self.max_steps = int(max_steps)
        self.n_actions = env.n_actions
        self.encode = getattr(env, "encode", lambda s: s)
        self.valid = getattr(env, "valid_actions", None)   # action masking
        self.rng = np.random.default_rng(seed)
        # Optimistic initialisation: unseen state-actions look attractive, which
        # drives *systematic* exploration — essential in corridor mazes where
        # ε-greedy random walks almost never reach a distant reward.
        self.q0 = float(optimistic_init)
        self.Q = defaultdict(lambda: np.full(self.n_actions, self.q0))

    def valid_of(self, raw):
        return self.valid(raw) if self.valid else list(range(self.n_actions))

    def decay(self, eps):
        return max(self.eps_min, eps - self.eps_k)          # ε = ε₀ − K·t

    def _act(self, s, eps, valid):
        """ε-greedy restricted to the state's legal (non-wall) actions."""
        if self.rng.random() < eps:
            return int(valid[self.rng.integers(len(valid))])
        q = self.Q[s]
        if len(valid) == self.n_actions:            # fast path: nothing masked
            return _argmax_random(q, self.rng)
        best, ties = valid[0], [valid[0]]           # pure-Python argmax over ≤4 acts
        for a in valid[1:]:
            if q[a] > q[best]:
                best, ties = a, [a]
            elif q[a] == q[best]:
                ties.append(a)
        return int(ties[0]) if len(ties) == 1 else int(self.rng.choice(ties))

    def _snapshot(self):
        return TabularPolicy({s: q.copy() for s, q in self.Q.items()},
                             self.encode, self.n_actions, valid_fn=self.valid)


class Sarsa(TDAgent):
    """On-policy TD(0): update towards the action actually taken next."""

    def train(self, snapshots=6, progress=None, record=40):
        rewards, eps_hist, lengths, snaps, tapes, succ, treasure = [], [], [], [], [], [], []
        has_treasure = hasattr(self.env, "treasure_visited")
        milestones = _milestones(self.episodes, snapshots)
        rec_at = _record_points(self.episodes, record)
        eps = self.eps0
        for ep in range(self.episodes):
            raw = self.env.reset()
            s, valid = self.encode(raw), self.valid_of(raw)
            a = self._act(s, eps, valid)
            taping = ep in rec_at
            frames = [self.env.render_frame()] if taping else None
            acts, rews = ([], []) if taping else (None, None)
            done, total, steps = False, 0.0, 0
            while not done and steps < self.max_steps:
                raw2, r, done = self.env.step(a)
                if taping:
                    frames.append(self.env.render_frame()); acts.append(a); rews.append(r)
                s2, valid2 = self.encode(raw2), self.valid_of(raw2)
                a2 = self._act(s2, eps, valid2)
                target = r + (0.0 if done else self.gamma * self.Q[s2][a2])
                self.Q[s][a] += self.alpha * (target - self.Q[s][a])
                s, a, valid = s2, a2, valid2
                total += r
                steps += 1
            eps = self.decay(eps)
            rewards.append(total); eps_hist.append(eps); lengths.append(steps)
            succ.append(bool(self.env.is_success()))       # did this episode escape?
            if has_treasure:
                treasure.append(bool(self.env.treasure_visited()))
            if taping:
                tapes.append(dict(episode=ep, reward=total, steps=steps,
                                  success=self.env.is_success(), frames=frames,
                                  actions=acts, step_rewards=rews))
            if ep in milestones:
                snaps.append((ep, self._snapshot()))
            if progress:
                progress(ep + 1, self.episodes)
        final = self._snapshot()
        snaps.append((self.episodes, final))
        return dict(rewards=rewards, epsilons=eps_hist, lengths=lengths,
                    snapshots=snaps, final_policy=final, tapes=tapes, successes=succ,
                    treasure_visits=treasure)


class QLearning(TDAgent):
    """Off-policy TD(0): update towards the greedy next action (max)."""

    def train(self, snapshots=6, progress=None, record=40):
        rewards, eps_hist, lengths, snaps, tapes, succ, treasure = [], [], [], [], [], [], []
        has_treasure = hasattr(self.env, "treasure_visited")
        milestones = _milestones(self.episodes, snapshots)
        rec_at = _record_points(self.episodes, record)
        eps = self.eps0
        for ep in range(self.episodes):
            raw = self.env.reset()
            s, valid = self.encode(raw), self.valid_of(raw)
            taping = ep in rec_at
            frames = [self.env.render_frame()] if taping else None
            acts, rews = ([], []) if taping else (None, None)
            done, total, steps = False, 0.0, 0
            while not done and steps < self.max_steps:
                a = self._act(s, eps, valid)
                raw2, r, done = self.env.step(a)
                if taping:
                    frames.append(self.env.render_frame()); acts.append(a); rews.append(r)
                s2, valid2 = self.encode(raw2), self.valid_of(raw2)
                # bootstrap over the LEGAL actions of s2 (off-policy greedy max)
                nxt = 0.0 if done else max(self.Q[s2][aa] for aa in valid2)
                self.Q[s][a] += self.alpha * (r + self.gamma * nxt - self.Q[s][a])
                s, valid = s2, valid2
                total += r
                steps += 1
            eps = self.decay(eps)
            rewards.append(total); eps_hist.append(eps); lengths.append(steps)
            succ.append(bool(self.env.is_success()))       # did this episode escape?
            if has_treasure:
                treasure.append(bool(self.env.treasure_visited()))
            if taping:
                tapes.append(dict(episode=ep, reward=total, steps=steps,
                                  success=self.env.is_success(), frames=frames,
                                  actions=acts, step_rewards=rews))
            if ep in milestones:
                snaps.append((ep, self._snapshot()))
            if progress:
                progress(ep + 1, self.episodes)
        final = self._snapshot()
        snaps.append((self.episodes, final))
        return dict(rewards=rewards, epsilons=eps_hist, lengths=lengths,
                    snapshots=snaps, final_policy=final, tapes=tapes, successes=succ,
                    treasure_visits=treasure)


# --------------------------------------------------------------------------- #
# Room 4 — Tile coding + semi-gradient Q-Learning (Function Approximation)
# --------------------------------------------------------------------------- #
class TileCoder:
    """Classic tile coding over a bounded continuous space.  `n_tilings`
    overlapping grids, each `n_bins` per dimension, with asymmetric offsets."""

    def __init__(self, low, high, n_tilings=8, n_bins=8):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.n_tilings = int(n_tilings)
        self.n_bins = int(n_bins)
        self.dims = len(self.low)
        self.tile_width = (self.high - self.low) / self.n_bins
        disp = (2 * np.arange(self.dims) + 1)                 # asymmetric shifts
        # per-tiling, per-dim offset matrix  (n_tilings, dims)
        self.offsets = np.stack([((t * disp) % self.n_tilings / self.n_tilings) * self.tile_width 
                         for t in range(self.n_tilings)])
        self.card = self.n_bins + 1
        self.tiles_per_tiling = self.card ** self.dims
        self.n_features = self.n_tilings * self.tiles_per_tiling
        # mixed-radix weights + per-tiling base offsets (precomputed, vectorised)
        self.radix = self.card ** np.arange(self.dims - 1, -1, -1)
        self.tiling_base = np.arange(self.n_tilings) * self.tiles_per_tiling

    def features(self, s):
        """Return the `n_tilings` active feature indices (one per tiling)."""
        s = np.clip(np.asarray(s, dtype=float), self.low, self.high - 1e-9)
        coords = ((s - self.low + self.offsets) / self.tile_width).astype(int)
        np.clip(coords, 0, self.n_bins, out=coords)           # (n_tilings, dims)
        return self.tiling_base + coords.dot(self.radix)      # (n_tilings,) int array


class LinearFAAgent:
    def __init__(self, env, alpha=0.5, gamma=1.0, epsilon=0.1,
                 epsilon_k=0.0, epsilon_min=0.0, episodes=2000,
                 n_tilings=8, n_bins=8, optimistic_init=100.0,
                 max_steps=None, seed=None):
        self.env = env
        self.coder = TileCoder(env.state_low, env.state_high, n_tilings, n_bins)
        # Optimistic initialisation drives systematic exploration (each active
        # tiling contributes optimistic_init / n_tilings so Q(s,a) == optimistic_init).
        self.w = np.full((env.n_actions, self.coder.n_features),
                         float(optimistic_init) / n_tilings)
        self.alpha = float(alpha) / n_tilings                 # per-tiling step size
        self.gamma = float(gamma)
        self.eps0 = float(epsilon)
        self.eps_k = float(epsilon_k)          # LINEAR decrement per episode
        self.eps_min = float(epsilon_min)
        self.episodes = int(episodes)
        self.n_actions = env.n_actions
        self.max_steps = int(max_steps or env.max_steps)
        self.rng = np.random.default_rng(seed)

    def _q(self, feats):
        return self.w[:, feats].sum(axis=1)

    def train(self, snapshots=6, progress=None, record=25):
        rewards, lengths, snaps, tapes, successes, wall_hits = [], [], [], [], [], []
        has_wall_hits = hasattr(self.env, "wall_hits")
        milestones = _milestones(self.episodes, snapshots)
        rec_at = _record_points(self.episodes, record)
        eps = self.eps0
        for ep in range(self.episodes):
            feats = self.coder.features(self.env.reset())
            taping = ep in rec_at
            frames = [self.env.render_frame()] if taping else None
            acts, rews = ([], []) if taping else (None, None)
            done, total, steps = False, 0.0, 0
            while not done and steps < self.max_steps:
                if self.rng.random() < eps:
                    a = int(self.rng.integers(self.n_actions))
                else:
                    a = _argmax_random(self._q(feats), self.rng)
                s2, r, done = self.env.step(a)
                if taping:
                    frames.append(self.env.render_frame()); acts.append(a); rews.append(r)
                feats2 = self.coder.features(s2)
                q_sa = self.w[a, feats].sum()
                target = r + (0.0 if done else self.gamma * self._q(feats2).max())
                self.w[a, feats] += self.alpha * (target - q_sa)
                feats = feats2
                total += r
                steps += 1
            eps = max(self.eps_min, eps - self.eps_k)          # ε = ε₀ − K·t
            rewards.append(total); lengths.append(steps)
            successes.append(bool(self.env.is_success()))      # reached the finish?
            if has_wall_hits:
                wall_hits.append(int(self.env.wall_hits))
            if taping:
                tapes.append(dict(episode=ep, reward=total, steps=steps,
                                  success=self.env.is_success(), frames=frames,
                                  actions=acts, step_rewards=rews))
            if ep in milestones:
                snaps.append((ep, FAPolicy(self.w.copy(), self.coder)))
            if progress:
                progress(ep + 1, self.episodes)
        final = FAPolicy(self.w.copy(), self.coder)
        snaps.append((self.episodes, final))
        return dict(rewards=rewards, lengths=lengths, epsilons=None,
                    snapshots=snaps, final_policy=final, tapes=tapes,
                    successes=successes, wall_hits=wall_hits)


# --------------------------------------------------------------------------- #
# Deep Q-Network (DQN) — neural-net function approximation (Room 4 option)
# torch is imported lazily so the other rooms never depend on it.
# --------------------------------------------------------------------------- #
def _build_qnet(in_dim, n_actions, hidden):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, n_actions))


class DQNPolicy:
    """Greedy policy over a trained Q-network (state normalised to [0,1])."""

    def __init__(self, net, low, span):
        self.net = net
        self.low = np.asarray(low, dtype=float)
        self.span = np.asarray(span, dtype=float)

    def action(self, raw_state):
        import torch
        x = ((np.asarray(raw_state, dtype=float) - self.low) / self.span).astype(np.float32)
        with torch.no_grad():
            return int(self.net(torch.as_tensor(x)).argmax().item())


class DQNAgent:
    """Deep Q-Network: an MLP Q(s,·) trained with experience replay + a target
    network (Mnih et al. 2015).  Same train()/policy interface as the other
    learners so it drops straight into the app's train/replay pipeline."""

    def __init__(self, env, alpha=1e-3, gamma=0.99, epsilon=1.0, epsilon_k=0.0025,
                 epsilon_min=0.05, episodes=600, max_steps=None, hidden=128,
                 buffer=50000, batch=64, target_every=500, train_freq=1,
                 warmup=1000, seed=None):
        import torch
        self.env = env
        self.n_actions = env.n_actions
        self.low = np.asarray(env.state_low, dtype=float)
        high = np.asarray(env.state_high, dtype=float)
        self.span = np.where(high > self.low, high - self.low, 1.0)
        self.gamma = float(gamma)
        self.eps0, self.eps_k, self.eps_min = float(epsilon), float(epsilon_k), float(epsilon_min)
        self.episodes = int(episodes)
        self.max_steps = int(max_steps or env.max_steps)
        self.batch, self.target_every = int(batch), int(target_every)
        self.train_freq, self.warmup = int(train_freq), int(warmup)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(int(seed or 0))
        self.q = _build_qnet(len(self.low), self.n_actions, hidden)
        self.tgt = _build_qnet(len(self.low), self.n_actions, hidden)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=float(alpha))
        self._buf, self._cap, self._pos = [], int(buffer), 0

    def _norm(self, s):
        return ((np.asarray(s, dtype=float) - self.low) / self.span).astype(np.float32)

    def _store(self, tr):
        if len(self._buf) < self._cap:
            self._buf.append(tr)
        else:
            self._buf[self._pos] = tr
        self._pos = (self._pos + 1) % self._cap

    def _policy(self):
        import copy
        return DQNPolicy(copy.deepcopy(self.q).eval(), self.low, self.span)

    def _learn(self, huber):
        import torch
        idx = self.rng.integers(0, len(self._buf), size=self.batch)
        b = [self._buf[i] for i in idx]
        s = torch.as_tensor(np.stack([t[0] for t in b]))
        a = torch.as_tensor(np.array([t[1] for t in b]), dtype=torch.long).unsqueeze(1)
        r = torch.as_tensor(np.array([t[2] for t in b], dtype=np.float32)).unsqueeze(1)
        s2 = torch.as_tensor(np.stack([t[3] for t in b]))
        d = torch.as_tensor(np.array([t[4] for t in b], dtype=np.float32)).unsqueeze(1)
        q = self.q(s).gather(1, a)
        with torch.no_grad():
            target = r + self.gamma * (1.0 - d) * self.tgt(s2).max(1, keepdim=True)[0]
        loss = huber(q, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

    def train(self, snapshots=6, progress=None, record=25):
        import torch
        rewards, lengths, snaps, tapes, eps_hist, succ = [], [], [], [], [], []
        milestones = _milestones(self.episodes, snapshots)
        rec_at = _record_points(self.episodes, record)
        huber = torch.nn.SmoothL1Loss()
        eps, gstep = self.eps0, 0
        for ep in range(self.episodes):
            s = self._norm(self.env.reset())
            taping = ep in rec_at
            frames = [self.env.render_frame()] if taping else None
            acts, rews = ([], []) if taping else (None, None)
            done, total, steps = False, 0.0, 0
            while not done and steps < self.max_steps:
                if self.rng.random() < eps:
                    a = int(self.rng.integers(self.n_actions))
                else:
                    with torch.no_grad():
                        a = int(self.q(torch.as_tensor(s)).argmax().item())
                s2raw, r, done = self.env.step(a)
                s2 = self._norm(s2raw)
                if taping:
                    frames.append(self.env.render_frame()); acts.append(a); rews.append(r)
                self._store((s, a, float(r), s2, float(done)))
                s = s2; total += r; steps += 1; gstep += 1
                if len(self._buf) >= max(self.batch, self.warmup) and gstep % self.train_freq == 0:
                    self._learn(huber)
                if gstep % self.target_every == 0:
                    self.tgt.load_state_dict(self.q.state_dict())
            eps = max(self.eps_min, eps - self.eps_k)
            rewards.append(total); lengths.append(steps); eps_hist.append(eps)
            succ.append(bool(self.env.is_success()))
            if taping:
                tapes.append(dict(episode=ep, reward=total, steps=steps,
                                  success=self.env.is_success(), frames=frames,
                                  actions=acts, step_rewards=rews))
            if ep in milestones:
                snaps.append((ep, self._policy()))
            if progress:
                progress(ep + 1, self.episodes)
        final = self._policy()
        snaps.append((self.episodes, final))
        return dict(rewards=rewards, lengths=lengths, epsilons=eps_hist,
                    snapshots=snaps, final_policy=final, tapes=tapes, successes=succ)


# --------------------------------------------------------------------------- #
# Uniform greedy rollout (drives the Episode Replay tab for every room)
# --------------------------------------------------------------------------- #
def rollout(env, policy, max_steps=600):
    """Run one greedy episode; return recorded frames + per-step log + stats."""
    s = env.reset()
    frames = [env.render_frame()]
    acts, rews = [], []
    total, done, steps = 0.0, False, 0
    while not done and steps < max_steps:
        a = policy.action(s)
        s, r, done = env.step(a)
        frames.append(env.render_frame()); acts.append(int(a)); rews.append(r)
        total += r
        steps += 1
    return dict(frames=frames, reward=total, steps=steps,
                success=env.is_success(), actions=acts, step_rewards=rews)
