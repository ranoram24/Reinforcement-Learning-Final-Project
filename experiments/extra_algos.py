"""Extra RL algorithms for the Room 5 shoot-out (STANDALONE — not used by the app).

From-scratch, deliberately compact implementations of the policy-gradient /
actor-critic / model-based families from the comparison table, so they can be
benchmarked against the tabular + value-FA learners already in algorithms.py:

    • REINFORCE  — Monte-Carlo policy gradient
    • A2C        — advantage actor-critic (per-episode)
    • PPO        — clipped actor-critic with GAE
    • Dyna-Q     — tabular Q-Learning + a learned one-step model + planning

The neural ones (REINFORCE/A2C/PPO) train on a continuous observation and return
a greedy `NetPolicy`; Dyna-Q trains on the discrete sensor and returns a greedy
tabular policy.  All expose `.action(raw_obs)` so they drop into `rollout()`.

These are teaching-grade baselines (small nets, modest budgets) — good enough to
compare, not tuned to their published best.
"""
from collections import defaultdict

import numpy as np


def _mlp(in_dim, out_dim, hidden=128):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, out_dim))


def _norm_fns(env):
    low = np.asarray(env.state_low, dtype=float)
    high = np.asarray(env.state_high, dtype=float)
    span = np.where(high > low, high - low, 1.0)
    return low, span


class NetPolicy:
    """Greedy policy over a softmax actor network."""

    def __init__(self, net, low, span):
        self.net, self.low, self.span = net, np.asarray(low, float), np.asarray(span, float)

    def action(self, raw):
        import torch
        x = ((np.asarray(raw, dtype=float) - self.low) / self.span).astype(np.float32)
        with torch.no_grad():
            return int(self.net(torch.as_tensor(x)).argmax().item())


# --------------------------------------------------------------------------- #
def train_reinforce(make_env, episodes=2500, lr=1e-3, gamma=0.97, hidden=128, seed=0):
    import torch
    torch.manual_seed(seed)
    env = make_env(seed)
    low, span = _norm_fns(env)

    def nrm(o):
        return torch.as_tensor(((np.asarray(o, float) - low) / span).astype(np.float32))

    pol = _mlp(len(low), env.n_actions, hidden)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    for _ in range(episodes):
        o = env.reset(); logps, rews = [], []
        done, steps = False, 0
        while not done and steps < env.max_steps:
            dist = torch.distributions.Categorical(logits=pol(nrm(o)))
            a = dist.sample(); logps.append(dist.log_prob(a))
            o, r, done = env.step(int(a.item())); rews.append(r); steps += 1
        G, returns = 0.0, []
        for r in reversed(rews):
            G = r + gamma * G; returns.append(G)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -(torch.stack(logps) * returns).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return NetPolicy(pol, low, span)


def train_a2c(make_env, episodes=2500, lr=1e-3, gamma=0.97, hidden=128, seed=0):
    import torch
    torch.manual_seed(seed)
    env = make_env(seed)
    low, span = _norm_fns(env)

    def nrm(o):
        return torch.as_tensor(((np.asarray(o, float) - low) / span).astype(np.float32))

    actor, critic = _mlp(len(low), env.n_actions, hidden), _mlp(len(low), 1, hidden)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    for _ in range(episodes):
        o = env.reset(); logps, vals, rews, ents = [], [], [], []
        done, steps = False, 0
        while not done and steps < env.max_steps:
            x = nrm(o); dist = torch.distributions.Categorical(logits=actor(x))
            a = dist.sample()
            logps.append(dist.log_prob(a)); ents.append(dist.entropy())
            vals.append(critic(x).squeeze())
            o, r, done = env.step(int(a.item())); rews.append(r); steps += 1
        G, returns = 0.0, []
        for r in reversed(rews):
            G = r + gamma * G; returns.append(G)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        vals, logps, ents = torch.stack(vals), torch.stack(logps), torch.stack(ents)
        adv = returns - vals.detach()
        loss = -(logps * adv).sum() - 0.01 * ents.sum() + 0.5 * ((returns - vals) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return NetPolicy(actor, low, span)


def train_ppo(make_env, updates=250, steps_per=2000, epochs=4, minibatch=256,
              lr=3e-4, gamma=0.97, lam=0.95, clip=0.2, hidden=128, seed=0):
    import torch
    torch.manual_seed(seed)
    env = make_env(seed)
    low, span = _norm_fns(env)

    def nrm(o):
        return ((np.asarray(o, float) - low) / span).astype(np.float32)

    actor, critic = _mlp(len(low), env.n_actions, hidden), _mlp(len(low), 1, hidden)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    o, steps = env.reset(), 0
    for _ in range(updates):
        O, A, LP, R, D, V = [], [], [], [], [], []
        for _ in range(steps_per):
            xo = nrm(o); x = torch.as_tensor(xo)
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=actor(x))
                a = dist.sample(); v = float(critic(x).squeeze())
            O.append(xo); A.append(int(a)); LP.append(float(dist.log_prob(a))); V.append(v)
            o, r, dn = env.step(int(a)); R.append(r); D.append(dn); steps += 1
            if dn or steps >= env.max_steps:
                o, steps = env.reset(), 0
        with torch.no_grad():
            last_v = float(critic(torch.as_tensor(nrm(o))).squeeze())
        adv = np.zeros(len(R), dtype=np.float32); gae = 0.0
        for t in reversed(range(len(R))):
            nextv = last_v if t == len(R) - 1 else V[t + 1]
            nonterm = 0.0 if D[t] else 1.0
            delta = R[t] + gamma * nextv * nonterm - V[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + np.asarray(V, dtype=np.float32)
        O = torch.as_tensor(np.asarray(O)); A = torch.as_tensor(np.asarray(A))
        LP = torch.as_tensor(np.asarray(LP, dtype=np.float32))
        adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8))
        ret_t = torch.as_tensor(ret)
        idx = np.arange(len(R))
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s0 in range(0, len(R), minibatch):
                mb = idx[s0:s0 + minibatch]
                dist = torch.distributions.Categorical(logits=actor(O[mb]))
                ratio = torch.exp(dist.log_prob(A[mb]) - LP[mb])
                a1 = ratio * adv_t[mb]
                a2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[mb]
                ploss = -torch.min(a1, a2).mean()
                vloss = ((ret_t[mb] - critic(O[mb]).squeeze()) ** 2).mean()
                loss = ploss + 0.5 * vloss - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()
    return NetPolicy(actor, low, span)


# --------------------------------------------------------------------------- #
class _DynaPolicy:
    def __init__(self, Q, env):
        self.Q, self.env, self.nA = Q, env, env.n_actions

    def action(self, raw):
        s = tuple(raw)
        valid = self.env.valid_actions(s) if hasattr(self.env, "valid_actions") else list(range(self.nA))
        qs = self.Q.get(s)
        return int(valid[0]) if qs is None else int(max(valid, key=lambda a: qs[a]))


def train_dyna_q(make_env, episodes=4000, alpha=0.1, gamma=0.95, eps0=1.0,
                 eps_k=0.0004, eps_min=0.05, planning=5, seed=0):
    """Tabular Q-Learning on the discrete sensor + a learned one-step model with
    `planning` simulated updates per real step (Sutton's Dyna-Q)."""
    env = make_env(seed)
    nA = env.n_actions
    Q = defaultdict(lambda: np.zeros(nA))
    model, seen = {}, []
    rng = np.random.default_rng(seed); eps = eps0
    va = getattr(env, "valid_actions", None)
    for _ in range(episodes):
        s = tuple(env.reset()); done, steps = False, 0
        while not done and steps < env.max_steps:
            valid = va(s) if va else list(range(nA))
            a = int(rng.choice(valid)) if rng.random() < eps else int(max(valid, key=lambda aa: Q[s][aa]))
            raw2, r, done = env.step(a); s2 = tuple(raw2)
            Q[s][a] += alpha * (r + (0.0 if done else gamma * Q[s2].max()) - Q[s][a])
            if (s, a) not in model:
                seen.append((s, a))
            model[(s, a)] = (r, s2, done)
            for _ in range(planning):                      # plan on the learned model
                ps, pa = seen[int(rng.integers(len(seen)))]
                pr, ps2, pd = model[(ps, pa)]
                Q[ps][pa] += alpha * (pr + (0.0 if pd else gamma * Q[ps2].max()) - Q[ps][pa])
            s = s2; steps += 1
        eps = max(eps_min, eps - eps_k)
    return _DynaPolicy(Q, make_env(seed))
