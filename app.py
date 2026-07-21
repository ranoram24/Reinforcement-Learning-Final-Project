"""Hezki the Dog vs. Men in Black — an RL Escape Room.

Streamlit entry point.  Hezki (a dog) escapes Agent J through five Hollywood
movie sets of increasing difficulty, each solved by a different RL algorithm:

    Room 1  Frozen Archive  (Ice Age)          Value Iteration / DP
    Room 2  Dark Temple      (Indiana Jones)    SARSA
    Room 3  Cloning Lab      (The Matrix)       Q-Learning (Cliff Walking)
    Room 4  Hovercar Garage  (Fast & Furious)   Tile-coding Function Approximation
    Room 5  Space Escape     (Star Wars)        Q-Learning + partial observability

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

import algorithms as A
import environments as E
import utils as U

st.set_page_config(page_title="Hezki vs. MIB — RL Escape Room",
                   page_icon="🐕", layout="wide")


def embed(html, height):
    """Render self-contained HTML (scripts included) in an isolated iframe.
    Uses the modern `st.iframe` when available, else the classic component."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:                                                    # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=height)

# --------------------------------------------------------------------------- #
# Room catalogue (drives the difficulty progression selector)
# --------------------------------------------------------------------------- #
ROOMS = {
    "room1": dict(label="Frozen Archive", emoji="❄️", stars=1, kind="dp",
                  movie="Ice Age (2002)", algo="Value Iteration (Dynamic Programming)",
                  plot="The MIB archive is frozen solid. Agent J's patrols are known, "
                       "so Hezki plans the perfect escape with a full model of the world."),
    "room2": dict(label="Dark Temple", emoji="🏛️", stars=2, kind="grid",
                  movie="Raiders of the Lost Ark (1981)", algo="SARSA (on-policy TD)",
                  plot="Hezki falls into an ancient temple. He must grab the golden idol "
                       "🏆 to open the stone gate, dodge spike pits that hurl him back to "
                       "the start — and never touch the pressure plate, which wakes a "
                       "boulder 🪨 that retraces his own trail."),
    "room3": dict(label="Cloning Lab", emoji="🕶️", stars=3, kind="grid",
                  movie="The Matrix (1999)", algo="Q-Learning (off-policy TD)",
                  plot="Agent J clones himself like Agent Smith. Hezki must take a "
                       "dangerous, aggressive shortcut along the cliff of clones."),
    "room4": dict(label="Hovercar Garage", emoji="🏎️", stars=4, kind="fa",
                  movie="The Fast and the Furious (2001)",
                  algo="Tile-coding + semi-gradient Q-Learning (Function Approximation)",
                  plot="Hezki steals a hovercar. Continuous physics, momentum and two "
                       "parked cars force a high-speed weave to the exit."),
    "room5": dict(label="Space Escape", emoji="🚀", stars=5, kind="space",
                  movie="Star Wars (1977)", algo="Q-Learning over a partial-observation sensor",
                  plot="Hezki flies an escape pod. Agent J fires neuralyzer-drones. "
                       "Hezki only senses what's right in front of him — dodge or be erased."),
}
ORDER = ["room1", "room2", "room3", "room4", "room5"]

# iframe heights: generous, with the static board vertically centred and the
# themed background filling the frame (fill=True) so the view never scrolls.
GRID_H, SPACE_H = 540, 540


def store():
    return st.session_state.setdefault("store", {})


def eps_note(e0, decay, emin, episodes):
    """Tell the user when ε actually bottoms out — a 0.9 decay hits the floor in
    ~30 episodes, which is invisible on a 3000-episode run."""
    import math
    if decay >= 1.0:
        return f"ε stays at **{e0:g}** for all {episodes} episodes (no decay)."
    if e0 <= emin:
        return f"ε is already at its minimum ({emin:g})."
    n = math.ceil(math.log(emin / e0) / math.log(decay))
    pct = 100 * min(n, episodes) / max(episodes, 1)
    return (f"ε falls from {e0:g} → {emin:g} after **~{n} episodes** "
            f"({pct:.0f}% of the {episodes}-episode run).")


# --------------------------------------------------------------------------- #
# Sidebar — room selector + ALL hyperparameters + train/reset
# --------------------------------------------------------------------------- #
def sidebar():
    st.sidebar.title("🐕 Escape Room")
    st.sidebar.caption("Hezki the Dog vs. Men in Black")

    key = st.sidebar.radio(
        "**Difficulty progression**", ORDER,
        format_func=lambda k: f"{ROOMS[k]['emoji']}  Room {k[-1]} · {ROOMS[k]['label']}  "
                              f"{'★' * ROOMS[k]['stars']}{'☆' * (5 - ROOMS[k]['stars'])}")
    r = ROOMS[key]
    st.sidebar.markdown(f"**🎬 {r['movie']}**")
    st.sidebar.markdown(f"**🧠 {r['algo']}**")
    st.sidebar.divider()

    st.sidebar.subheader("⚙️ Hyperparameters")
    p = {}
    if key == "room1":
        p["gamma"] = st.sidebar.slider("γ  discount", 0.50, 0.999, 0.99, 0.001, key="g1")
        p["theta"] = st.sidebar.number_input("θ  convergence threshold", 1e-6, 1e-1,
                                             1e-4, format="%.6f", key="t1")
    elif key in ("room2", "room3"):
        r2 = (key == "room2")
        # Room 2 is a zig-zag corridor maze: ε-greedy random walks never reach the
        # idol, so it defaults to low ε + optimistic init (directed exploration).
        p["alpha"] = st.sidebar.slider("α  learning rate", 0.01, 1.0, 0.10, 0.01, key=f"a{key}")
        p["gamma"] = st.sidebar.slider("γ  discount", 0.50, 0.999, 0.99, 0.001, key=f"g{key}")
        p["epsilon"] = st.sidebar.slider("ε  initial exploration", 0.10, 1.0,
                                         0.10 if r2 else 1.0, 0.01, key=f"e{key}")
        p["epsilon_decay"] = st.sidebar.slider("ε decay / episode", 0.9900, 1.0,
                                               1.0 if r2 else 0.9950, 0.0005,
                                               format="%.4f", key=f"ed{key}")
        p["epsilon_min"] = st.sidebar.slider("ε minimum", 0.0, 0.50, 0.01, 0.01, key=f"em{key}")
        p["optimistic_init"] = st.sidebar.slider("optimistic init Q₀", 0.0, 3000.0,
                                                 500.0 if r2 else 0.0, 50.0, key=f"oi{key}")
        p["episodes"] = st.sidebar.number_input("episodes", 100, 40000,
                                                3000 if r2 else 1000, 100, key=f"ep{key}")
        p["max_steps"] = st.sidebar.number_input("max steps / episode", 0, 2000, 400, 1, key=f"ms{key}")
        st.sidebar.caption(eps_note(p["epsilon"], p["epsilon_decay"],
                                    p["epsilon_min"], p["episodes"]))
        if r2:
            st.sidebar.caption("Step reward is 0 here; the idol (+1000) opens the gate to the "
                               "exit (+2000). Pressing the plate wakes the boulder instantly; "
                               "it then retraces your route one step per move. ⏱️ ~7 s to train.")
    elif key == "room4":
        p["alpha"] = st.sidebar.slider("α  learning rate", 0.05, 1.0, 0.50, 0.05, key="a4")
        p["gamma"] = st.sidebar.slider("γ  discount", 0.50, 0.999, 0.99, 0.001, key="g4")
        p["epsilon"] = st.sidebar.slider("ε  exploration", 0.10, 1.0, 0.10, 0.01, key="e4")
        p["epsilon_decay"] = st.sidebar.slider("ε decay / episode", 0.9900, 1.0, 1.0, 0.0005,
                                              format="%.4f", key="ed4")
        p["episodes"] = st.sidebar.number_input("episodes", 200, 20000, 2000, 100, key="ep4")
        with st.sidebar.expander("Function-approximation & physics"):
            p["optimistic_init"] = st.slider("optimistic init Q₀", 0.0, 200.0, 100.0, 10.0, key="oi4")
            p["n_tilings"] = st.slider("tilings", 4, 16, 8, 1, key="nt4")
            p["n_bins"] = st.slider("bins / dim", 4, 12, 8, 1, key="nb4")
            p["v_max"] = st.slider("max speed (m/s)", 1.0, 6.0, 3.0, 0.5, key="vm4")
            p["max_steps"] = st.number_input("max steps / episode", 0, 2000, 800, 1, key="ms4")
            p["shaping"] = st.checkbox("reward shaping (wave-front potential)", True, key="sh4")
            p["shaping_coef"] = st.slider("shaping coefficient", 0.0, 10.0, 5.0, 0.5, key="sc4")
            p["hard_walls"] = st.checkbox("hard walls (−100 terminal on boundary)", False, key="hw4")
        p["epsilon_min"] = 0.0
        st.sidebar.caption("⏱️ ~1 min to train at default settings.")
    else:  # room5
        p["vision"] = st.sidebar.slider("👁️ vision range X_obs (m)", 0.5, 8.0, 3.0, 0.5, key="v5")
        p["spawn_every"] = st.sidebar.slider("drone spawn every N steps", 5, 60, 15, 1, key="sp5")
        p["alpha"] = st.sidebar.slider("α  learning rate", 0.01, 1.0, 0.10, 0.01, key="a5")
        p["gamma"] = st.sidebar.slider("γ  discount", 0.50, 0.999, 0.95, 0.001, key="g5")
        p["epsilon"] = st.sidebar.slider("ε  initial exploration", 0.10, 1.0, 1.0, 0.01, key="e5")
        p["epsilon_decay"] = st.sidebar.slider("ε decay / episode", 0.9900, 1.0, 0.9990, 0.0005,
                                              format="%.4f", key="ed5")
        p["epsilon_min"] = st.sidebar.slider("ε minimum", 0.0, 0.5, 0.01, 0.01, key="em5")
        p["episodes"] = st.sidebar.number_input("episodes", 200, 30000, 3000, 100, key="ep5")
        with st.sidebar.expander("Physics"):
            p["v_max"] = st.slider("max speed (m/s)", 1.0, 6.0, 5.0, 0.5, key="vm5")
            p["max_steps"] = st.number_input("max steps (survival cap)", 0, 3000, 500, 1, key="ms5")

    st.sidebar.divider()
    train_clicked = st.sidebar.button("🚀 Train (resets this room)",
                                      use_container_width=True, type="primary")

    random_clicked = False
    if key == "room5":
        random_clicked = st.sidebar.button("🎲 Generate Random Room & Test Policy",
                                           use_container_width=True,
                                           disabled=key not in store())
    return key, p, train_clicked, random_clicked


# --------------------------------------------------------------------------- #
# Env / agent construction
# --------------------------------------------------------------------------- #
def build_env(key, p, seed=None):
    if key == "room1":
        return E.Room1FrozenArchive(seed=seed)
    if key == "room2":
        return E.Room2DarkTemple(seed=seed)
    if key == "room3":
        return E.Room3CloningLab(seed=seed)
    if key == "room4":
        return E.Room4Garage(v_max=p["v_max"], max_steps=p["max_steps"],
                             shaping=p["shaping"], shaping_coef=p["shaping_coef"],
                             hard_walls=p["hard_walls"])
    return E.Room5SpaceEscape(vision=p["vision"], spawn_every=p["spawn_every"],
                              v_max=p["v_max"], max_steps=p["max_steps"], seed=seed)


def train(key, p):
    # Training always starts from a clean slate for this room.
    store().pop(key, None)
    st.session_state.get("evalcache", {}).clear()
    st.session_state.get("epcache", {}).clear()
    st.session_state.pop(f"stage_{key}", None)
    st.session_state.pop(f"epsel_{key}", None)
    st.session_state.pop("rand_seed", None)

    env = build_env(key, p, seed=0)
    bar = st.progress(0.0, text="Training…")

    def cb(i, n):
        if i % max(1, n // 100) == 0 or i == n:
            bar.progress(i / n, text=f"Training… {i}/{n} episodes")

    entry: dict = dict(params=dict(p), meta=env.render_meta(), kind=ROOMS[key]["kind"])
    if key == "room1":
        vi = A.ValueIteration(env, gamma=p["gamma"], theta=p["theta"])
        res = vi.run()
        entry.update(res=res, policy=res["policy"], final_policy=res["final_policy"])
    else:
        common = dict(alpha=p["alpha"], gamma=p["gamma"], epsilon=p["epsilon"],
                      epsilon_decay=p["epsilon_decay"], epsilon_min=p["epsilon_min"],
                      episodes=p["episodes"], seed=0)
        if key == "room4":
            agent = A.LinearFAAgent(env, optimistic_init=p["optimistic_init"],
                                    n_tilings=p["n_tilings"], n_bins=p["n_bins"],
                                    max_steps=p["max_steps"], **common)
        elif key == "room2":
            agent = A.Sarsa(env, max_steps=p["max_steps"],
                            optimistic_init=p.get("optimistic_init", 0.0), **common)
        else:  # room3 & room5 -> Q-Learning
            agent = A.QLearning(env, max_steps=p["max_steps"],
                                optimistic_init=p.get("optimistic_init", 0.0), **common)
        res = agent.train(snapshots=6, progress=cb)
        entry.update(res=res, final_policy=res["final_policy"],
                     policy=getattr(agent, "Q", None) and
                            {s: int(a.argmax()) for s, a in agent.Q.items()})
    bar.empty()
    store()[key] = entry
    st.session_state.get("evalcache", {}).clear()      # fresh policy → fresh replays


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def board_html(key, meta, agent=None, policy=None, obstacles=None, vision=None,
               trail=None, fill=False, mask=0, chaser=None):
    theme = U.ROOM_THEME[key]
    if ROOMS[key]["kind"] in ("dp", "grid"):
        return U.render_grid_html(meta, theme, agent=agent, policy=policy,
                                  fill=fill, mask=mask, chaser=chaser)
    return U.render_space_svg(meta, theme, agent=agent, obstacles=obstacles,
                              vision=vision, trail=trail, fill=fill)


def legend_html(key, meta):
    return U.render_legend(U.ROOM_THEME[key], meta)


def replay_budget(entry):
    """Steps allowed in an evaluation/replay episode — the SAME cap the user
    trained with, so `max steps = 1` really means a one-step replay. (Room 1 /
    DP has no such knob, so it gets a sensible fixed budget.)"""
    return int(entry["params"].get("max_steps", 200))


def eval_roll(key, entry, policy, seed=1):
    """Run one greedy episode with `policy` (no rendering)."""
    env = build_env(key, entry["params"], seed=seed)
    return A.rollout(env, policy, max_steps=replay_budget(entry))


def evaluate(key, entry, policy, n=25, seed=7):
    """Greedy success RATE over n episodes (robust to slippery stochasticity),
    plus a representative episode (a successful one if the policy usually wins)."""
    env = build_env(key, entry["params"], seed=seed)
    ms = replay_budget(entry)
    wins, steps, rep = 0, [], None
    for _ in range(n):
        roll = A.rollout(env, policy, max_steps=ms)
        if roll["success"]:
            wins += 1
            steps.append(roll["steps"])
        if rep is None or (roll["success"] and not rep["success"]):
            rep = roll
    import numpy as _np
    return dict(rate=wins / n, rep=rep,
                avg_steps=(float(_np.mean(steps)) if steps else 0.0))


def render_frames(key, entry, roll, cap=320):
    """Render a rollout's frames to themed HTML for the replay player.
    Very long episodes are down-sampled so the animation payload stays light."""
    meta = entry["meta"]
    vision = entry["params"].get("vision") if key == "room5" else None
    frames = roll["frames"]
    if len(frames) > cap:                                    # keep the last frame
        step = (len(frames) - 1) / (cap - 1)
        frames = [frames[int(round(i * step))] for i in range(cap)]
    return [board_html(key, meta, agent=f["agent"], obstacles=f.get("obstacles"),
                       vision=vision, mask=f.get("mask", 0),
                       chaser=f.get("chaser")) for f in frames]


def replay_eval(key, entry, tag, policy):
    """Cached success-rate evaluation + rendered representative episode, so the
    replay is computed once per training-stage rather than on every rerun."""
    cache = st.session_state.setdefault("evalcache", {})
    ck = (key, tag, id(policy))
    if ck not in cache:
        n = 25 if ROOMS[key]["kind"] in ("dp", "grid") else 12
        ev = evaluate(key, entry, policy, n=n)
        cache[ck] = dict(rate=ev["rate"], avg=ev["avg_steps"], n=n,
                         steps=ev["rep"]["steps"], success=ev["rep"]["success"],
                         frames=render_frames(key, entry, ev["rep"]))
    return cache[ck]


LEGEND_H = 92


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_simulation(key, entry):
    r = ROOMS[key]
    st.markdown(f"### {r['emoji']} Room {key[-1]} · {r['label']}  ·  🎬 *{r['movie']}*")
    st.caption(r["plot"])
    meta = entry["meta"] if entry else build_env(key, default_params(key)).render_meta()
    board_h = GRID_H if r["kind"] in ("dp", "grid") else SPACE_H

    if not entry:
        st.info("Configure the hyperparameters on the left and press **🚀 Train** "
                "to teach Hezki this room. The level layout is shown below.")
        embed(board_html(key, meta, fill=True), height=board_h)
        embed(legend_html(key, meta), height=LEGEND_H)
        return

    if r["kind"] in ("dp", "grid"):
        st.caption("Arrows are the learned greedy policy · every special tile shows its reward.")
        embed(board_html(key, meta, agent=meta["start"], policy=entry.get("policy"), fill=True),
              height=board_h)
    else:
        roll = eval_roll(key, entry, entry["final_policy"], seed=1)
        last = roll["frames"][-1]
        embed(board_html(key, meta, agent=last["agent"], obstacles=last.get("obstacles"),
                         vision=entry["params"].get("vision") if key == "room5" else None,
                         trail=[f["agent"] for f in roll["frames"]] if key == "room4" else None,
                         fill=True), height=board_h)
        ok = "✅ escaped" if roll["success"] else "❌ did not finish"
        st.metric("Greedy evaluation", f"{ok} · {roll['steps']} steps · return {roll['reward']:.1f}")
    embed(legend_html(key, meta), height=LEGEND_H)


def tab_charts(key, entry):
    if not entry:
        st.info("Train the room to see learning-progress charts.")
        return
    res = entry["res"]
    if key == "room1":
        c1, c2 = st.columns(2)
        c1.plotly_chart(U.convergence_chart(res["deltas"]), use_container_width=True)
        c2.plotly_chart(U.value_heatmap(res["V"], entry["meta"]), use_container_width=True)
        sx, sy = entry["meta"]["start"]
        v0 = res["V"].get((sx, sy, 0), res["V"].get((sx, sy)))
        st.success(f"Converged in **{res['iterations']}** sweeps "
                   f"(θ = {entry['params']['theta']:g}). V(start) = {v0:.2f}  "
                   f"— the value of the whole plan: key → gate → exit (plus bonuses).")
    elif key in ("room2", "room3"):
        c1, c2 = st.columns(2)
        c1.plotly_chart(U.reward_curve(res["rewards"]), use_container_width=True)
        c2.plotly_chart(U.epsilon_curve(res["epsilons"]), use_container_width=True)
        if key == "room3":
            roll = eval_roll(key, entry, entry["final_policy"], seed=1)
            path = [f["agent"] for f in roll["frames"]]
            st.plotly_chart(U.path_compare(entry["meta"], path), use_container_width=True)
    elif key == "room4":
        c1, c2 = st.columns(2)
        c1.plotly_chart(U.reward_curve(res["rewards"], window=50), use_container_width=True)
        c2.plotly_chart(U.length_curve(res["lengths"], window=50,
                        title="Episode duration (moving avg)"), use_container_width=True)
    else:  # room5
        st.plotly_chart(U.length_curve(res["lengths"], window=50,
                        title="Survival time per episode", ylab="Steps survived"),
                        use_container_width=True)
        st.plotly_chart(U.reward_curve(res["rewards"], window=50), use_container_width=True)


def episode_list(key, entry):
    """Recorded episodes for this room, best total reward first."""
    cache = st.session_state.setdefault("epcache", {})
    ck = (key, id(entry))
    if ck in cache:
        return cache[ck]
    if ROOMS[key]["kind"] == "dp":
        # Value Iteration has no training episodes — run the optimal policy a
        # number of times instead (the ice makes every run different).
        eps = []
        for i in range(20):
            roll = eval_roll(key, entry, entry["final_policy"], seed=100 + i)
            eps.append(dict(episode=i, reward=roll["reward"], steps=roll["steps"],
                            success=roll["success"], frames=roll["frames"]))
    else:
        eps = list(entry["res"].get("tapes", []))
    eps.sort(key=lambda e: -e["reward"])
    cache[ck] = eps
    return eps


def step_notes(entry, ep):
    """One line per frame: the action taken, what happened, and the reward."""
    meta = entry["meta"]
    inv_bit = {i: c for c, i in meta.get("bit", {}).items()}
    pickups, prew = meta.get("pickups", {}), meta.get("pickup_rewards", {})
    start = meta.get("start")
    frames = ep["frames"]
    acts = ep.get("actions") or []
    rews = ep.get("step_rewards") or []
    notes, cum = ["▶ start of episode"], 0.0
    for i in range(1, len(frames)):
        prev, cur = frames[i - 1], frames[i]
        a = acts[i - 1] if i - 1 < len(acts) else None
        r = float(rews[i - 1]) if i - 1 < len(rews) else 0.0
        cum += r
        arrow = E.ACTION_ARROWS.get(a, "") if a is not None else ""
        name = E.ACTION_NAMES.get(a, "?") if a is not None else "?"
        ev = []
        gained = int(cur.get("mask", 0)) & ~int(prev.get("mask", 0))
        for b in range(gained.bit_length()):
            if (gained >> b) & 1:
                ch = pickups.get(inv_bit.get(b))
                ev.append(f"took the {'key' if ch == 'K' else 'bonus'} "
                          f"(+{prew.get(ch, 0):g})")
        if prev.get("chaser") is not None and cur.get("chaser") is None and r < 0:
            ev.append("caught by the chaser → room reset")
        elif (start and tuple(cur["agent"]) == tuple(start)
              and tuple(prev["agent"]) != tuple(start) and r < 0):
            ev.append("fell into a pit → back to start")
        if tuple(cur["agent"]) == tuple(prev["agent"]) and not ev:
            ev.append("blocked — walked into a wall/edge")
        if i == len(frames) - 1 and ep.get("success"):
            ev.append("reached the EXIT 🎉")
        notes.append(f"step {i}  {arrow} {name}  ·  "
                     f"{', '.join(ev) if ev else 'moved'}  ·  "
                     f"reward {r:+.0f}  ·  total {cum:+.0f}")
    return notes


def grid_player(key, entry, ep, token):
    """Exact, step-by-step replay for the grid rooms (no down-sampling)."""
    theme, meta = U.ROOM_THEME[key], entry["meta"]
    T = U.THEMES[theme]
    masks = sorted({f.get("mask", 0) for f in ep["frames"]})
    boards = {str(m): U.render_grid_html(meta, theme, agent=None, mask=m, ids=True)
              for m in masks}
    frames = [{"m": str(f.get("mask", 0)), "a": list(f["agent"]),
               "c": (list(f["chaser"]) if f.get("chaser") else None)}
              for f in ep["frames"]]
    return U.render_player_grid(boards, frames, T["agent"], T.get("chaser", "🪨"),
                                T["accent"], T["board"], delay_ms=160, token=token,
                                notes=step_notes(entry, ep))


def tab_replay(key, entry, random_clicked):
    if not entry:
        st.info("Train the room, then browse its episodes here.")
        return
    r = ROOMS[key]
    is_grid = r["kind"] in ("dp", "grid")
    embed(legend_html(key, entry["meta"]), height=LEGEND_H)
    board_h = (GRID_H if is_grid else SPACE_H) + 130

    if key == "room5" and random_clicked:
        seed = int(st.session_state.get("rand_seed", 0)) + 1
        st.session_state["rand_seed"] = seed
        roll = eval_roll(key, entry, entry["final_policy"], seed=1000 + seed)
        st.caption(f"🎲 Random test room (seed {seed}) · final policy · "
                   f"{'survived ✅' if roll['success'] else 'hit a drone ❌'} — {roll['steps']} steps")
        embed(U.render_player(render_frames(key, entry, roll), delay_ms=70), height=board_h)
        return

    eps = episode_list(key, entry)
    if not eps:
        st.info("No episodes were recorded for this run.")
        return

    st.caption("Pick an episode to replay — **sorted best total reward first**. "
               "Every step is shown exactly; if Hezki seems to stand still he tried to "
               "walk into a wall (the only actions are up / down / left / right).")

    def label(i):
        e = eps[i]
        tick = "✅" if e["success"] else "❌"
        return (f"{tick}  return {e['reward']:>8.0f}   ·   {e['steps']:>4d} steps"
                f"   ·   episode #{e['episode']}")

    idx = st.selectbox("Episode", range(len(eps)), format_func=label,
                       key=f"epsel_{key}")
    ep = eps[idx]
    st.caption(f"Episode **#{ep['episode']}** · total reward **{ep['reward']:.0f}** · "
               f"{ep['steps']} steps · {'escaped ✅' if ep['success'] else 'did not escape ❌'}")
    # token changes with the selection, so the player remounts and restarts
    token = f"{key}-{idx}-{ep['episode']}"
    if is_grid:
        embed(grid_player(key, entry, ep, token), height=board_h)
    else:
        embed(U.render_player(render_frames(key, entry, ep), delay_ms=70,
                              caption=f"episode #{ep['episode']}"), height=board_h)


PRETTY = {"alpha": "α", "gamma": "γ", "epsilon": "ε₀", "epsilon_decay": "ε decay",
          "epsilon_min": "ε min", "optimistic_init": "Q₀ optimistic init",
          "theta": "θ", "episodes": "episodes", "max_steps": "max steps/episode",
          "n_tilings": "tilings", "n_bins": "bins/dim", "v_max": "v_max",
          "shaping": "reward shaping", "shaping_coef": "shaping coef",
          "hard_walls": "hard walls", "vision": "vision range", "spawn_every": "spawn every"}


def params_panel(key, entry):
    """Show the hyperparameters the displayed model was actually trained with."""
    def fmt(v):
        if isinstance(v, bool):
            return "on" if v else "off"
        return f"{v:g}" if isinstance(v, float) else str(v)

    chips = "  ".join(f"`{PRETTY.get(k, k)} = {fmt(v)}`" for k, v in entry["params"].items())
    with st.container(border=True):
        st.markdown(f"**🧠 Trained model — {ROOMS[key]['algo']}**")
        st.markdown(chips)


def default_params(key):
    """Minimal params so an untrained room can still render its static layout."""
    return dict(v_max=3.0, max_steps=800, shaping=True, shaping_coef=5.0,
                hard_walls=False, vision=3.0, spawn_every=15)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    key, p, train_clicked, random_clicked = sidebar()

    st.title("🐕 Hezki the Dog vs. the Men in Black")
    st.caption("A Reinforcement-Learning escape room — five movie sets, five algorithms, "
               "rising difficulty. Agent J wants to neuralyze Hezki; help him escape.")

    if train_clicked:
        with st.spinner("Hezki is learning…"):
            train(key, p)

    entry = store().get(key)
    if entry:
        params_panel(key, entry)
    if entry and entry.get("params") != p:
        st.warning("⚙️ You've changed hyperparameters since the last training. Everything below "
                   "still reflects the **previous** run — click **🚀 Train** to apply the new "
                   "settings.", icon="⚠️")

    t1, t2, t3 = st.tabs(["🎬 Room Simulation (HTML)", "📈 Training Metrics", "⏪ Episode Replay"])
    with t1:
        tab_simulation(key, entry)
    with t2:
        tab_charts(key, entry)
    with t3:
        tab_replay(key, entry, random_clicked)


if __name__ == "__main__":
    main()
