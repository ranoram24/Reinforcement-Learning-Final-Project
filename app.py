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
    "room2": dict(label="Cloning Lab", emoji="🕶️", stars=2, kind="grid",
                  movie="The Matrix (1999)", algo="SARSA (on-policy TD)",
                  plot="A cloning-lab puzzle: Hezki must push the two 📦 boxes onto the two "
                       "pressure plates 🔘 to open the ice-gate 🔒, then reach the exit. "
                       "The reset tile 🔄 undoes a dead-lock; 💊 bonuses pay once."),
    "room3": dict(label="Dark Temple", emoji="🏛️", stars=3, kind="grid",
                  movie="Raiders of the Lost Ark (1981)", algo="Q-Learning (off-policy TD)",
                  plot="Hezki falls into an ancient temple. Off-policy Q-Learning is "
                       "aggressive — it grabs the idol 🏆, presses the plate for the bonus, "
                       "takes the boulder 🪨 hit, and still races to the exit."),
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


def eps_note(e0, k, emin, episodes):
    """Linear decay ε = ε₀ − K·t — tell the user when ε reaches its floor."""
    import math
    if k <= 0:
        return f"ε stays at **{e0:g}** for all {episodes} episodes (K = 0, no decay)."
    if e0 <= emin:
        return f"ε is already at its minimum ({emin:g})."
    n = math.ceil((e0 - emin) / k)
    pct = 100 * min(n, episodes) / max(episodes, 1)
    return (f"ε = ε₀ − K·t : falls **{e0:g} → {emin:g}** at −{k:g}/episode, reaching the "
            f"floor after **~{n} episodes** ({pct:.0f}% of the {episodes}-episode run).")


def _randomize_hp(key):
    """Write a random-but-sane value into every hyperparameter widget of `key`.
    Called before the widgets are built, so session_state feeds their defaults.
    Bounds stay inside each widget's (min,max) and snap to its step."""
    import random as _r
    ss = st.session_state

    def flt(lo, hi, step):           # random float in [lo,hi], snapped to `step`
        v = round(round(_r.uniform(lo, hi) / step) * step, 6)
        return float(min(hi, max(lo, v)))

    def integer(lo, hi, step):       # random int in [lo,hi], multiple of `step`
        v = int(round(_r.uniform(lo, hi) / step) * step)
        return int(min(hi, max(lo, v)))

    if key == "room1":
        ss["g1"] = flt(0.80, 0.999, 0.001)
        ss["t1"] = _r.choice([1e-3, 1e-4, 1e-5, 1e-6])
    elif key in ("room2", "room3"):
        ss[f"a{key}"] = flt(0.05, 0.80, 0.01)
        ss[f"g{key}"] = flt(0.85, 0.999, 0.001)
        ss[f"e{key}"] = flt(0.50, 1.00, 0.01)
        ss[f"ed{key}"] = flt(0.0001, 0.0010, 0.0001)
        ss[f"em{key}"] = flt(0.00, 0.10, 0.01)
        ss[f"oi{key}"] = flt(0.0, 2000.0, 50.0)
        ss[f"ep{key}"] = integer(2000, 8000, 100)
        ss[f"ms{key}"] = integer(300, 800, 1)
    elif key == "room4":
        ss["a4"] = flt(0.15, 0.80, 0.05)
        ss["g4"] = flt(0.90, 0.999, 0.001)
        ss["e4"] = flt(0.10, 0.40, 0.01)
        ss["ed4"] = flt(0.0000, 0.0020, 0.0001)
        ss["ep4"] = integer(1500, 4000, 100)
        ss["oi4"] = flt(50.0, 200.0, 10.0)
        ss["nt4"] = integer(6, 12, 1)
        ss["nb4"] = integer(6, 10, 1)
        ss["vm4"] = flt(2.0, 4.0, 0.5)
        ss["ms4"] = integer(500, 1000, 1)
        ss["sh4"] = True                       # shaping on — the room rarely solves without it
        ss["sc4"] = flt(3.0, 8.0, 0.5)
        ss["hw4"] = False
    else:  # room5
        ss["v5"] = flt(2.0, 5.0, 0.5)
        ss["sp5"] = integer(8, 30, 1)
        ss["a5"] = flt(0.05, 0.40, 0.01)
        ss["g5"] = flt(0.90, 0.999, 0.001)
        ss["e5"] = flt(0.60, 1.00, 0.01)
        ss["ed5"] = flt(0.0002, 0.0010, 0.0001)
        ss["em5"] = flt(0.00, 0.10, 0.01)
        ss["ep5"] = integer(2000, 6000, 100)
        ss["vm5"] = flt(4.0, 6.0, 0.5)
        ss["ms5"] = integer(400, 800, 1)


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
    # A pending "🎲 Randomize" click writes fresh values into the widget keys BEFORE
    # the widgets below are instantiated (Streamlit forbids editing a widget's state
    # after it exists in the same run), so the sliders pick the new values up.
    if st.session_state.pop("_randhp", None) == key:
        _randomize_hp(key)
    p = {}
    if key == "room1":
        p["gamma"] = st.sidebar.slider("γ  discount", 0.50, 0.999, 0.99, 0.001, key="g1")
        p["theta"] = st.sidebar.number_input("θ  convergence threshold", 1e-6, 1e-1,
                                             1e-4, format="%.6f", key="t1")
    elif key in ("room2", "room3"):
        # Room 2 = Cloning-Lab Sokoban + SARSA ; Room 3 = Dark Temple boulder + Q-Learning.
        boulder = (key == "room3")
        p["alpha"] = st.sidebar.slider("α  learning rate", 0.01, 1.0, 0.10, 0.01, key=f"a{key}")
        p["gamma"] = st.sidebar.slider("γ  discount", 0.50, 0.999, 0.99, 0.001, key=f"g{key}")
        p["epsilon"] = st.sidebar.slider("ε₀  initial exploration", 0.10, 1.0, 1.0, 0.01, key=f"e{key}")
        p["epsilon_k"] = st.sidebar.slider("K  ε decrement / episode (linear ε=ε₀−K·t)",
                                           0.0, 0.05, 0.0002 if boulder else 0.0003, 0.0001,
                                           format="%.4f", key=f"ed{key}")
        p["epsilon_min"] = st.sidebar.slider("ε minimum", 0.0, 0.50, 0.01, 0.01, key=f"em{key}")
        p["optimistic_init"] = st.sidebar.slider("optimistic init Q₀", 0.0, 3000.0,
                                                 0.0, 50.0, key=f"oi{key}")
        p["episodes"] = st.sidebar.number_input("episodes", 100, 40000,
                                                4000 if boulder else 6000, 100, key=f"ep{key}")
        p["max_steps"] = st.sidebar.number_input("max steps / episode", 0, 2000,
                                                 400 if boulder else 500, 1, key=f"ms{key}")
        st.sidebar.caption(eps_note(p["epsilon"], p["epsilon_k"],
                                    p["epsilon_min"], p["episodes"]))
        if boulder:
            st.sidebar.caption("Off-policy Q-Learning + a decaying ε solves this. Step reward 0; "
                               "idol +100000 opens the gate, plate +10000 (once, wakes the "
                               "boulder), catch −100000 (kept everything), exit +200000.")
        else:
            st.sidebar.caption("On-policy SARSA. Push both 📦 onto the plates 🔘 (+5000 each) "
                               "to open the gate, then reach the exit (+10000). Step −1; "
                               "💊 bonuses +1000 (one-off); ⚠️ hazards −100 each step on them.")
    elif key == "room4":
        p["alpha"] = st.sidebar.slider("α  learning rate", 0.05, 1.0, 0.50, 0.05, key="a4")
        p["gamma"] = st.sidebar.slider("γ  discount", 0.50, 0.999, 0.99, 0.001, key="g4")
        p["epsilon"] = st.sidebar.slider("ε₀  exploration", 0.10, 1.0, 0.10, 0.01, key="e4")
        p["epsilon_k"] = st.sidebar.slider("K  ε decrement / episode (linear ε=ε₀−K·t)",
                                           0.0, 0.05, 0.0, 0.0001, format="%.4f", key="ed4")
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
        st.sidebar.caption(eps_note(p["epsilon"], p["epsilon_k"], p["epsilon_min"], p["episodes"]))
        st.sidebar.caption("⏱️ ~1 min to train at default settings.")
    else:  # room5
        p["vision"] = st.sidebar.slider("👁️ vision range X_obs (m)", 0.5, 8.0, 3.0, 0.5, key="v5")
        p["spawn_every"] = st.sidebar.slider("drone spawn every N steps", 5, 60, 15, 1, key="sp5")
        p["alpha"] = st.sidebar.slider("α  learning rate", 0.01, 1.0, 0.10, 0.01, key="a5")
        p["gamma"] = st.sidebar.slider("γ  discount", 0.50, 0.999, 0.95, 0.001, key="g5")
        p["epsilon"] = st.sidebar.slider("ε₀  initial exploration", 0.10, 1.0, 1.0, 0.01, key="e5")
        p["epsilon_k"] = st.sidebar.slider("K  ε decrement / episode (linear ε=ε₀−K·t)",
                                           0.0, 0.05, 0.0004, 0.0001, format="%.4f", key="ed5")
        p["epsilon_min"] = st.sidebar.slider("ε minimum", 0.0, 0.5, 0.01, 0.01, key="em5")
        p["episodes"] = st.sidebar.number_input("episodes", 200, 30000, 3000, 100, key="ep5")
        st.sidebar.caption(eps_note(p["epsilon"], p["epsilon_k"], p["epsilon_min"], p["episodes"]))
        with st.sidebar.expander("Physics"):
            p["v_max"] = st.slider("max speed (m/s)", 1.0, 6.0, 5.0, 0.5, key="vm5")
            p["max_steps"] = st.number_input("max steps (survival cap)", 0, 3000, 500, 1, key="ms5")

    if st.sidebar.button("🎲 Randomize hyperparameters", use_container_width=True):
        st.session_state["_randhp"] = key
        st.rerun()

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
    if key == "room2":                       # Room 2 = Cloning Lab (Sokoban, SARSA)
        return E.Room2CloningLab(seed=seed)
    if key == "room3":                       # Room 3 = Dark Temple boulder (Q-Learning)
        return E.Room2DarkTemple(seed=seed)
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
                      epsilon_k=p["epsilon_k"], epsilon_min=p["epsilon_min"],
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
               trail=None, fill=False, mask=0, chaser=None, boxes=None, door_open=False):
    theme = U.ROOM_THEME[key]
    if meta.get("kind") == "sokoban":
        return U.render_sokoban_html(meta, theme, agent=agent, boxes=boxes,
                                     door_open=door_open, mask=mask, fill=fill)
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
                       vision=vision, mask=f.get("mask", 0), chaser=f.get("chaser"),
                       boxes=f.get("boxes"), door_open=f.get("door_open", False))
            for f in frames]


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
        succ = res.get("successes", [])
        n, total = len(succ), int(sum(succ))
        if n:
            m1, m2, m3 = st.columns(3)
            m1.metric("🏆 Episodes escaped", f"{total:,}")
            m2.metric("❌ Episodes not escaped", f"{n - total:,}")
            m3.metric("🎯 Escape rate", f"{100 * total / n:.1f}%")
        c1, c2 = st.columns(2)
        c1.plotly_chart(U.reward_curve(res["rewards"]), use_container_width=True)
        c2.plotly_chart(U.epsilon_curve(res["epsilons"]), use_container_width=True)
        if key == "room2" and entry["meta"].get("cliff"):   # cliff room → learned-path plot
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


def discounted_return(step_rewards, gamma):
    """G = Σ γᵗ·rₜ  — the *return* as defined in RL (the raw sum is not a return)."""
    return sum((gamma ** t) * r for t, r in enumerate(step_rewards or []))


def episode_list(key, entry):
    """Recorded episodes for this room, best (discounted) return first."""
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
                            success=roll["success"], frames=roll["frames"],
                            actions=roll.get("actions"),
                            step_rewards=roll.get("step_rewards")))
    else:
        eps = list(entry["res"].get("tapes", []))
    gamma = float(entry["params"].get("gamma", 1.0))
    for e in eps:                                   # attach the discounted return
        e["ret"] = discounted_return(e.get("step_rewards"), gamma)
    eps.sort(key=lambda e: -e["ret"])               # best return first
    cache[ck] = eps
    return eps


def step_notes(entry, ep):
    """One line per frame: the action taken, what happened, and the reward."""
    meta = entry["meta"]
    inv_bit = {i: c for c, i in meta.get("bit", {}).items()}
    pickups, prew = meta.get("pickups", {}), meta.get("pickup_rewards", {})
    start = meta.get("start")
    gamma = float(entry["params"].get("gamma", 1.0))
    frames = ep["frames"]
    acts = ep.get("actions") or []
    rews = ep.get("step_rewards") or []
    notes = ["▶ start of episode  ·  discounted reward 0  ·  total reward 0"]
    cum, disc = 0.0, 0.0
    for i in range(1, len(frames)):
        prev, cur = frames[i - 1], frames[i]
        a = acts[i - 1] if i - 1 < len(acts) else None
        r = float(rews[i - 1]) if i - 1 < len(rews) else 0.0
        cum += r
        disc += (gamma ** (i - 1)) * r              # discount by steps-so-far
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
            ev.append("caught by the boulder → big penalty (stays put, boulder gone)")
        elif (start and tuple(cur["agent"]) == tuple(start)
              and tuple(prev["agent"]) != tuple(start) and r < 0):
            ev.append("fell into a pit → back to start")
        if tuple(cur["agent"]) == tuple(prev["agent"]) and not ev:
            ev.append("slipped on ice → stayed in place")
        if i == len(frames) - 1 and ep.get("success"):
            ev.append("reached the EXIT 🎉")
        notes.append(f"step {i}  {arrow} {name}  ·  "
                     f"{', '.join(ev) if ev else 'moved'}  ·  reward {r:+.0f}  ·  "
                     f"discounted reward {disc:+.0f}  ·  total reward {cum:+.0f}")
    return notes


def sokoban_notes(entry, ep):
    """One line per frame for the Cloning-Lab replay: the move Hezki chose, what
    happened (pushed a box, hit a plate, grabbed a bonus, opened the gate…), and
    the reward — mirroring the step-by-step narration the grid rooms show."""
    meta = entry["meta"]
    walls = meta.get("walls", set())
    buttons = set(meta.get("buttons", ()))
    negatives = set(meta.get("negatives", ()))
    reset_tile = meta.get("reset_tile")
    size = meta.get("size", 10)
    br, bor = meta.get("button_reward", 0), meta.get("bonus_reward", 0)
    ngr = meta.get("neg_reward", 0)
    gamma = float(entry["params"].get("gamma", 1.0))
    frames = ep["frames"]
    acts = ep.get("actions") or []
    rews = ep.get("step_rewards") or []
    notes = ["▶ start of episode  ·  discounted reward 0  ·  total reward 0"]
    cum = disc = 0.0
    for i in range(1, len(frames)):
        prev, cur = frames[i - 1], frames[i]
        a = acts[i - 1] if i - 1 < len(acts) else None
        r = float(rews[i - 1]) if i - 1 < len(rews) else 0.0
        cum += r
        disc += (gamma ** (i - 1)) * r
        arrow = E.ACTION_ARROWS.get(a, "") if a is not None else ""
        name = E.ACTION_NAMES.get(a, "?") if a is not None else "?"
        pa, ca = tuple(prev["agent"]), tuple(cur["agent"])
        pbx = {tuple(b) for b in prev.get("boxes", [])}
        cbx = {tuple(b) for b in cur.get("boxes", [])}
        ev = []
        if pbx != cbx:                                   # a box slid one cell
            landed = cbx - pbx
            if landed & buttons:
                ev.append(f"pushed 📦 onto a plate (+{br:g})")
            else:
                ev.append("pushed 📦 one cell")
        if int(cur.get("mask", 0)) & ~int(prev.get("mask", 0)):
            ev.append(f"grabbed a 💊 bonus (+{bor:g})")
        if ca in negatives:
            ev.append(f"stepped on a hazard ({ngr:g})")
        if reset_tile is not None and a is not None:     # deliberately moved onto reset
            dx, dy = E._DELTA[a]
            if (pa[0] + dx, pa[1] + dy) == tuple(reset_tile):
                ev.append("hit the reset 🔄 — everything back to start")
        if cur.get("door_open") and not prev.get("door_open"):
            ev.append("both plates covered — the ice-gate opens 🔓")
        if ca == pa and pbx == cbx and a is not None:    # Hezki did not move
            dx, dy = E._DELTA[a]
            tgt = (pa[0] + dx, pa[1] + dy)
            if tgt in pbx:
                ev.append("shoved a stuck 📦 — it wouldn't budge")
            elif tgt in walls or not (0 <= tgt[0] < size and 0 <= tgt[1] < size):
                ev.append("blocked by a wall")
            else:
                ev.append("slipped on ice → stayed put")
        if i == len(frames) - 1 and ep.get("success"):
            ev.append("reached the EXIT 🎉")
        notes.append(f"step {i}  {arrow} {name}  ·  "
                     f"{', '.join(ev) if ev else 'moved'}  ·  reward {r:+.0f}  ·  "
                     f"discounted reward {disc:+.0f}  ·  total reward {cum:+.0f}")
    return notes


def sokoban_player(key, entry, ep, cap=320):
    """Full-frame replay (boxes move, so cells can't just be swapped) WITH the
    per-step action narration shown beneath the board."""
    meta = entry["meta"]
    frames = ep["frames"]
    notes = sokoban_notes(entry, ep)
    idxs = list(range(len(frames)))
    if len(frames) > cap:                                # keep payload light
        step = (len(frames) - 1) / (cap - 1)
        idxs = [int(round(i * step)) for i in range(cap)]
    html = [board_html(key, meta, agent=frames[j]["agent"], mask=frames[j].get("mask", 0),
                       boxes=frames[j].get("boxes"), door_open=frames[j].get("door_open", False))
            for j in idxs]
    return U.render_player(html, delay_ms=110, caption=f"episode #{ep['episode']}",
                           notes=[notes[j] for j in idxs])


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

    gamma = float(entry["params"].get("gamma", 1.0))
    st.caption("Pick an episode to replay — **sorted by discounted reward first**. "
               "**discounted reward** = Σγᵗ·rₜ (the objective the agent optimises, "
               f"γ = {gamma:g}), so a *faster* escape scores higher. "
               "**total reward (without discount)** = raw Σrₜ. "
               "Hezki can only choose *legal* moves (walls are removed from his actions), "
               "so if he stays put he **slipped** on ice.")

    def label(i):
        e = eps[i]
        tick = "✅" if e["success"] else "❌"
        return (f"{tick}  discounted reward {e['ret']:>8.0f}   ·   "
                f"total reward {e['reward']:>8.0f}   ·   {e['steps']:>4d} steps   ·   #{e['episode']}")

    idx = st.selectbox("Episode", range(len(eps)), format_func=label,
                       key=f"epsel_{key}")
    ep = eps[idx]
    st.caption(f"Episode **#{ep['episode']}** · "
               f"discounted reward Σγᵗrₜ = **{ep['ret']:.0f}** (γ = {gamma:g}) · "
               f"total reward (without discount) Σrₜ = **{ep['reward']:.0f}** · "
               f"{ep['steps']} steps · {'escaped ✅' if ep['success'] else 'did not escape ❌'}")
    # token changes with the selection, so the player remounts and restarts
    token = f"{key}-{idx}-{ep['episode']}"
    sokoban = entry["meta"].get("kind") == "sokoban"      # boxes move → full-frame player
    if is_grid and not sokoban:
        embed(grid_player(key, entry, ep, token), height=board_h)
    elif sokoban:
        embed(sokoban_player(key, entry, ep), height=board_h + 40)
    else:
        embed(U.render_player(render_frames(key, entry, ep), delay_ms=110,
                              caption=f"episode #{ep['episode']}"), height=board_h)


PRETTY = {"alpha": "α", "gamma": "γ", "epsilon": "ε₀", "epsilon_k": "ε decrement K",
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
