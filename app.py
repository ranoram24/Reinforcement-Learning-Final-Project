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
                  plot="Hezki falls into an ancient alien temple — unknown terrain, "
                       "slippery mud and deadly spike pits. He must learn by trial."),
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
        p["alpha"] = st.sidebar.slider("α  learning rate", 0.01, 1.0, 0.10, 0.01, key=f"a{key}")
        p["gamma"] = st.sidebar.slider("γ  discount", 0.50, 0.999, 0.99, 0.001, key=f"g{key}")
        p["epsilon"] = st.sidebar.slider("ε  initial exploration", 0.0, 1.0, 1.0, 0.01, key=f"e{key}")
        p["epsilon_decay"] = st.sidebar.slider("ε decay / episode", 0.90, 1.0, 0.995, 0.001,
                                               format="%.3f", key=f"ed{key}")
        p["epsilon_min"] = st.sidebar.slider("ε minimum", 0.0, 0.5, 0.01, 0.01, key=f"em{key}")
        p["episodes"] = st.sidebar.number_input("episodes", 100, 20000,
                                               1500 if key == "room2" else 1000, 100, key=f"ep{key}")
        p["max_steps"] = st.sidebar.number_input("max steps / episode", 0, 2000, 400, 1, key=f"ms{key}")
    elif key == "room4":
        p["alpha"] = st.sidebar.slider("α  learning rate", 0.05, 1.0, 0.50, 0.05, key="a4")
        p["gamma"] = st.sidebar.slider("γ  discount", 0.50, 0.999, 0.99, 0.001, key="g4")
        p["epsilon"] = st.sidebar.slider("ε  exploration", 0.0, 1.0, 0.10, 0.01, key="e4")
        p["epsilon_decay"] = st.sidebar.slider("ε decay / episode", 0.90, 1.0, 1.0, 0.001,
                                              format="%.3f", key="ed4")
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
        p["epsilon"] = st.sidebar.slider("ε  initial exploration", 0.0, 1.0, 1.0, 0.01, key="e5")
        p["epsilon_decay"] = st.sidebar.slider("ε decay / episode", 0.90, 1.0, 0.999, 0.001,
                                              format="%.3f", key="ed5")
        p["epsilon_min"] = st.sidebar.slider("ε minimum", 0.0, 0.5, 0.01, 0.01, key="em5")
        p["episodes"] = st.sidebar.number_input("episodes", 200, 30000, 3000, 100, key="ep5")
        with st.sidebar.expander("Physics"):
            p["v_max"] = st.slider("max speed (m/s)", 1.0, 6.0, 5.0, 0.5, key="vm5")
            p["max_steps"] = st.number_input("max steps (survival cap)", 0, 3000, 500, 1, key="ms5")

    st.sidebar.divider()
    c1, c2 = st.sidebar.columns(2)
    train_clicked = c1.button("🚀 Train", use_container_width=True, type="primary")
    if c2.button("🗑️ Reset", use_container_width=True):
        store().pop(key, None)
        st.session_state.get("evalcache", {}).clear()
        st.rerun()

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
            agent = A.Sarsa(env, max_steps=p["max_steps"], **common)
        else:  # room3 & room5 -> Q-Learning
            agent = A.QLearning(env, max_steps=p["max_steps"], **common)
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
               trail=None, fill=False):
    theme = U.ROOM_THEME[key]
    if ROOMS[key]["kind"] in ("dp", "grid"):
        return U.render_grid_html(meta, theme, agent=agent, policy=policy, fill=fill)
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
    return [board_html(key, meta, agent=f["agent"], obstacles=f.get("obstacles"), vision=vision)
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
        st.success(f"Converged in **{res['iterations']}** sweeps "
                   f"(θ = {entry['params']['theta']:g}). "
                   f"V(start) = {res['V'][entry['meta']['start']]:.2f}")
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


def tab_replay(key, entry, random_clicked):
    if not entry:
        st.info("Train the room, then replay what Hezki learned at each stage here.")
        return
    r = ROOMS[key]

    embed(legend_html(key, entry["meta"]), height=LEGEND_H)
    board_h = (GRID_H if r["kind"] in ("dp", "grid") else SPACE_H) + 118
    delay = 200 if r["kind"] in ("dp", "grid") else 70

    if r["kind"] == "dp":
        st.caption("Dynamic Programming computes the optimal policy directly — there are no "
                   "training episodes. Because the ice is slippery, each run can differ:")
        ev = replay_eval(key, entry, "dp", entry["final_policy"])
        st.caption(f"Optimal policy · **{ev['rate']*100:.0f}% escape rate** over {ev['n']} runs "
                   f"· showing a representative run ({ev['steps']} steps).")
        embed(U.render_player(ev["frames"], delay_ms=delay), height=board_h)
        return

    if key == "room5" and random_clicked:
        seed = int(st.session_state.get("rand_seed", 0)) + 1
        st.session_state["rand_seed"] = seed
        roll = eval_roll(key, entry, entry["final_policy"], seed=1000 + seed)
        st.caption(f"🎲 Random test room (seed {seed}) · final policy · "
                   f"{'survived ✅' if roll['success'] else 'hit a drone ❌'} — {roll['steps']} steps")
        embed(U.render_player(render_frames(key, entry, roll), delay_ms=delay), height=board_h)
        return

    snaps = entry["res"]["snapshots"]
    labels = [f"episode {ep}" if ep < snaps[-1][0] else f"episode {ep} (final)"
              for ep, _ in snaps]
    st.caption("Replay the policy **as it was at a given training stage** — watch Hezki improve. "
               "The escape rate is measured over many runs so a single unlucky slip doesn't mislead.")
    choice = st.select_slider("Training stage", options=list(range(len(snaps))),
                              value=len(snaps) - 1, format_func=lambda i: labels[i],
                              key=f"stage_{key}")
    _, policy = snaps[choice]
    ev = replay_eval(key, entry, f"stage{choice}", policy)
    st.caption(f"Stage **{labels[choice]}** · **{ev['rate']*100:.0f}% escape rate** over {ev['n']} runs"
               + (f" · avg {ev['avg']:.0f} steps when it wins" if ev["rate"] else "")
               + " · showing a representative run.")
    embed(U.render_player(ev["frames"], delay_ms=delay), height=board_h)


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
